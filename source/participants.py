import copy
import math
import os
from typing import Dict, Type, Union, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

import source.evaluation_utils as eval_utils

from source.data_handlers import BaseDataLoader, FastDataLoader
from source.models import BaseModel
from source.optimizer_utils import get_optimizer
from source.similarity_utils import get_similarity_function
from source.weighting_utils import get_weighting_update_function


class BaseParticipant:
    def __init__(
        self,
        index,
        name,
        config,
        device,
        dataset_loader: Union[BaseDataLoader, FastDataLoader],
        model_class: Type[BaseModel],
        model_init_params: dict,
        class_weights: Optional[dict],
    ):
        self.accept_update = True
        self.index = index
        self.name = name
        self.config = config
        self.device = device
        self.dataset_loader = dataset_loader
        self._model = model_class(**model_init_params).to(self.device)
        self.model_state_beginning = {}
        self._optimizer = get_optimizer(self.config, self._model.parameters())
        self._lr_scheduler = StepLR(
            self._optimizer,
            int(self.config["n_rounds"] / self.config["decay_steps"]),
            self.config["decay_mult"],
        )
        self.delta = {}
        self.delta_flat = torch.Tensor()
        self._class_weights = None
        self._logging_stats = pd.DataFrame(
            columns=["name", "round", "n_samples"] + self.config["metrics"]
        )

        # set number of batches to length of dataset loader (= full epoch) or limit
        self.n_batches = len(self.dataset_loader.train_loader)
        if "n_batch_limit" in self.config and self.config["n_batch_limit"] is not None:
            # no looping (i.e. max 1 epoch)
            self.n_batches = min(self.n_batches, int(self.config["n_batch_limit"]))

            # allowed to loop
            # self.n_batches = int(self.config["n_batch_limit"])

        self._class_weights = torch.tensor(class_weights, dtype=torch.float).to(
            self.device, non_blocking=True
        )

        self._init_auxilliary_functions()

    def _init_auxilliary_functions(self):
        self._weight_update_function = get_weighting_update_function(self.config)
        self._similarity_function = get_similarity_function(self.config)

    def get_similarity(self, other_participant):
        return self._similarity_function(self, other_participant)

    def _local_model_batch_step(
        self, data, target, running_stats,
    ):

        data, target = (
            data.to(self.device, non_blocking=True),
            target.to(self.device, non_blocking=True),
        )

        self._optimizer.zero_grad()
        output = self._model(data)
        pred_loss = F.nll_loss(
            output, target, weight=self._class_weights, reduction="sum",
        )
        reg_loss = self._get_regularization_loss()
        loss = pred_loss + reg_loss

        loss.backward()
        self._optimizer.step()
        output = output.detach()

        (train_loss, train_targets, train_predictions, train_probas) = running_stats
        train_loss += loss
        train_targets = torch.cat([train_targets, target])
        train_predictions = torch.cat([train_predictions, output.argmax(dim=1)], dim=0)
        train_probas = torch.cat(
            [train_probas, torch.nn.functional.softmax(output, dim=1)], dim=0
        )

        return (train_loss, train_targets, train_predictions, train_probas)

    def _get_running_stats(self) -> Tuple:
        train_loss = 0
        train_targets = torch.Tensor().to(self.device, non_blocking=True)
        train_predictions = torch.Tensor().to(self.device, non_blocking=True)
        train_probas = torch.Tensor().to(self.device, non_blocking=True)
        return (train_loss, train_targets, train_predictions, train_probas)

    def local_train_step(
        self, round_n, n_epochs: int = 1,
    ):
        self.model_state_beginning = copy.deepcopy(self._model.state_dict())
        self._model.train()

        running_stats = self._get_running_stats()

        for e in range(n_epochs):
            for batch_idx in range(self.n_batches):
                data, target = next(iter(self.dataset_loader.train_loader))

                running_stats = self._local_model_batch_step(
                    data, target, running_stats
                )
        self._lr_scheduler.step()

        self._compute_delta()

        return self._evaluate(running_stats)

    def _compute_delta(self):
        self.delta = {}
        self.delta_flat = torch.Tensor().to(self.device)
        for key in self._model.state_dict().keys():
            self.delta[key] = torch.sub(
                self._model.state_dict()[key], self.model_state_beginning[key]
            )
            self.delta_flat = torch.cat([self.delta_flat, self.delta[key].flatten()])

    def local_evaluate_step(self, round_n, loader=None):
        if loader is None:
            loader = self.dataset_loader.test_loader

        (
            test_loss,
            test_targets,
            test_predictions,
            test_probas,
        ) = self._get_running_stats()
        self._model.eval()
        with torch.no_grad():
            for data, target in loader:
                data, target = (
                    data.to(self.device, non_blocking=True),
                    target.to(self.device, non_blocking=True),
                )
                output = self._model(data)
                loss = F.nll_loss(
                    output, target, weight=self._class_weights, reduction="sum",
                )

                test_loss += loss
                test_targets = torch.cat([test_targets, target])
                test_predictions = torch.cat(
                    [test_predictions, output.argmax(dim=1)], dim=0
                )
                test_probas = torch.cat(
                    [test_probas, torch.nn.functional.softmax(output, dim=1)], dim=0
                )

        eval_stats = self._evaluate(
            (test_loss, test_targets, test_predictions, test_probas)
        )
        self._write_eval_stats(round_n, eval_stats)

        return eval_stats

    def set_local_model(self, new_state: Dict[str, torch.Tensor], round_i: int = 0):
        self._model.load_state_dict(new_state)

    def get_local_model(self) -> Dict[str, torch.Tensor]:
        return self._model.state_dict()

    def get_model_at_beginning(self) -> Dict[str, torch.Tensor]:
        return self.model_state_beginning

    def get_updated_mixing_matrix(self, current_weights, participants):
        new_weights = self._weight_update_function(self, current_weights, participants)
        if (new_weights[self.index]) > (3 * (1 / len(participants))):
            self.accept_update = False
        return new_weights

    def _evaluate(self, running_stats):
        (loss_sum, target, predictions, probas) = running_stats

        true = target.cpu().numpy()
        pred = predictions.cpu().numpy()
        pred_probas = probas.cpu().numpy()

        metrics = []
        for metric in self.config["metrics"]:
            if metric == "loss":
                metrics.append(loss_sum.cpu().detach().numpy() / len(target))
            else:
                metrics.append(eval_utils.get_metric(metric, true, pred, pred_probas))

        return metrics

    def _get_regularization_loss(self):
        return 0.0

    def _write_eval_stats(self, round_n: int, stats: list):
        row = [self.name, round_n, self.dataset_loader.test_loader.n_samples] + stats
        self._logging_stats.loc[round_n] = row

        log_path = self.config["logdir"] + "logs/"
        if round_n <= 1:
            os.makedirs(log_path, exist_ok=True)

        if (round_n + 1) % 5 == 0:
            self._logging_stats.to_csv(
                log_path + str(self.name) + "-log.csv", index=False, sep=";",
            )


class SplitParticipant(BaseParticipant):
    def __init__(self, **args):
        super(SplitParticipant, self).__init__(**args)

        assert self.config["mixing_init"] != "local"
        assert self.config["process"] == "base"

        self._personalized_delta = {}
        self._sharable_param_keys = [
            x
            for x in self._model.state_dict().keys()
            if x not in self.config["local_layers"]
        ]
        self.personalized_update_start_round = self.config[
            "personalized_update_start_round"
        ]
        if self.personalized_update_start_round < 1.0:
            # is a fraction of total rounds
            self.personalized_update_start_round = int(
                self.personalized_update_start_round * self.config["n_rounds"]
            )

        assert self.config["process"] == "base"
        assert self.config["weight_update_method"] == "fixed"

    def get_local_model(self) -> Dict[str, torch.Tensor]:
        sharable_params = {}
        for key in self._sharable_param_keys:
            sharable_params[key] = self._model.state_dict()[key]
        return sharable_params

    def get_model_at_beginning(self) -> Dict[str, torch.Tensor]:
        sharable_params = {}
        for key in self._sharable_param_keys:
            sharable_params[key] = self.model_state_beginning[key]
        return sharable_params

    def set_local_model(self, new_state: Dict[str, torch.Tensor], round_i: int = 0):
        combined_state = {}
        for key in self._model.state_dict().keys():
            if key in self._sharable_param_keys:
                combined_state[key] = new_state[key]
            else:
                if (
                    round_i > self.personalized_update_start_round
                    and key in self._personalized_delta
                ):
                    combined_state[key] = self.model_state_beginning[key] + (
                        self._personalized_delta[key]
                    )
                else:
                    if key in self.model_state_beginning.keys():
                        combined_state[key] = self.model_state_beginning[key]
                    else:
                        combined_state[key] = self._model.state_dict()[key]
        self._model.load_state_dict(combined_state)

    def _compute_delta(self):
        self.delta = {}
        self.delta_flat = torch.Tensor().to(self.device)
        self._personalized_delta = {}
        for key in self._model.state_dict().keys():
            if key in self._sharable_param_keys:
                self.delta[key] = torch.sub(
                    self._model.state_dict()[key], self.model_state_beginning[key]
                )
                self.delta_flat = torch.cat(
                    [self.delta_flat, self.delta[key].flatten()]
                )
            else:
                self._personalized_delta[key] = torch.sub(
                    self._model.state_dict()[key], self.model_state_beginning[key]
                )

    def _get_regularization_loss(self):
        if self.config["reg_multiplier"] == 0.0:
            return 0.0

        def _custom(w: torch.Tensor, exponent=4, stretch=10, increase=2):
            w = w.abs().clip(0.001)
            l = (
                (1 / (w))
                - torch.pow(1 / (w * stretch), exponent)
                - 1
                + increase * (w - 1)
            ) / stretch
            return torch.nansum(torch.clip(l, min=0.0, max=1.0)) / torch.numel(
                l
            )  # torch.mean(torch.clip(l, min=0.0, max=1.0))

        def _gamma(w: torch.Tensor, a=2.0, b=2.0, stretch=4.0):
            assert a >= 2.0
            gamma = (
                np.power(b, a)
                * torch.pow(w, (a - 1))
                * torch.exp(-b * stretch * w)
                * stretch
            )
            return torch.mean(torch.clip(gamma, min=0.0, max=1.0))

        reg = Variable(torch.zeros(1, dtype=torch.float), requires_grad=True).to(
            self.device
        )
        for w in self._model.named_parameters():
            if self.config["reg_type"] == "norm":
                if w[0] not in self._sharable_param_keys:
                    if w[0].endswith("_w"):
                        reg = reg + (w[1].abs() - 1).norm(
                            p=self.config["weight_reg_norm"]
                        )
                    else:
                        reg = reg + (w[1].abs()).norm(p=self.config["weight_reg_norm"])
            elif self.config["reg_type"] == "gamma":
                if w[0] not in self._sharable_param_keys:
                    if w[0].endswith("_w"):
                        reg = reg + _gamma((w[1].abs() - 1))
                    else:
                        reg = reg + _gamma(w[1].abs())
            else:
                if w[0] not in self._sharable_param_keys:
                    if w[0].endswith("_w"):
                        reg = reg + _custom((w[1].abs() - 1))
                    else:
                        reg = reg + _custom(w[1].abs())
        return self.config["reg_multiplier"] * reg.clip(0.0)


class APFLParticipant(BaseParticipant):
    def __init__(self, model_class: Type[BaseModel], model_init_params: dict, **args):
        super(APFLParticipant, self).__init__(
            model_class=model_class, model_init_params=model_init_params, **args
        )

        assert self.config["optimizer"] == "SGD"
        assert self.config["process"] == "apfl"
        assert self.config["reg_multiplier"] == 0.0
        assert self.config["mixing_init"] != "local"

        # mixing parameter (low = local)
        self.alpha = 0.5

        # local model and optimizer (v)
        self.personalized_model_state_beginning = {}
        self._personalized_model = model_class(**model_init_params).to(self.device)
        self._personalized_optimizer = get_optimizer(
            self.config, self._personalized_model.parameters()
        )
        self._personalized_lr_scheduler = StepLR(
            self._personalized_optimizer,
            int(self.config["n_rounds"] / self.config["decay_steps"]),
            self.config["decay_mult"],
        )
        self.personalized_delta = {}
        self.personalized_delta_flat = torch.Tensor()

        # global model parameters:
        self.global_model_params = copy.deepcopy(self._model.state_dict())
        self.personalized_model_params = copy.deepcopy(
            self._personalized_model.state_dict()
        )
        self.mixed_model_params = copy.deepcopy(self._compute_mixed_model())

    def local_train_step(
        self, round_n, n_epochs: int = 1,
    ):
        # set the models
        self._model.load_state_dict(self.global_model_params)
        self._personalized_model.load_state_dict(self.personalized_model_params)

        # copy beginning states
        self.model_state_beginning = copy.deepcopy(self._model.state_dict())
        self.personalized_model_state_beginning = copy.deepcopy(
            self._personalized_model.state_dict()
        )

        running_stats = self._get_running_stats()

        # training of global and local model
        for e in range(n_epochs):
            for batch_idx in range(self.n_batches):
                data, target = next(iter(self.dataset_loader.train_loader))

                # global update step
                self._model.train()
                running_stats = self._local_model_batch_step(
                    data, target, running_stats
                )
                self._model.eval()

                # personalized model update
                self._personalized_model.train()
                self._batch_step_apfl_local(
                    data, target,
                )
                self._personalized_model.eval()
        self._lr_scheduler.step()
        self._personalized_lr_scheduler.step()

        # set global and local model to current state
        self.global_model_params = copy.deepcopy(self._model.state_dict())
        self.personalized_model_params = copy.deepcopy(
            self._personalized_model.state_dict()
        )

        self._compute_delta()

        self.alpha = self._alpha_update()

        # compute and load the mixed model for evaluation
        self.mixed_model_params = self._compute_mixed_model()
        self._model.load_state_dict(self.mixed_model_params)

        return self._evaluate(running_stats)

    def _alpha_update(self):
        lr = self.config["optimizer_params"]["lr"]
        new_alpha = self.alpha - lr * (
            torch.dot(self.model_dif_flat, self.mixed_grad_flat)
        )
        return new_alpha.clip(0, 1)

    def _compute_delta(self):
        # deltas
        self.delta = {}
        self.delta_flat = torch.Tensor().to(self.device)
        self.personalized_delta = {}
        self.personalized_delta_flat = torch.Tensor().to(self.device)

        # custom
        self.model_dif_flat = torch.Tensor().to(self.device)
        for key in self._model.state_dict().keys():
            # for the shared model
            self.delta[key] = torch.sub(
                self._model.state_dict()[key], self.model_state_beginning[key]
            )
            self.delta_flat = torch.cat([self.delta_flat, self.delta[key].flatten()])

            # for the personalized model
            self.personalized_delta[key] = torch.sub(
                self._personalized_model.state_dict()[key],
                self.personalized_model_state_beginning[key],
            )
            self.personalized_delta_flat = torch.cat(
                [self.personalized_delta_flat, self.personalized_delta[key].flatten()]
            )

            # model difference
            self.model_dif_flat = torch.cat(
                [
                    self.model_dif_flat,
                    self.personalized_model_state_beginning[key].flatten()
                    - self.model_state_beginning[key].flatten(),
                ]
            )

        # gradient of mixed model (negative scaled delta of personalized/local model)
        self.mixed_grad_flat = (
            -self.personalized_delta_flat / self.config["optimizer_params"]["lr"]
        )

    def set_local_model(self, new_state: Dict[str, torch.Tensor], round_i: int = 0):
        self.global_model_params = new_state

    def _compute_mixed_model(self):
        mixed_model = {}
        w = self._model.state_dict()
        v = self._personalized_model.state_dict()
        for key in w.keys():
            mixed_model[key] = self.alpha * v[key] + (1 - self.alpha) * w[key]
        return mixed_model

    def _batch_step_apfl_local(
        self, data, target,
    ):
        data, target = (
            data.to(self.device, non_blocking=True),
            target.to(self.device, non_blocking=True),
        )

        self._optimizer.zero_grad()
        self._personalized_optimizer.zero_grad()
        output_global = self._model(data)
        output_personalized = self._personalized_model(data)
        output = self.alpha * output_personalized + (1 - self.alpha) * output_global
        loss = F.nll_loss(output, target, weight=self._class_weights, reduction="sum",)
        loss.backward()
        self._personalized_optimizer.step()

        return True
