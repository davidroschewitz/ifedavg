import copy

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from typing import Dict, Type

from source.data_handlers import BaseDataLoader, FastDataLoader
from source.fl_datasets import BaseFLDataset
from source.models import BaseModel
from source.participants import BaseParticipant
from source.weighting_utils import get_weighting_init_function_from_config

import source.fl_process_evaluation_utils as process_eval_utils


class BaseFLProcess:
    def __init__(self, config: dict, device: str, writer: SummaryWriter):
        self.config = config
        self.device = device
        self.writer = writer
        self.participants = None
        self.fl_dataset = None
        self.mixing_matrix = None

        self._init_auxilliary_functions()

    def init(
        self,
        participant_class: Type[BaseParticipant],
        fl_dataset: BaseFLDataset,
        model_class: Type[BaseModel],
    ):
        """
        Initializes the actual learning process including the participants,
        dataset & fl_iterator
        """
        self.fl_dataset = fl_dataset
        self.participants = self._init_participants_dataloaders(
            participant_class, fl_dataset, model_class
        )
        self.mixing_matrix = self._get_init_mixing_matrix()

        process_eval_utils.write_n_samples(self)

    def train(self):
        for round_i in range(self.config["n_rounds"]):
            # for each round train and evaluate
            print("Round:", round_i)
            self._train_round(round_i)
            self._evaluate_round(round_i)

    def evaluate(self):
        process_eval_utils.write_mixing_matrix(self)
        # process_eval_utils.visualize_features(self)

        if self.config["visualize_local_weights"]:
            process_eval_utils.write_local_weights(self)

    def _init_participants_dataloaders(
        self,
        participant_class: Type[BaseParticipant],
        fl_dataset: BaseFLDataset,
        model_class: Type[BaseModel],
    ):
        participants = []
        for i in range(len(fl_dataset)):
            name = fl_dataset[i][0]
            participant_data_loader = FastDataLoader(
                self.config, fl_dataset[i][1], fl_dataset[i][2]
            )

            if self.config["hotstart_layers"]:
                norm_params = fl_dataset.participant_normalizations[name]
            else:
                norm_params = None

            class_weights = fl_dataset.class_weights[name]

            p = participant_class(
                index=i,
                name=name,
                config=self.config,
                device=self.device,
                dataset_loader=participant_data_loader,
                model_class=model_class,
                model_init_params={
                    "input_dim": fl_dataset.input_dim,
                    "output_dim": fl_dataset.output_dim,
                    "norm_params": norm_params,
                },
                class_weights=class_weights,
            )
            participants.append(p)

        if self.config["sync_beginning"]:
            reference_i = np.random.choice(len(participants))
            for p in participants:
                p.set_local_model(participants[reference_i].get_local_model())

                # optionally, the optimizers can be synchronized also
                # p.optimizer.load_state_dict(
                #     participants[reference_i].optimizer.state_dict()
                # )

        return participants

    def _train_round(self, round_i):
        # trigger all participants to do the local training step
        train_metrics = []
        for participant in self.participants:
            train_metrics.append(participant.local_train_step(round_i))

        # update mixing matrix for each participant
        for i, participant in enumerate(self.participants):
            self.mixing_matrix[i] = participant.get_updated_mixing_matrix(
                self.mixing_matrix[i], self.participants
            )

        # metrics
        train_metrics = np.array(train_metrics, dtype=object).astype(float).transpose()
        self._write_metrics(round_i, train_metrics, "train")
        self._write_training_round_visualizations(round_i)

        # updates all participants by computing their aggregate
        self._update_participants_with_aggregate(round_i)

    def _update_participants_with_aggregate(self, round_i):
        # computes the aggregate for each participant
        for participant in self.participants:
            aggregate_state = self._get_aggregate_from_delta_of_participant(participant)
            participant.set_local_model(aggregate_state, round_i)

    def _get_aggregate_from_delta_of_participant(
        self, reference_participant: BaseParticipant,
    ) -> Dict[str, torch.Tensor]:
        original_state = reference_participant.get_model_at_beginning()
        aggregate_state = {}
        for key in original_state.keys():
            stacked_deltas = torch.stack(
                [p.delta[key] for p in self.participants], dim=0
            )

            # get weight and reshape
            weight = self.mixing_matrix[reference_participant.index]
            weight = (
                torch.Tensor(weight)
                .view(-1, 1)
                .repeat(1, np.product(stacked_deltas.shape[1:]))
                .reshape(stacked_deltas.shape)
            ).to(self.device)
            weighted_update = torch.sum(stacked_deltas * weight, 0)
            aggregate_state[key] = original_state[key] + weighted_update
        return aggregate_state

    def _write_metrics(self, n, metrics, prefix):
        metric_names = self.config["metrics"]
        for i, name in enumerate(metric_names):
            # add scalar
            self.writer.add_scalar(prefix + "/avg_" + name, metrics[i].mean(), n)

            if metrics.ndim == 2:
                # add distribution
                try:
                    self.writer.add_histogram(prefix + "/" + name, metrics[i], n)
                except:
                    print("issue with histogram")

                # for each participant
                for j, p in enumerate(self.participants):
                    self.writer.add_scalar(
                        prefix + "-indiv-" + str(name) + "/" + str(p.name),
                        metrics[i][j],
                        n,
                    )

    def _evaluate_round(self, round_i):
        test_metrics = [
            participant.local_evaluate_step(round_i)
            for participant in self.participants
        ]

        test_metrics = np.array(test_metrics).transpose()
        self._write_metrics(round_i, test_metrics, "test")

    def _get_init_mixing_matrix(self):
        return self._weight_init_function(self)

    def _init_auxilliary_functions(self):
        self._weight_init_function = get_weighting_init_function_from_config(
            self.config
        )

    def _write_training_round_visualizations(self, round_i):
        self.writer.add_image(
            "mixing-matrix", self.mixing_matrix, dataformats="HW", global_step=round_i
        )

        # analyzing the norm of the updates...
        # for p in self.participants:
        #     self.writer.add_scalar(
        #         "norm/" + p.name, torch.linalg.norm(p.delta_flat), global_step=round_i,
        #     )

        if self.config["visualize_local_weights"]:
            local_keys = [
                x
                for x in self.participants[0]._model.state_dict().keys()
                if x in self.config["local_layers"]
            ]
            for local_key in local_keys:
                list_of_images = []
                for p in self.participants:
                    list_of_images.append(p._model.state_dict()[local_key].cpu())
                img = torch.stack(list_of_images, dim=0).flatten(1)

                # if the weight is multiplicative, the default value is 1 and has to be removed
                if local_key.endswith("_w"):
                    img = img - 1

                # create colored image
                c_img = torch.ones((3, img.shape[0], img.shape[1]))

                # remove to create red...
                c_img[1, :, :] = c_img[1, :, :] - img.clip(-1, 0).abs()
                c_img[2, :, :] = c_img[2, :, :] - img.clip(-1, 0).abs()

                # remove to create green
                c_img[0, :, :] = c_img[0, :, :] - img.clip(0, 1).abs()
                c_img[2, :, :] = c_img[2, :, :] - img.clip(0, 1).abs()

                # write image
                self.writer.add_image(
                    local_key, c_img, dataformats="CHW", global_step=round_i
                )


class GlobalCentralizedFLProcess(BaseFLProcess):
    def __init__(self, config, device, writer):
        super(GlobalCentralizedFLProcess, self).__init__(config, device, writer)
        self.centralized_participant = None

    def init(self, participant_class, fl_dataset, model_class):
        super(GlobalCentralizedFLProcess, self).init(
            participant_class, fl_dataset, model_class
        )

        assert self.config["weight_update_method"] == "fixed"
        assert self.config["participant"] == "base"
        assert self.config["mixing_init"] == "equal"

        # assert that a centralized dataset has been initialized
        assert fl_dataset.centralized_train_dataset is not None

        global_data_loader = BaseDataLoader(
            self.config, fl_dataset.centralized_train_dataset, None, make_test=False
        )

        class_weights = fl_dataset.class_weights["centralized"]

        self.centralized_participant = participant_class(
            index=-1,
            name="centralized",
            config=self.config,
            device=self.device,
            dataset_loader=global_data_loader,
            model_class=model_class,
            model_init_params={
                "input_dim": fl_dataset.input_dim,
                "output_dim": fl_dataset.output_dim,
                "norm_params": None,
            },
            class_weights=class_weights,
        )

        # sets the number of batches to be equal to the federated case
        if "n_batch_limit" in self.config and self.config["n_batch_limit"] is not None:
            self.centralized_participant.n_batches = int(
                self.config["n_batch_limit"]
            ) * len(self.participants)

    def _train_round(self, round_i):
        # only train the centralized "participant"
        train_metrics = self.centralized_participant.local_train_step(round_i)

        # set all participants to central model for evaluation
        for p in self.participants:
            p.set_local_model(self.centralized_participant._model.state_dict(), round_i)

        self._write_metrics(round_i, np.array(train_metrics), "train")


class APFLProcess(BaseFLProcess):
    def __init__(self, config, device, writer):
        super(APFLProcess, self).__init__(config, device, writer)

        assert self.config["weight_update_method"] == "fixed"
        assert self.config["participant"] == "apfl"
        assert self.config["mixing_init"] == "equal"
        assert self.config["optimizer_params"]["momentum"] == 0.0

    def _update_participants_with_aggregate(self, round_i):
        # computes the mean of the global models
        keys = self.participants[0].get_local_model().keys()
        global_model = {}
        for key in keys:
            stacked_model = torch.stack(
                [p.get_local_model()[key] for p in self.participants], dim=0
            )

            global_model[key] = stacked_model.mean(dim=0)

        # sets them for each participant
        for participant in self.participants:
            participant.set_local_model(global_model)
