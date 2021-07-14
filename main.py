import argparse
import copy
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter

import source.experiment_utils as experiment_utils


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=lambda p: Path(p).absolute(),
        help="Path to the config yaml file",
        required=True,
    )
    parser.add_argument("--gpu", action="store_true")
    return parser.parse_args()


def fix_seeds(config, device):
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.random.manual_seed(config["seed"])
    if device == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
        torch.cuda.manual_seed(config["seed"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_experiment(config, logdir=None):
    print("Run: ", config["run_name"])
    if logdir is None:
        logdir = experiment_utils.init_logdir(config)

    # sets device
    if args.gpu:
        # cudnn.benchmark = True
        torch.cuda.set_device(0)
        device = "cuda"
    else:
        device = "cpu"

    fix_seeds(config, device)

    writer = SummaryWriter(log_dir=logdir)

    dataset_class = experiment_utils.get_dataset_class_from_config(config)
    model_class = experiment_utils.get_model_class_from_config(config)
    process_class = experiment_utils.get_process_class_from_config(config)
    participant_class = experiment_utils.get_participant_class_from_config(config)

    process = process_class(config, device, writer)
    dataset = dataset_class(config)
    process.init(participant_class, dataset, model_class)
    process.train()
    process.evaluate()

    writer.close()


def run_seeded_experiment(config):
    logdir = experiment_utils.init_logdir(config)
    for seed in config["seed"]:
        sub_config = copy.deepcopy(config)
        sub_config["seed"] = seed
        sub_config["run_name"] = sub_config["run_name"] + "-s" + str(seed)

        sub_logdir = logdir + "/" + str(seed) + "/"
        os.makedirs(sub_logdir, exist_ok=True)
        sub_config["logdir"] = sub_logdir

        run_experiment(sub_config, logdir=sub_logdir)


if __name__ == "__main__":
    args = get_arguments()

    # loads config file
    config = experiment_utils.load_config_file(args.config_path)

    if type(config["seed"]) == int:
        run_experiment(config)
    elif type(config["seed"]) == list:
        run_seeded_experiment(config)
