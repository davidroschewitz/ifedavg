from typing import Type

import yaml
import os
from datetime import datetime

from source.fl_datasets import (
    BaseFLDataset,
    HumanActivityRecognitionFLDataset,
    VehicleSensorNetworkFLDataset,
)
from source.fl_process import (
    BaseFLProcess,
    GlobalCentralizedFLProcess,
    APFLProcess,
)
from source.models import (
    BaseModel,
    MLPModel,
)
from source.participants import (
    BaseParticipant,
    SplitParticipant,
    APFLParticipant,
)


def init_logdir(config):
    date_str = datetime.now().strftime("%m-%d-%H-%M-%S")
    logdir = (
        "./outputs/"
        + config["dataset"]
        + "/"
        + config["experiment_name"]
        + "/"
        + date_str
        + "-"
        + config["run_name"]
        + "/"
    )

    os.makedirs(logdir, exist_ok=True)
    with open(logdir + "config.yml", "w") as file:
        yaml.dump(config, file)

    config["logdir"] = logdir
    return logdir


def get_dataset_class_from_config(config: dict) -> Type[BaseFLDataset]:
    if config["dataset"] == "har":
        return HumanActivityRecognitionFLDataset
    elif config["dataset"] == "vehicle":
        return VehicleSensorNetworkFLDataset
    else:
        raise NotImplementedError("The dataset is not known: " + str(config["dataset"]))


def get_model_class_from_config(config: dict) -> Type[BaseModel]:
    if config["model"] == "MLPModel":
        return MLPModel
    else:
        raise NotImplementedError("The model is not known: " + str(config["model"]))


def get_process_class_from_config(config: dict) -> Type[BaseFLProcess]:
    if config["process"] == "base":
        return BaseFLProcess
    elif config["process"] == "centralized":
        return GlobalCentralizedFLProcess
    elif config["process"] == "apfl":
        return APFLProcess
    else:
        raise NotImplementedError("The process is not known: " + str(config["process"]))


def load_config_file(path):
    with path.open(mode="r") as yamlfile:
        return yaml.safe_load(yamlfile)


def get_participant_class_from_config(config):
    if config["participant"] == "base":
        return BaseParticipant
    elif config["participant"] == "split":
        return SplitParticipant
    elif config["participant"] == "apfl":
        return APFLParticipant
    else:
        raise NotImplementedError(
            "The type of participant unknown: " + str(config["participant"])
        )
