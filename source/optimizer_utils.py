import torch


def _get_params(accepted_params, given_params):
    optimizer_params = {}
    for param in accepted_params:
        if param in given_params:
            optimizer_params[param] = given_params[param]
    return optimizer_params


def get_optimizer(config, model_parameters):
    if config["optimizer"] == "SGD":
        accepted_params = ["lr", "momentum", "weight_decay", "dampening", "nesterov"]
        optimizer_params = _get_params(accepted_params, config["optimizer_params"])
        return torch.optim.SGD(model_parameters, **optimizer_params)
    elif config["optimizer"] == "Adam":
        accepted_params = ["lr", "betas", "eps", "weight_decay", "amsgrad"]
        optimizer_params = _get_params(accepted_params, config["optimizer_params"])
        return torch.optim.Adam(model_parameters, **optimizer_params)
    else:
        raise NotImplementedError(
            "The following optimizer is not supported: " + str(config["optimizer"])
        )
