import copy

import torch


def get_similarity_function(config):
    if config["similarity_function"] == "identity":
        return identity_function
    elif config["similarity_function"] == "cosine":
        return cosine_function
    elif config["similarity_function"] == "norm":
        return norm_function
    elif config["similarity_function"] == "metric":
        return metric_evaluation_function
    elif config["similarity_function"] == "we":
        return we_function
    else:
        raise NotImplementedError(
            "The type of similarity function, "
            + str(config["similarity_function"])
            + ", is not known"
        )


def identity_function(participant, p):
    return 1.0


def cosine_function(participant, p):
    unnormalized_dot = torch.dot(participant.delta_flat, p.delta_flat)

    # normalize (divide) by any
    if participant.config["similarity_weight_normalization"] == "self":
        result = unnormalized_dot / torch.linalg.norm(participant.delta_flat)
    elif participant.config["similarity_weight_normalization"] == "other":
        result = unnormalized_dot / torch.linalg.norm(p.delta_flat)
    elif participant.config["similarity_weight_normalization"] == "both":
        result = unnormalized_dot / (
            torch.linalg.norm(participant.delta_flat) * torch.linalg.norm(p.delta_flat)
        )
    else:
        result = unnormalized_dot

    return result.cpu().numpy()


def norm_function(participant, p):
    # TODO normalize for number of samples / batches
    norm_similarity = 1 / (
        1.0
        + torch.linalg.norm(
            participant.delta_flat - p.delta_flat,
            ord=participant.config["similarity_norm"],
        )
    )
    return norm_similarity.cpu().numpy()


def we_function(participant, p):
    norm_distance = torch.linalg.norm(
        participant.delta_flat - p.delta_flat,
        ord=participant.config["similarity_norm"],
    ) / torch.linalg.norm(
        participant.delta_flat, ord=participant.config["similarity_norm"],
    )
    return norm_distance.cpu().numpy()


def metric_evaluation_function(self_participant, comparison_participant):
    model_state_beginning = copy.deepcopy(self_participant.model.state_dict())

    state_combined = {}

    if self_participant.config["metric_comparison"] == "delta":
        # aggregate the deltas (only consider update)
        for key in self_participant.model.state_dict().keys():
            state_combined[key] = (
                model_state_beginning[key] + comparison_participant.delta[key]
            )
    elif self_participant.config["metric_comparison"] == "model":
        # consider the entire model of comparison participant
        for key in self_participant.model.state_dict().keys():
            state_combined[key] = (
                comparison_participant.model.state_dict()[key]
                + comparison_participant.delta[key]
            )
    elif self_participant.config["metric_comparison"] == "mix":
        # blend between models
        self_mix = 0.75
        for key in self_participant.model.state_dict().keys():
            state_combined[key] = self_mix * (
                model_state_beginning[key] + self_participant.delta[key]
            ) + (1 - self_mix) * (
                comparison_participant.model.state_dict()[key]
                + comparison_participant.delta[key]
            )

    # set state
    self_participant.model.load_state_dict(state_combined)

    # run eval
    eval_result = self_participant.local_evaluate_step(
        -1, self_participant.dataset_loader.train_loader
    )

    loss = eval_result[0]
    ba = eval_result[1]
    f1 = eval_result[2]
    recall = eval_result[3]

    # reset
    self_participant.model.load_state_dict(model_state_beginning)
    return (ba + f1) / 2
