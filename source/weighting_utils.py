import numpy as np


def get_weighting_init_function_from_config(config):
    if config["mixing_init"] == "local":
        return get_local_matrix
    elif config["mixing_init"] == "equal":
        return get_equal_matrix
    elif config["mixing_init"] == "sample":
        return get_sample_matrix
    else:
        raise NotImplementedError(
            "The type of process," + str(config["mixing_init"]) + ", is not known"
        )


def get_local_matrix(process):
    n = len(process.fl_dataset)
    matrix = np.identity(n)
    return matrix


def get_equal_matrix(process):
    n = len(process.fl_dataset)
    matrix = np.ones((n, n)) / n
    return matrix


def get_sample_matrix(process):
    n = len(process.fl_dataset)
    sample_sizes = np.array([x.n_samples for x in process.fl_dataset.fl_train_datasets])
    sample_sizes = sample_sizes / sample_sizes.sum()
    matrix = sample_sizes.repeat(n).reshape((n, n)).transpose()
    return matrix


def get_weighting_update_function(config):
    if config["weight_update_method"] == "fixed":
        return fixed_update
    elif config["weight_update_method"] == "similarity":
        return similarity_update
    elif config["weight_update_method"] == "we":
        return we
    else:
        raise NotImplementedError(
            "The type of process, "
            + str(config["weight_update_method"])
            + ", is not known"
        )


def fixed_update(participant, current_weights, all_participants):
    return current_weights


def similarity_update(participant, current_weights, all_participants):
    # assumption: higher value = more similar
    metrics = np.array([participant.get_similarity(p) for p in all_participants])

    # making multiplicative weights of metrics
    if metrics.std() == 0:
        metric_weight = np.ones_like(metrics)
    else:
        metric_weight = ((metrics - metrics.mean()) / metrics.std()) + 1

    # clipping
    metric_weight = metric_weight.clip(0.01, 1.99)

    # mix old and new weights
    memory_factor = participant.config["similarity_memory_factor"]
    weights = (memory_factor * current_weights) + (
        metric_weight / metric_weight.sum()
    ) * (1 - memory_factor)

    return weights


def we(participant, current_weights, all_participants):
    # assumption: higher value = bigger difference
    distances = np.array([participant.get_similarity(p) for p in all_participants])

    # mix old and new weights
    distance_penalty = 1 - participant.config["similarity_memory_factor"]

    # weight erosion update step
    weights = current_weights - distances * distance_penalty

    # scaling
    weights = weights.clip(0)
    weights = weights / weights.sum()

    return weights
