def warmup_fn(step: int, warmup_steps: int) -> float:
    """Learning rate warmup.

    Maps the step to a factor for rescaling the learning rate.
    """
    if warmup_steps:
        return min(1.0, step / warmup_steps)
    else:
        return 1.0


def exp_decay_with_warmup_fn(
    step: int, decay_rate: float, decay_steps: int, warmup_steps: int
) -> float:
    """Decay function for exponential decay with learning rate warmup.

    Maps the step to a factor for rescaling the learning rate.
    """
    factor = warmup_fn(step, warmup_steps)
    return factor * (decay_rate ** (step / decay_steps))