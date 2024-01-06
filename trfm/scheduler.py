from torch.optim.lr_scheduler import StepLR

def get_scheduler(optimizer, scheduler_name, **kwargs):
    """
    Returns the learning rate scheduler based on the scheduler name.

    Args:
    optimizer: Optimizer associated with the model parameters.
    scheduler_name (str): Name of the scheduler.
    **kwargs: Additional arguments for the scheduler.

    Returns:
    Learning rate scheduler.
    """
    if scheduler_name == "StepLR":
        return StepLR(optimizer, **kwargs)
    # Add more conditions here for different schedulers
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
