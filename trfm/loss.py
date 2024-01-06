import torch.nn as nn

def get_loss_function(loss_name):
    """
    Returns the loss function based on the loss name.

    Args:
    loss_name (str): Name of the loss function.

    Returns:
    nn.Module: The loss function.
    """
    if loss_name == "MSELoss":
        return nn.MSELoss()
    # return MAE
    elif loss_name == "MAELoss":
        return nn.L1Loss()
    # return Huber
    elif loss_name == "HuberLoss":
        return nn.HuberLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
