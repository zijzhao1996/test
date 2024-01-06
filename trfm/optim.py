import torch.optim as optim

def get_optimizer(optimizer_name, parameters, lr):
    """
    Returns the optimizer based on the optimizer name.

    Args:
    optimizer_name (str): Name of the optimizer.
    parameters: Model parameters to optimize.
    lr (float): Learning rate.

    Returns:
    optim.Optimizer: The optimizer.
    """
    if optimizer_name == "SGD":
        return optim.SGD(parameters, lr=lr)
    elif optimizer_name == "SGDM":
        return optim.SGD(parameters, lr=lr, momentum=0.9)
    elif optimizer_name == "RMSprop":
        return optim.RMSprop(parameters, lr=lr)
    elif optimizer_name == "Adam":
        return optim.Adam(parameters, lr=lr)
    elif optimizer_name == "AdamW":
        return optim.AdamW(parameters, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
