import torch.optim as optim

def get_optimizer(optimizer_name, parameters, lr, weight_decay=0):
    """
    Returns the optimizer based on the optimizer name.

    Args:
    optimizer_name (str): Name of the optimizer.
    parameters: Model parameters to optimize.
    lr (float): Learning rate.
    weight_decay (float): Weight decay (L2 penalty) (default: 0).

    Returns:
    optim.Optimizer: The optimizer.
    """
    if optimizer_name == "SGD":
        return optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "SGDM":
        # Assuming momentum is always 0.9 when using SGD with momentum
        return optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == "RMSprop":
        return optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "Adam":
        return optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        # AdamW already includes weight decay handling
        return optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
