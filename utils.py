
def move_batch(batch, device):
    """Method to move a batch to the required device

    This is required to deal with the nested dict batch structure

    Args:
        batch (dict): the minibatch of tensors

    Returns:
        dict: the same minibatch but all tensors are on the required device
    """
    out = {}
    for key, value in batch.items():
        if key == "label":
            out[key] = value.to(device)

        else:
            out[key] = {k: v.to(device) for k, v in value.items()}

    return out