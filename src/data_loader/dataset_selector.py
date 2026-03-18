def load_dataset(name):
    """
    Select and load the specified dataset.
    """
    if name == "cicids":
        from .load_cicids import load
    elif name == "nslkdd":
        from .load_nslkdd import load
    elif name == "unsw":
        from .load_unsw_nb15 import load
    else:
        raise ValueError("Unknown dataset name")
    return load()