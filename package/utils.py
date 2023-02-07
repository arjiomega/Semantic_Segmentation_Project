import json
import numpy as np
import random

def load_dict(filepath):
    """Load a dictionary from a JSON's filepath"""
    with open(filepath, "r") as path_:
        dict_ = json.load(path_)
    return dict_

def save_dict(dict_, filepath, cls=None, sortkeys=False):
    """Save a dictionary to a specific location"""
    with open(filepath, "w") as path_:
        json.dump(dict_, indent=2, fp=path_, cls=cls, sort_keys=sortkeys)

def set_seeds(seed=42):
    """Set seed for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)