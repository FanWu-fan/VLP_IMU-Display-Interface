import numpy as np
import GPy
import os
import pickle


def load_gp_models(directory, num_models):
    """
    Loads a specified number of GP models from a directory.

    Parameters:
    -----------
    directory : str
        Directory from which to load the models.
    num_models : int
        Number of models to load.

    Returns:
    --------
    models : list
        List of loaded GP models.
    """
    models = []
    for i in range(num_models):
        filename = os.path.join(directory, f"gp_model_{i}.pkl")
        with open(filename, "rb") as f:
            model = pickle.load(f)
        models.append(model)
    return models


