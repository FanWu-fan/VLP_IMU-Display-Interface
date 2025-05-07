import numpy as np
import GPy
import os
import pickle
from scipy.optimize import differential_evolution,least_squares


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

# which LEDs to use throughout
using_led = [2,3,4,5]
# extract their 2D coords once
LED_coor = [[8.382,2.910],[8.382,1.080],
            [5.975,2.910],[5.975,1.080],
            [3.561,2.910],[3.561,1.080],
            [1.151,2.910],[1.151,1.080]]
r_S_coords = np.array(LED_coor)[using_led]


def estimate_position_from_g(
    alphas: np.ndarray,
    g_leds: list = [2.2859990974228803, 1.9059483847400323, 1.5702923274815557, 1.1889722180473554],
    r_S_coordinates: np.ndarray = r_S_coords,
    height: float = 2.1555,
    m: float = 0.5,
    bounds: list = [(2, 8), (0, 4)],
    x0: np.ndarray = None  # 新增：初始猜测
) -> np.ndarray:
    exponent = m + 3
    n_led = len(g_leds)

    # 残差向量
    def residuals(r_R):
        x, y = r_R
        # vector 化距离计算
        dx = x - r_S_coordinates[:,0]
        dy = y - r_S_coordinates[:,1]
        d = np.sqrt(dx*dx + dy*dy + height**2)
        alpha_pred = np.array(g_leds) / (d**exponent)
        return alphas - alpha_pred

    # 默认初始点：领域中心
    if x0 is None:
        x0 = np.array([ (bounds[0][0]+bounds[0][1]) / 2,
                        (bounds[1][0]+bounds[1][1]) / 2 ])

    lower = [bounds[0][0], bounds[1][0]]
    upper = [bounds[0][1], bounds[1][1]]

    res = least_squares(
        residuals,
        x0,
        bounds=(lower, upper),
        xtol=1e-3,  # 你可以根据精度需求再调
        ftol=1e-3,
        max_nfev=100  # 限制最多 100 次函数调用
    )
    return res.x