import numpy as np
import torch

from scipy.optimize import linear_sum_assignment

# yanked from https://github.com/ilkhem/icebeem/blob/master/metrics/mcc.py#L391
def mcc(x: torch.Tensor, y: torch.Tensor) -> float:
    correlations = np.corrcoef(x, y, rowvar=False)
    # multiply by -1 because we want the maximum weight matchings
    # abs because we only care about the strength of the correlation
    assignments = linear_sum_assignment(-1 * np.abs(correlations))
    score = correlations[assignments].mean()

    return score