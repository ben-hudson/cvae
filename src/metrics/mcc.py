import numpy as np
import torch

from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

# yanked from https://github.com/ilkhem/icebeem/blob/master/metrics/mcc.py#L391
def mean_corr_coef_np(x, y, method='pearson'):
    """
    A numpy implementation of the mean correlation coefficient metric.

    :param x: numpy.ndarray
    :param y: numpy.ndarray
    :param method: str, optional
            The method used to compute the correlation coefficients.
                The options are 'pearson' and 'spearman'
                'pearson':
                    use Pearson's correlation coefficient
                'spearman':
                    use Spearman's nonparametric rank correlation coefficient
    :return: float
    """
    d = x.shape[1]
    if method == 'pearson':
        cc = np.corrcoef(x, y, rowvar=False)[:d, d:]
    elif method == 'spearman':
        cc = spearmanr(x, y)[0][:d, d:]
    else:
        raise ValueError('not a valid method: {}'.format(method))
    cc = np.abs(cc)
    score = cc[linear_sum_assignment(-1 * cc)].mean()
    return score

# my own implementation
def mcc(latents_hat: torch.Tensor, latents: torch.Tensor):
    latents_hat_np = latents_hat.cpu().numpy()
    latents_np = latents.cpu().numpy()

    # mean correlation coefficient
    # from https://github.com/ilkhem/icebeem/blob/0077f0120c83bcc6d9b199b831485c42bed2401f/metrics/mcc.py#L391
    d = latents_hat_np.shape[1]
    cc = np.corrcoef(latents_hat_np, latents_np, rowvar=False, dtype=np.float32)
    cc = np.abs(cc[:d, d:]) # remove self-correlations
    mcc = cc[linear_sum_assignment(-1 * cc)].mean()

    return mcc
