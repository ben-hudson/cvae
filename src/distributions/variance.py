import torch.distributions as D


def variance(p: D.Distribution):
    # prefer a built-in method if the distribution has one
    if hasattr(p, "_variance") and callable(p._variance):
        return p._variance()
    # otherwise, use implementation for some common distributions
    elif isinstance(p, D.Normal):
        return _variance_normal(p)
    else:
        raise ValueError(f"not sure how to compute the variance of {type(p)}")


def _variance_normal(p: D.Normal):
    return p.scale**2
