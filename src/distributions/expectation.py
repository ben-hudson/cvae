import torch.distributions as D


def expectation(p: D.Distribution):
    # prefer a built-in method if the distribution has one
    if hasattr(p, "_expectation") and callable(p._expectation):
        return p._expectation()
    # otherwise, use implementation for some common distributions
    elif isinstance(p, D.Normal) and not issubclass(p, D.Normal):
        return _expectation_normal(p)
    else:
        raise ValueError(f"not sure how to compute the expecation of {type(p)}")


def _expectation_normal(p: D.Normal):
    return p.loc
