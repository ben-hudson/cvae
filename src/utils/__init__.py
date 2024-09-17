from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull
from torch.distributions import Normal

from .val_metrics import *
from .wandb import *


# https://stackoverflow.com/a/52442331/2426888
@contextmanager
def quiet():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


def norm_normal(dist):
    assert isinstance(dist, Normal), f"expected a normal distribution but got {type(dist)}"
    assert dist.loc.dim() == 2, f"expected a 2D tensor (a batch of vectors), but got {dist.loc.dim()}D"
    norm = dist.loc.norm(dim=-1).unsqueeze(-1)
    return Normal(dist.loc / norm, dist.scale / norm)
