import torch

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull


# silence output to stdout and stderr
# https://stackoverflow.com/a/52442331/2426888
@contextmanager
def hush():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


def norm(batch: torch.Tensor):
    assert batch.dim() == 2, f"expected a 2D tensor (a batch of vectors), but got a {batch.dim()}D one"
    return batch / batch.norm(p=2, dim=-1).unsqueeze(-1)


def norm_normal(batch: torch.distributions.Normal):
    assert isinstance(batch, torch.distributions.Normal), f"expected a normal distribution but got {type(batch)}"
    assert batch.loc.dim() == 2, f"expected a 2D tensor (a batch of vectors), but got {batch.loc.dim()}D one"
    norm = batch.loc.norm(dim=-1).unsqueeze(-1)
    return torch.distributions.Normal(batch.loc / norm, batch.scale / norm)


def is_integer(batch):
    return ((batch == 0) | (batch == 1)).all()
