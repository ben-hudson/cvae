from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull

from .val_metrics import *
from .wandb import *


# https://stackoverflow.com/a/52442331/2426888
@contextmanager
def quiet():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)
