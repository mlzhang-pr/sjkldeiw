from typing import Callable

import sys
import os
import traceback
import pdb
import distutils.util

import attrs
import torch
import functools


@attrs.define(kw_only=True)
class FlagsDefinition:
    DEBUG: bool = attrs.field(
        default=distutils.util.strtobool(os.environ.get("QRL_DEBUG", "False")),
        on_setattr=lambda self, field, val: (
            torch.autograd.set_detect_anomaly(val),
            val,
        )[1],
    )


FLAGS = FlagsDefinition()


def pdb_if_DEBUG(fn: Callable):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            fn(*args, **kwargs)
        except:
            exc = sys.exc_info()[1]
            if isinstance(exc, KeyboardInterrupt):
                raise

            if isinstance(exc, SystemExit) and not exc.code:
                raise

            if FLAGS.DEBUG:
                traceback.print_exc()
                print()
                print(" *** Entering post-mortem debugging ***")
                print()
                pdb.post_mortem()
            raise

    return wrapped


__all__ = ["FLAGS", "pdb_if_DEBUG"]
