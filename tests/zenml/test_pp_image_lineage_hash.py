import random
import multiprocess
from ..context import model_xray
from model_xray.configs.enums import *
from model_xray.configs.models import *
from model_xray.config_options import *

import pytest

import subprocess

def _mp_run(func, timeout=10):
    ctx = multiprocess.get_context('spawn')
    def wrapper(*args, **kwargs):
        q = ctx.Queue()
        p = ctx.Process(target=func, args=args, kwargs={**kwargs, 'q':q})
        p.start()
        p.join(timeout)

        if p.is_alive():
            p.terminate()
            p.join()

            raise TimeoutError("Process did not finish in time!")

        assert p.exitcode == 0, "Process did not exit cleanly!"

        ret = q.get(block=False)
        assert ret is not None, "Queue is empty!"

        return ret
    return wrapper


def _get_hashes(options):
    return [x.str_hash() for x in options]

@_mp_run
def _get_hashes_from_mc(*, mc_name='famous_le_10m', q=None):
    from model_xray.config_options import get_options_for_mc

    ret = _get_hashes(get_options_for_mc(mc_name))
    q.put(ret)

@_mp_run
def _get_hashes_from_options(*, options=[], q=None):
    ret = _get_hashes(options)
    q.put(ret)

def repeatfunc(func, times=None, **kwargs):
    "Repeat calls to func with specified arguments."
    rets = [None] * times
    for i in range(times):
        rets[i] = func(**kwargs)
    return rets


def test_famous_le_10m_options(chunk_size=250, x=3):
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    options = list(get_options_for_mc('famous_le_10m'))
    for curr_chunk in chunks(options, chunk_size):
        rets = repeatfunc(_get_hashes_from_options, times=x, options=curr_chunk)
        assert all([rets[0] == ret_curr for ret_curr in rets]), "Not all runs are the same!"