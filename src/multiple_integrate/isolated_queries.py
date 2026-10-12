"""Opt-in process isolation for individual expensive backend calls."""

from __future__ import annotations

import importlib
import multiprocessing as mp

from sympy.polys.polyerrors import BasePolynomialError

from .planner import record_strategy

_ERRORS = (
    ArithmeticError,
    ValueError,
    TypeError,
    NotImplementedError,
    ImportError,
    AttributeError,
    RuntimeError,
    BasePolynomialError,
)


def _worker(connection, module, name, args, kwargs):
    try:
        try:
            value = getattr(importlib.import_module(module), name)(*args, **kwargs)
            connection.send((True, value))
        except _ERRORS as error:
            connection.send((False, error))
    finally:
        connection.close()


def isolated_query(module, name, args, kwargs, seconds):
    """Return None on timeout/crash, otherwise return or raise the backend response.

    Spawn overhead counts toward the per-call deadline. No thread is abandoned.
    """
    context = mp.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(send, module, name, args, kwargs), daemon=True)
    started = False
    try:
        process.start()
        started = True
        send.close()
        if not receive.poll(seconds):
            record_strategy(
                "backend_deadline",
                succeeded=False,
                reason=f"{module}.{name} exceeded {seconds} seconds",
            )
            return None
        try:
            ok, value = receive.recv()
        except EOFError:
            record_strategy(
                "backend_deadline",
                succeeded=False,
                reason="worker exited without a response",
            )
            return None
        if not ok:
            raise value
        return value
    finally:
        receive.close()
        send.close()
        if started:
            if process.is_alive():
                process.terminate()
            process.join(timeout=1)
            if process.is_alive():
                process.kill()
                process.join(timeout=1)
            process.close()
