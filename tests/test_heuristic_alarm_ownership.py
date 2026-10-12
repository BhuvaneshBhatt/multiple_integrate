"""Application alarm ownership and robust heuristic interruption."""

import gc
import signal
import time

import pytest

from multiple_integrate.recognition import _run_with_signal_timeout, _signal_timeout_ready

timer_only = pytest.mark.skipif(
    not _signal_timeout_ready(), reason="POSIX main-thread timer unavailable"
)


@pytest.mark.parametrize("interval", [0, 0.1])
@timer_only
def test_existing_application_alarm_is_never_replaced(interval):
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)

    def handler(*args):
        return None

    signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, 5, interval)
    try:
        called = []
        assert _run_with_signal_timeout(lambda: called.append(True), 0.01, "declined") == "declined"
        assert not called
        assert signal.getsignal(signal.SIGALRM) is handler
        delay, repeat = signal.getitimer(signal.ITIMER_REAL)
        assert 4 < delay <= 5 and repeat == interval
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)


@timer_only
def test_timeout_cannot_be_swallowed_by_an_exception_handler():
    def heuristic():
        try:
            time.sleep(0.1)
        except Exception:
            return "wrong success"
        return "late success"

    assert _run_with_signal_timeout(heuristic, 0.01, "expired") == "expired"


@timer_only
def test_swallowed_base_exception_receives_another_interruption():
    def heuristic():
        try:
            time.sleep(0.1)
        except BaseException:
            time.sleep(0.1)
        return "late success"

    assert _run_with_signal_timeout(heuristic, 0.01, "expired") == "expired"


@timer_only
def test_late_return_after_swallowed_alarm_is_not_success():
    def heuristic():
        try:
            time.sleep(0.1)
        except BaseException:
            return "late success"

    assert _run_with_signal_timeout(heuristic, 0.01, "expired") == "expired"


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("ending", ["success", "timeout", "error"])
@timer_only
def test_gc_handler_and_timer_state_restored(enabled, ending):
    initial = gc.isenabled()
    old_handler = signal.getsignal(signal.SIGALRM)
    (gc.enable if enabled else gc.disable)()

    def heuristic():
        assert not gc.isenabled()
        if ending == "timeout":
            time.sleep(0.1)
        if ending == "error":
            raise ValueError("backend error")
        return 7

    try:
        if ending == "error":
            with pytest.raises(ValueError, match="backend error"):
                _run_with_signal_timeout(heuristic, 0.01, None)
        else:
            assert _run_with_signal_timeout(heuristic, 0.01, None) == (
                None if ending == "timeout" else 7
            )
        assert gc.isenabled() is enabled
        assert signal.getsignal(signal.SIGALRM) is old_handler
        assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
    finally:
        (gc.enable if initial else gc.disable)()


@pytest.mark.parametrize("kind", ["finite_part", "pv"])
def test_collapsed_regulator_only_marks_its_own_parameter_cell_unsupported(kind):
    import sympy as sp

    from multiple_integrate import (
        HadamardFinitePart,
        HypersurfacePrincipalValue,
        IntegrationStatus,
        ParameterCell,
        integrate_cells,
    )

    x, y, a = sp.symbols("x y a", real=True)
    options = (
        {"regularization": HadamardFinitePart(hypersurface=a * x * y)}
        if kind == "finite_part"
        else {"principal_value": HypersurfacePrincipalValue(a * x * y)}
    )
    result = integrate_cells(
        1 / (x * y) ** 2 if kind == "finite_part" else 1 / (x * y),
        (x, -1, 1),
        (y, -1, 1),
        parameters=(a,),
        assumptions=a >= 0,
        cells=(ParameterCell(a > 0), ParameterCell(sp.Eq(a, 0))),
        **options,
    )
    children = result.diagnostics["cell_results"]
    assert children[0].value == (4 if kind == "finite_part" else 0)
    assert children[0].complete is True
    assert children[1].status is IntegrationStatus.UNSUPPORTED
    assert children[1].complete is False
    assert result.complete is False and result.coverage[0].complete is True
    assert "identically zero" in children[1].limitations[0]


@pytest.mark.parametrize("kind", ["finite_part", "pv"])
def test_undefined_regulator_is_retained_on_its_exceptional_cell(kind):
    import sympy as sp

    from multiple_integrate import (
        HadamardFinitePart,
        HypersurfacePrincipalValue,
        IntegrationStatus,
        ParameterCell,
        integrate_cells,
    )

    x, y, a = sp.symbols("x y a", real=True)
    options = (
        {"regularization": HadamardFinitePart(hypersurface=x * y / a)}
        if kind == "finite_part"
        else {"principal_value": HypersurfacePrincipalValue(x * y / a)}
    )
    result = integrate_cells(
        1 / (x * y),
        (x, -1, 1),
        (y, -1, 1),
        parameters=(a,),
        cells=(ParameterCell(sp.Eq(a, 0)),),
        **options,
    )
    child = result.diagnostics["cell_results"][0]
    assert child.status is IntegrationStatus.UNSUPPORTED
    assert "undefined or nonfinite" in child.limitations[0]
    assert result.complete is False
