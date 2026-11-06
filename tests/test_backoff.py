"""Unit tests for ExponentialBackoff reconnect delays."""

from __future__ import annotations

import pytest

from geo_weed_station.network.backoff import ExponentialBackoff


def test_exponential_growth_and_max_cap():
    """Verify monotonic exponential delay growth capped at max_s."""
    backoff = ExponentialBackoff(initial_s=1.0, max_s=10.0, factor=2.0)

    # Sequence should be: 1.0, 2.0, 4.0, 8.0, 10.0, 10.0...
    delays = [backoff.next_delay() for _ in range(7)]

    assert delays[0] == pytest.approx(1.0)
    assert delays[1] == pytest.approx(2.0)
    assert delays[2] == pytest.approx(4.0)
    assert delays[3] == pytest.approx(8.0)
    assert delays[4] == pytest.approx(10.0)
    assert delays[5] == pytest.approx(10.0)
    assert delays[6] == pytest.approx(10.0)


def test_backoff_reset():
    """Verify reset returns backoff delay to initial_s."""
    backoff = ExponentialBackoff(initial_s=0.5, max_s=5.0, factor=3.0)
    assert backoff.next_delay() == pytest.approx(0.5)
    assert backoff.next_delay() == pytest.approx(1.5)

    backoff.reset()
    assert backoff.next_delay() == pytest.approx(0.5)
