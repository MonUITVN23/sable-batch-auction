import itertools
import random
from fractions import Fraction as Fr

import pytest

from sable import buy, sell, clear
from sable.pricelaw import BASELINE_ENVELOPE, BASELINE_LAW, PriceLaw, estimate_moments
from sable.serial import serial_match
from sable.experiments import exp_worked_example, exp_matched_input


def test_baseline_law_moments():
    assert BASELINE_LAW.mean() == Fr(1, 4)
    assert BASELINE_LAW.variance() == Fr(19, 16)
    assert BASELINE_LAW.positive_part_mean() == Fr(1, 2)
    assert BASELINE_ENVELOPE.contains(BASELINE_LAW)


def test_pricelaw_validation_and_sampling():
    with pytest.raises(ValueError):
        PriceLaw([(1, Fr(1, 2))])
    est = estimate_moments(BASELINE_LAW.sample(random.Random(1), 20000))
    assert abs(est["mu_h"] - 0.25) < 0.03 and abs(est["sigma2_h"] - 1.1875) < 0.1


def test_serial_matcher_price_time_priority():
    S, H, A = sell(4, 10, "S"), buy(4, 10, "H"), buy(4, 10, "A")
    fills = {perm: serial_match(perm)["A"] for perm in itertools.permutations([S, H, A])}
    assert set(fills.values()) == {0, 4}
    assert clear([S, H, A], list(range(1, 21)), 10).fills["A"] == 2


def test_serial_matcher_trades_at_resting_price():
    fills = serial_match([sell(2, 7, "s"), buy(5, 9, "b")])
    assert fills == {"s": 2, "b": 2}
    assert serial_match([buy(5, 6, "b"), sell(2, 7, "s")]) == {"b": 0, "s": 0}


def test_experiment_smoke():
    w = exp_worked_example()
    assert w["C_imm"] == "3" and w["L_v"] == "12"
    m = exp_matched_input()
    assert m["uniform_fill_ADV"] == 2 and m["serial_fill_set"] == [0, 4]
