import itertools
import random
from fractions import Fraction as Fr

import pytest

from sable import Book, Order, buy, sell, clear, clear_reference
from sable.clearing import select_price

GRID = list(range(1, 11))


def test_empty_book_clears_at_reference():
    out = clear([], GRID, 5)
    assert out.price == 5 and out.volume == 0 and out.fills == {}


def test_single_cross_at_reference():
    out = clear([buy(3, 5, "b"), sell(3, 5, "s")], GRID, 5)
    assert (out.price, out.volume, out.imbalance) == (5, 3, 0)
    assert out.fills == {"b": 3, "s": 3}


def test_volume_beats_imbalance_beats_distance():
    # tick 8 is the only tick with volume 4, although it is far from the reference
    orders = [buy(3, 5, "b"), sell(3, 5, "s"), buy(4, 8, "v"), sell(1, 8, "a")]
    assert select_price(orders, GRID, 5)[0] == 8


def test_distance_rule_and_directional_tie():
    # no trade anywhere; imbalance zero on ticks 3..7; closest to the reference 5 is 5
    assert select_price([buy(1, 2, "b"), sell(1, 8, "s")], GRID, 5)[0] == 5
    # symmetric tie between 4 and 6: the tick above the reference wins
    orders = [buy(2, 4, "b"), sell(2, 6, "s"), buy(1, 5, "c")]
    p = select_price(orders, GRID, 5)[0]
    assert p in (5, 6)  # 5 has imbalance 1, 6 has imbalance 2 -> 5
    assert p == 5


def test_strict_before_marginal_and_pro_rata_rounding():
    # buyers: strict 5@6, marginal 4@5 and 2@5; supply 7@5 -> volume 7 at tick 5
    out = clear([buy(5, 6, "strict"), buy(4, 5, "m1"), buy(2, 5, "m2"), sell(7, 5, "s")], GRID, 5)
    assert out.price == 5 and out.volume == 7
    assert out.fills["strict"] == 5
    assert out.fills["m1"] + out.fills["m2"] == 2
    assert out.fills["m1"] >= out.fills["m2"]  # larger order gets at least as much


def test_no_overfill_and_conservation_random():
    rng = random.Random(7)
    for _ in range(500):
        g = list(range(1, rng.randint(2, 20) + 1))
        orders = [Order(rng.choice((1, -1)), rng.randint(1, 9), rng.choice(g), f"o{i}") for i in range(rng.randint(1, 8))]
        out = clear(orders, g, rng.choice(g))
        for o in orders:
            assert 0 <= out.fills[o.id] <= o.qty
        assert sum(out.fills[o.id] for o in orders if o.side == 1) == out.volume
        assert sum(out.fills[o.id] for o in orders if o.side == -1) == out.volume


def test_differential_fast_vs_reference():
    rng = random.Random(11)
    for _ in range(2000):
        g = list(range(1, rng.randint(3, 30) + 1))
        orders = [Order(rng.choice((1, -1)), rng.randint(1, 12), rng.choice(g), f"o{i}") for i in range(rng.randint(1, 9))]
        collar = rng.choice([None, 1, 2])
        pref = rng.choice(g)
        assert clear(orders, g, pref, "b", collar) == clear_reference(orders, g, pref, "b", collar)


def test_sequence_invariance_exhaustive():
    orders = [buy(3, 5, "b"), sell(2, 5, "s1"), sell(2, 4, "s2"), buy(1, 7, "v"), sell(1, 6, "a")]
    canon = clear(orders, GRID, 5)
    for perm in itertools.permutations(orders):
        assert clear(list(perm), GRID, 5) == canon


def test_collar_restricts_candidates():
    orders = [buy(3, 5, "b"), sell(3, 5, "s"), buy(4, 8, "v"), sell(1, 8, "a")]
    assert clear(orders, GRID, 5).price == 8
    assert clear(orders, GRID, 5, collar=1).price == 6
    assert clear(orders, GRID, 5, collar=1).fills["a"] == 0


def test_rejects_duplicate_ids_and_bad_orders():
    with pytest.raises(ValueError):
        Book([buy(1, 1, "x"), sell(1, 1, "x")])
    with pytest.raises(ValueError):
        Order(0, 1, 1, "z")
    with pytest.raises(ValueError):
        Order(1, 0, 1, "z")
