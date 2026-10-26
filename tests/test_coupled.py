from fractions import Fraction as Fr

from sable import (ActionSpace, Book, PriceLaw, buy, sell, run_coupled, optimise, optimise_leaky,
                   value_of_victim, displacement, victim_displacement, collar_bound, resilience_bound,
                   signed_bound, tail_bound, tail_bound_loose)
from sable.coupled import Market
from sable.pricelaw import BASELINE_ENVELOPE, BASELINE_LAW, Envelope

GRID = list(range(1, 11))
MK = Market(GRID, 5)
E = Book([buy(3, 5, "B1"), sell(3, 5, "S1")])
V = buy(4, 8, "V")
LAW = PriceLaw([(1, Fr(1, 2)), (-1, Fr(1, 2))])


def test_worked_example_numbers():
    s = run_coupled(E, V, sell(1, 8, "ADV"), MK)
    assert (s.plus.outcome.price, s.minus.outcome.price) == (8, 5)
    assert s.dQ == -1 and s.C_imm == 3 and s.dF == 0
    assert s.expected_pi(LAW) == 3 and s.expected_G(LAW) == 0 and s.expected_G_pos(LAW) == Fr(1, 2)
    assert s.victim_fill == 4 and s.victim_loss == 12
    assert s.check_identity(LAW)


def test_worked_example_optimum_and_collar():
    opt = optimise(E, V, ActionSpace(2, GRID), MK, LAW)
    assert opt.value == 3
    optc = optimise(E, V, ActionSpace(2, GRID), MK.replace(collar=1), LAW)
    assert optc.value == 1 and optc.surplus.C_imm == 1
    assert displacement(E, 2, GRID, 5) == 0
    assert displacement(E.with_orders(V), 2, GRID, 5) == 2
    assert victim_displacement(E, V, GRID, 5) == 1
    assert resilience_bound(2, 0, 0, 2, 1, 1) == 6 >= 3
    assert collar_bound(2, 0, 1, 1) == 6 >= 1


def test_proposition_impossibility_construction():
    for N in (10, 20, 40):
        g = list(range(1, N + 1))
        s = run_coupled(Book([sell(10, N - 1, "H")]), sell(9, 1, "V"), buy(9, N, "ADV"), Market(g, 1))
        assert s.dQ == 0 and s.C_imm == 9 * (N - 2) and s.victim_loss == 0


def test_identity_with_fees_random():
    import random
    rng = random.Random(3)
    for _ in range(300):
        g = list(range(1, rng.randint(3, 30) + 1))
        orders = [buy(rng.randint(1, 9), rng.choice(g), f"b{i}") for i in range(rng.randint(0, 4))]
        orders += [sell(rng.randint(1, 9), rng.choice(g), f"s{i}") for i in range(rng.randint(0, 4))]
        v = sell(rng.randint(1, 9), rng.choice(g), "V") if rng.random() < 0.5 else buy(rng.randint(1, 9), rng.choice(g), "V")
        a = sell(rng.randint(1, 9), rng.choice(g), "A") if rng.random() < 0.5 else buy(rng.randint(1, 9), rng.choice(g), "A")
        mk = Market(g, rng.choice(g), Fr(rng.choice([1, 2, 4]), 8), fee_rate=Fr(1, 200), unwind_cost=Fr(1, 10),
                    collar=rng.choice([None, 2]))
        s = run_coupled(Book(orders), v, a, mk)
        assert s.check_identity(BASELINE_LAW)


def test_induced_inventory_bounded_by_cap():
    import random
    rng = random.Random(5)
    for _ in range(300):
        g = list(range(1, 21))
        orders = [buy(rng.randint(1, 9), rng.choice(g), f"b{i}") for i in range(2)] + [sell(rng.randint(1, 9), rng.choice(g), f"s{i}") for i in range(2)]
        v = sell(rng.randint(1, 9), rng.choice(g), "V")
        q = rng.randint(1, 8)
        a = buy(q, rng.choice(g), "A") if rng.random() < 0.5 else sell(q, rng.choice(g), "A")
        s = run_coupled(Book(orders), v, a, Market(g, 10))
        assert abs(s.dQ) <= q


def test_exposure_bounds_hold_on_fixture():
    env = Envelope(BASELINE_LAW.mean(), BASELINE_ENVELOPE.sigma2_h)
    grid = list(range(1, 65))
    for d in range(0, 13):
        victim = sell(d, 32, "V") if d else None
        res = optimise(Book(), victim, ActionSpace(8, grid), Market(grid, 32), BASELINE_LAW, objective="G")
        assert res.surplus.expected_G(BASELINE_LAW) <= signed_bound(8, env)
        assert float(res.surplus.expected_G_pos(BASELINE_LAW)) <= tail_bound(8, env) <= tail_bound_loose(8, env)
        if d >= 8:
            assert res.surplus.expected_G(BASELINE_LAW) == signed_bound(8, env)  # attained


def test_sharp_tail_bound_is_valid_for_any_two_moment_law():
    # E[(X)^+] <= (m + sqrt(m^2 + var)) / 2 for a few laws
    for law in (BASELINE_LAW, PriceLaw([(2, Fr(1, 2)), (-1, Fr(1, 2))]), PriceLaw([(5, Fr(1, 4)), (-1, Fr(3, 4))])):
        m, v = float(law.mean()), float(law.variance())
        assert float(law.positive_part_mean()) <= (m + (m * m + v) ** 0.5) / 2 + 1e-12


def test_leaky_pair_can_exceed_single_order_exposure():
    grid = list(range(1, 65))
    honest = Book([buy(10, 31, "HB"), sell(10, 33, "HS")])
    victim = sell(6, 32, "V")
    hid = optimise(honest, victim, ActionSpace(8, grid), Market(grid, 32), BASELINE_LAW, objective="G")
    _, _, leaky = optimise_leaky(honest, victim, ActionSpace(8, grid), Market(grid, 32), BASELINE_LAW)
    assert hid.surplus.expected_G(BASELINE_LAW) <= signed_bound(8, BASELINE_ENVELOPE)
    assert leaky.expected_G(BASELINE_LAW) > signed_bound(8, BASELINE_ENVELOPE)
    assert abs(leaky.dQ) > 8


def test_value_of_victim_never_exceeds_sup_pi():
    vv, _, _ = value_of_victim(E, V, ActionSpace(2, GRID), MK, LAW)
    assert vv <= optimise(E, V, ActionSpace(2, GRID), MK, LAW).value
