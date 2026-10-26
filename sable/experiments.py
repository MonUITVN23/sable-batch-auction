"""All experiments of the paper. Each function returns a JSON-serialisable dict.

Run everything with ``python run_all.py`` (writes ``results/*.json``), then
``python scripts/make_tables.py`` and ``python scripts/make_figures.py``.

Conventions: the protocol grid is 64 ticks {1..64}, the reference tick is 32, the tick
size is 1 price unit unless stated, and the external price at batch time equals the
reference (P_k = p_ref * tick). Expectations are exact over the finite price law;
Monte-Carlo is only a consistency check.
"""

from __future__ import annotations

import itertools
import math
import random
from fractions import Fraction as Fr
from typing import Dict, List, Optional

from .bounds import collar_bound, resilience_bound, signed_bound, tail_bound, tail_bound_loose
from .clearing import clear, clear_reference
from .coupled import Market, run_coupled
from .displacement import displacement, reference_gap, switch_cap, victim_displacement
from .optimizer import ActionSpace, optimise, optimise_leaky, value_of_victim
from .orders import Book, Order, buy, sell
from .pricelaw import BASELINE_ENVELOPE, BASELINE_LAW, Envelope, PriceLaw, estimate_moments
from .serial import serial_match

GRID = list(range(1, 65))
P_REF = 32
SEED = 20260729
MC_DRAWS = 20_000


def fr(x) -> str:
    """Exact rational as a string for JSON."""
    return str(Fr(x))


def fl(x) -> float:
    return float(x)


def market(tick=1, collar=None, **kw) -> Market:
    return Market(GRID, P_REF, Fr(tick), collar=collar, **kw)


# ----------------------------------------------------------------------------------
# Books used throughout
# ----------------------------------------------------------------------------------

def spread_book(depth: int, width: int = 1) -> Book:
    """Honest demand ``depth`` at p_ref - width and supply ``depth`` at p_ref + width.
    No honest trade; the book clears at p_ref by the reference-distance rule."""
    return Book([buy(depth, P_REF - width, "HB"), sell(depth, P_REF + width, "HS")])


def channel_book(near: int = 3, far_tick: int = 54) -> Book:
    """The multi-tick book of the channel sweep (Fig. 2b).

    Strict honest demand 5 at tick 33 is matched by honest supply 5 at 32, so the book
    clears at the reference with volume 5. Near supply ``near`` at 33 sets how much
    adversary demand the imbalance rule tolerates at 32 without relabelling the price,
    and far supply 10 at ``far_tick`` is the bad price a large hidden buy pays when the
    victim is absent. The switch to the clearing-price channel needs the far supply to
    be reached: q > 5 + near.
    """
    return Book([buy(5, 33, "HB33"), sell(5, 32, "HS32"), sell(near, 33, "HS33"), sell(10, far_tick, "HS54")])


CHANNEL_VICTIM = sell(9, P_REF, "VICTIM")       # victim sells 9 at the reference
CHANNEL_TICK = Fr(1, 8)                         # one tick of displacement is worth less than the drift 1/4


def harm_book() -> Book:
    """Honest trade of 5 at the reference; the victim is an aggressive buyer (limit 40)."""
    return Book([buy(5, P_REF, "HB32"), sell(5, P_REF, "HS32")])


HARM_VICTIM = buy(20, 40, "VICTIM")


def leakage_books() -> Dict[str, Book]:
    return {
        "thin": Book([buy(2, 30, "HB"), sell(2, 34, "HS")]),
        "balanced": Book([buy(5, 31, "HB"), sell(5, 33, "HS")]),
        "deep": Book([buy(10, 31, "HB1"), buy(10, 30, "HB2"), sell(10, 33, "HS1"), sell(10, 34, "HS2")]),
    }


# ----------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------

def mc_check(surplus, law: PriceLaw, seed: int, draws: int = MC_DRAWS, z: float = 3.2905) -> Dict:
    """99.9% Monte-Carlo interval for E[G_v] and E[(G_v)^+]; returns whether it contains the exact value."""
    rng = random.Random(seed)
    xs = law.sample(rng, draws)
    g = [float(surplus.G(x)) for x in xs]
    gp = [max(v, 0.0) for v in g]
    out = {}
    for name, vals, exact in (("G", g, surplus.expected_G(law)), ("G_pos", gp, surplus.expected_G_pos(law))):
        m = sum(vals) / draws
        sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (draws - 1))
        se = sd / math.sqrt(draws)
        out[name] = {"mean": m, "se": se, "lo": m - z * se, "hi": m + z * se,
                     "exact": fl(exact), "contains": (m - z * se) <= float(exact) <= (m + z * se)}
    return out


def summarize(res, law: PriceLaw, env: Envelope, q_max: int) -> Dict:
    s = res.surplus
    return {
        "action": None if res.action is None else {"side": res.action.side, "qty": res.action.qty, "limit": res.action.limit},
        "n_actions": res.n_actions,
        "E_pi": fr(res.value), "E_pi_float": fl(res.value),
        "second_best": None if res.second_best is None else fr(res.second_best),
        "dQ": s.dQ, "C_imm": fr(s.C_imm), "dF": fr(s.dF),
        "E_G": fr(s.expected_G(law)), "E_G_pos": fr(s.expected_G_pos(law)),
        "p_plus": s.plus.outcome.price, "p_minus": s.minus.outcome.price,
        "X_plus": s.plus.adversary_position, "X_minus": s.minus.adversary_position,
        "victim_fill": s.victim_fill, "L_v": fr(s.victim_loss),
        "signed_bound": fr(signed_bound(q_max, env)),
        "tail_bound": tail_bound(q_max, env), "tail_bound_loose": tail_bound_loose(q_max, env),
        "identity_ok": s.check_identity(law),
    }


# ----------------------------------------------------------------------------------
# 1. Exposure bounds in three regimes (Table 1)
# ----------------------------------------------------------------------------------

def exp_regimes(seeds=(SEED,)) -> Dict:
    """Three regimes: thin-book (q=12), high-drift (q=12, looser envelope), small-cap (q=3).
    The victim sells into an otherwise empty-at-reference book; dQ_v = min(q_max, victim depth)."""
    law = BASELINE_LAW
    # honest depth 2 on each side, 12 ticks away from the reference: present but not reachable
    # at a profit, so the surplus-optimal policy trades only with the victim, at the reference
    regimes = {
        "thin-book": dict(q_max=12, env=Envelope(Fr(1, 4), Fr(2)), honest=spread_book(2, 12), victim=sell(5, P_REF, "VICTIM")),
        "high-drift": dict(q_max=12, env=Envelope(Fr(1), Fr(2)), honest=spread_book(2, 12), victim=sell(5, P_REF, "VICTIM")),
        "small-cap": dict(q_max=3, env=Envelope(Fr(1, 4), Fr(2)), honest=spread_book(2, 12), victim=sell(1, P_REF, "VICTIM")),
    }
    out = {"law": repr(law), "law_mean": fr(law.mean()), "law_var": fr(law.variance()),
           "objective": "attained exposure: sup over single orders of E[G_v] and of E[(G_v)^+]", "rows": {}}
    for name, r in regimes.items():
        space = ActionSpace(r["q_max"], GRID)
        res = optimise(r["honest"], r["victim"], space, market(), law, objective="G")
        res_pos = optimise(r["honest"], r["victim"], space, market(), law, objective="G_pos")
        row = summarize(res, law, r["env"], r["q_max"])
        row.update({"q_max": r["q_max"], "mu_h": fr(r["env"].mu_h), "sigma_sqrt_h": r["env"].sigma_sqrt_h,
                    "sup_E_G_pos": fr(res_pos.value), "sup_E_G_pos_dQ": res_pos.surplus.dQ,
                    "within_signed": Fr(row["E_G"]) <= Fr(row["signed_bound"]),
                    "within_tail": float(res_pos.value) <= row["tail_bound"],
                    "mc": {str(sd): mc_check(res.surplus, law, sd) for sd in seeds}})
        out["rows"][name] = row
    return out


# ----------------------------------------------------------------------------------
# 2. Tightness: single-victim fixture (Fig. 2a)
# ----------------------------------------------------------------------------------

def exp_tightness(q_max: int = 8, depths=range(0, 13)) -> Dict:
    """Empty honest book, one victim sell of depth d at the reference, adversary buys.
    Drift envelope tight: mu_h = E[dP] = 1/4."""
    law = BASELINE_LAW
    env = Envelope(law.mean(), BASELINE_ENVELOPE.sigma2_h)
    rows = []
    for d in depths:
        victim = sell(d, P_REF, "VICTIM") if d > 0 else None
        res = optimise(Book(), victim, ActionSpace(q_max, GRID), market(), law)
        s = res.surplus
        sb = signed_bound(q_max, env)
        tb, tbl = tail_bound(q_max, env), tail_bound_loose(q_max, env)
        rows.append({"d": d, "dQ": s.dQ, "E_G": fr(s.expected_G(law)), "E_G_pos": fr(s.expected_G_pos(law)),
                     "signed_ratio": fl(s.expected_G(law) / sb) if sb else None,
                     "tail_ratio_sharp": fl(s.expected_G_pos(law)) / tb,
                     "tail_ratio_loose": fl(s.expected_G_pos(law)) / tbl,
                     "C_imm": fr(s.C_imm), "L_v": fr(s.victim_loss)})
    return {"q_max": q_max, "mu_h": fr(env.mu_h), "sigma2_h": fr(env.sigma2_h),
            "E_dP_pos": fr(law.positive_part_mean()), "jensen_gap": {"E_abs": fr(law.abs_mean()), "sqrt_E_sq": math.sqrt(law.second_moment())},
            "signed_bound": fr(signed_bound(q_max, env)), "tail_bound": tail_bound(q_max, env),
            "tail_bound_loose": tail_bound_loose(q_max, env), "rows": rows}


# ----------------------------------------------------------------------------------
# 3. Channel sweep (Fig. 2b) and its robustness
# ----------------------------------------------------------------------------------

def channel_sweep(honest: Book, victim: Order, caps=range(1, 11), tick=1, collar=None,
                  law: PriceLaw = BASELINE_LAW, c_max=None, with_value: bool = True) -> Dict:
    mk = market(tick, collar)
    rows = []
    p_Ev = clear(honest.with_orders(victim), GRID, P_REF, collar=collar).price
    p_E = clear(honest, GRID, P_REF, collar=collar).price
    for q in caps:
        space = ActionSpace(q, GRID, Fr(tick), c_max)
        res = optimise(honest, victim, space, mk, law)
        s = res.surplus
        pi = s.expected_pi(law)
        g = s.expected_G(law)
        c = s.C_imm - s.dF
        share_c = fl(c / pi) if pi != 0 else None
        row = {"q_max": q, "action": None if res.action is None else [res.action.side, res.action.qty, res.action.limit],
               "E_pi": fr(pi), "E_G": fr(g), "C_imm": fr(s.C_imm), "dF": fr(s.dF), "dQ": s.dQ,
               "share_C": share_c, "share_G": None if pi == 0 else fl(g / pi),
               "p_plus": s.plus.outcome.price, "p_minus": s.minus.outcome.price,
               "X_plus": s.plus.adversary_position, "X_minus": s.minus.adversary_position,
               "delta_E": displacement(honest, q, GRID, P_REF, collar),
               "delta_Ev": displacement(honest.with_orders(victim), q, GRID, P_REF, collar),
               "price_relabelled": (s.plus.outcome.price != p_Ev) or (s.minus.outcome.price != p_E),
               "trades_off_reference": (s.plus.adversary_fill > 0 and s.plus.clearing_price != mk.external_price)
                                       or (s.minus.adversary_fill > 0 and s.minus.clearing_price != mk.external_price),
               "L_v": fr(s.victim_loss), "victim_fill": s.victim_fill}
        if with_value:
            vv, ap, am = value_of_victim(honest, victim, space, mk, law)
            row["value_of_victim"] = fr(vv)
            row["rational_action_plus"] = None if ap is None else [ap.side, ap.qty, ap.limit]
            row["rational_action_minus"] = None if am is None else [am.side, am.qty, am.limit]
        rows.append(row)
    q_star = switch_cap(honest.with_orders(victim), GRID, P_REF, max(caps), collar)
    first_c = next((r["q_max"] for r in rows if r["share_C"] is not None and r["share_C"] > 0.5), None)
    return {"tick": fr(tick), "collar": collar, "q_star_delta": q_star, "first_cap_with_C_majority": first_c,
            "Delta_v": victim_displacement(honest, victim, GRID, P_REF, collar),
            "beta": fr(reference_gap(honest, GRID, P_REF, Fr(tick), mk.external_price, collar)), "rows": rows}


def exp_channel() -> Dict:
    return channel_sweep(channel_book(), CHANNEL_VICTIM, tick=CHANNEL_TICK)


def exp_channel_robustness() -> Dict:
    """Three books (near supply 1 / 3 / 6, i.e. far supply reachable from q = 7 / 9 / 12) x six
    tick sizes. Reports the delta threshold, the cap at which the optimal policy switches to
    the clearing-price channel, and the C_imm share at q_max = 10."""
    books = {
        "thin (near 1)": (channel_book(near=1), CHANNEL_VICTIM),
        "baseline (near 3)": (channel_book(near=3), CHANNEL_VICTIM),
        "deep (near 6)": (channel_book(near=6), CHANNEL_VICTIM),
    }
    scales = [Fr(1, 32), Fr(1, 16), Fr(1, 8), Fr(1, 4), Fr(1, 2), Fr(1)]
    out = {"books": {}, "scales": [fr(s) for s in scales]}
    for name, (b, v) in books.items():
        out["books"][name] = {}
        for sc in scales:
            r = channel_sweep(b, v, tick=sc, with_value=False)
            last = r["rows"][-1]
            first_pos_c = next((row["q_max"] for row in r["rows"] if Fr(row["C_imm"]) > 0), None)
            out["books"][name][fr(sc)] = {"q_star_delta": r["q_star_delta"], "switch_cap": r["first_cap_with_C_majority"],
                                          "first_cap_with_positive_C": first_pos_c,
                                          "share_C_at_10": last["share_C"], "C_imm_at_10": last["C_imm"], "E_G_at_10": last["E_G"],
                                          "E_pi_at_10": last["E_pi"], "sign_C_at_10": (Fr(last["C_imm"]) > 0) - (Fr(last["C_imm"]) < 0),
                                          "drift_per_unit": fr(BASELINE_LAW.mean()), "tick_price": fr(sc)}
    return out


def exp_sensitivity_scale() -> Dict:
    """Fixed adversary orders across tick-to-price scales: G_v is scale-free, C_imm tracks the scale."""
    law = BASELINE_LAW
    out = {"fixture_order": "buy 7 @ 32 against a victim sell of 7 at the reference",
           "channel_order": "buy 9 @ 54 on the channel book", "rows": []}
    for sc in [Fr(1, 32), Fr(1, 16), Fr(1, 8), Fr(1, 4), Fr(1, 2), Fr(1)]:
        mk = market(sc)
        s1 = run_coupled(spread_book(2, 12), sell(7, P_REF, "VICTIM"), buy(7, P_REF, "ADV"), mk)
        s2 = run_coupled(channel_book(), CHANNEL_VICTIM, buy(9, 54, "ADV"), mk)
        out["rows"].append({"tick": fr(sc), "fixture_E_G": fr(s1.expected_G(law)), "fixture_C_imm": fr(s1.C_imm),
                            "channel_E_G": fr(s2.expected_G(law)), "channel_C_imm": fr(s2.C_imm),
                            "channel_notional": fr(Fr(9) * 54 * sc)})
    return out


def exp_collateral() -> Dict:
    """Channel book (tick 1/8), q_max = 10: the surplus-optimal policy versus the collateral cap."""
    law = BASELINE_LAW
    rows = []
    for c in [Fr(10), Fr(20), Fr(30), Fr(40), Fr(50), Fr(60), Fr(243, 4), Fr(61), Fr(135, 2), Fr(70), Fr(100)]:
        res = optimise(channel_book(), CHANNEL_VICTIM, ActionSpace(10, GRID, CHANNEL_TICK, c), market(CHANNEL_TICK), law)
        s = res.surplus
        rows.append({"c_max": fr(c), "action": None if res.action is None else [res.action.side, res.action.qty, res.action.limit],
                     "E_pi": fr(res.value), "C_imm": fr(s.C_imm), "E_G": fr(s.expected_G(law)), "p_minus": s.minus.outcome.price,
                     "notional_of_action": None if res.action is None else fr(Fr(res.action.qty) * res.action.limit * CHANNEL_TICK)})
    return {"tick": fr(CHANNEL_TICK), "rows": rows}


# ----------------------------------------------------------------------------------
# 4. Harm sweep and collar (Theorem 7b in action)
# ----------------------------------------------------------------------------------

def exp_harm() -> Dict:
    """Aggressive victim buyer; adversary sells into it. Reports C_imm, L_v and the honest
    sellers' windfall versus the cap, with and without a collar."""
    law = BASELINE_LAW
    out = {"collars": {}}
    for B in [None, 1, 2, 4, 8]:
        r = channel_sweep(harm_book(), HARM_VICTIM, collar=B)
        mk = market(1, B)
        for row in r["rows"]:
            q = row["q_max"]
            row["collar_bound"] = None if B is None else fr(collar_bound(q, 0, B, 1))
            hv = harm_book().with_orders(HARM_VICTIM)
            row["resilience_bound"] = fr(resilience_bound(q, 0, displacement(harm_book(), q, GRID, P_REF, B),
                                                          displacement(hv, q, GRID, P_REF, B),
                                                          victim_displacement(harm_book(), HARM_VICTIM, GRID, P_REF, B), 1))
            # victim loss without the adversary, for the incremental loss
            base = clear(hv, GRID, P_REF, collar=B)
            row["L_v_no_adversary"] = fr(HARM_VICTIM.side * (base.price - P_REF) * base.fill("VICTIM"))
        out["collars"]["none" if B is None else str(B)] = r
    return out


# ----------------------------------------------------------------------------------
# 5. Leakage, assumption boundary, near-binding, matched-input baseline, calibration
# ----------------------------------------------------------------------------------

def exp_leakage(q_max: int = 8) -> Dict:
    """Hidden: sup over single orders (same in both worlds) of E[G_v]; never above q_max*mu_h.
    Leaky: the same supremum over pairs of orders (one per world). The pair takes opposite
    positions, so |dQ_v| can reach 2 q_max and the certificate fails; the excess grows with the
    honest depth that absorbs the no-victim leg."""
    law = BASELINE_LAW
    env = BASELINE_ENVELOPE
    victim = sell(6, P_REF, "VICTIM")
    out = {"q_max": q_max, "signed_bound": fr(signed_bound(q_max, env)), "victim": "sell 6 at the reference", "rows": {}}
    for name, b in leakage_books().items():
        hid = optimise(b, victim, ActionSpace(q_max, GRID), market(), law, objective="G")
        ap, am, leaky = optimise_leaky(b, victim, ActionSpace(q_max, GRID), market(), law)
        out["rows"][name] = {
            "hidden": {"action": [hid.action.side, hid.action.qty, hid.action.limit] if hid.action else None,
                       "E_G": fr(hid.surplus.expected_G(law)), "dQ": hid.surplus.dQ,
                       "within_bound": hid.surplus.expected_G(law) <= signed_bound(q_max, env)},
            "leaky": {"action_plus": [ap.side, ap.qty, ap.limit] if ap else None,
                      "action_minus": [am.side, am.qty, am.limit] if am else None,
                      "E_G": fr(leaky.expected_G(law)), "dQ": leaky.dQ,
                      "within_bound": leaky.expected_G(law) <= signed_bound(q_max, env)},
            "excess_over_bound": fr(leaky.expected_G(law) - signed_bound(q_max, env)),
        }
    return out


def exp_assumption_boundary(q_max: int = 8) -> Dict:
    """Retained envelope mu_h = 1/2, sigma2_h = 9/4 against several true laws on the d=8 fixture."""
    env = Envelope(Fr(1, 2), Fr(9, 4))
    laws = {
        "inside (baseline)": BASELINE_LAW,
        "on the envelope": PriceLaw([(2, Fr(1, 2)), (-1, Fr(1, 2))]),            # mean 1/2, var 9/4
        "violates drift": PriceLaw([(3, Fr(1, 2)), (-1, Fr(1, 2))]),             # mean 1, var 4
        "violates variance only": PriceLaw([(5, Fr(1, 4)), (-1, Fr(3, 4))]),     # mean 1/2, var 27/4
    }
    rows = {}
    for name, law in laws.items():
        res = optimise(Book(), sell(8, P_REF, "VICTIM"), ActionSpace(q_max, GRID), market(), law)
        s = res.surplus
        rows[name] = {"mean": fr(law.mean()), "var": fr(law.variance()), "inside_envelope": env.contains(law),
                      "dQ": s.dQ, "E_G": fr(s.expected_G(law)), "signed_bound": fr(signed_bound(q_max, env)),
                      "E_G_pos": fr(s.expected_G_pos(law)), "tail_bound": tail_bound(q_max, env),
                      "respects_signed": s.expected_G(law) <= signed_bound(q_max, env),
                      "respects_tail": float(s.expected_G_pos(law)) <= tail_bound(q_max, env)}
    return {"q_max": q_max, "mu_h": fr(env.mu_h), "sigma2_h": fr(env.sigma2_h), "rows": rows}


def exp_near_binding(q_max: int = 8) -> Dict:
    law = BASELINE_LAW
    env = Envelope(law.mean(), BASELINE_ENVELOPE.sigma2_h)
    honest = Book([buy(3, 29, "HB"), sell(3, 35, "HS")])
    res = optimise(honest, sell(7, P_REF, "VICTIM"), ActionSpace(q_max, GRID), market(), law, objective="G")
    s = res.surplus
    return {"dQ": s.dQ, "ratio_dQ": fl(Fr(abs(s.dQ), q_max)), "signed_ratio": fl(s.expected_G(law) / signed_bound(q_max, env)),
            "action": [res.action.side, res.action.qty, res.action.limit] if res.action else None}


def exp_matched_input() -> Dict:
    """One contested-supply instance: uniform-price clearing vs serial price-time priority."""
    S = sell(4, 10, "S")
    H = buy(4, 10, "H")
    A = buy(4, 10, "ADV")
    uni = clear([S, H, A], list(range(1, 21)), 10)
    serial = {"".join(o.id[0] for o in perm): serial_match(perm)["ADV"] for perm in itertools.permutations([S, H, A])}
    return {"uniform_fill_ADV": uni.fill("ADV"), "uniform_price": uni.price,
            "serial_fill_by_arrival_order": serial, "serial_fill_set": sorted(set(serial.values()))}


def exp_calibration(n: int = 40_000, seed: int = SEED) -> Dict:
    law = BASELINE_LAW
    rng = random.Random(seed)
    est = estimate_moments(law.sample(rng, n))
    return {"true_mu_h": fl(law.mean()), "true_sigma2_h": fl(law.variance()), "estimated": est}


# ----------------------------------------------------------------------------------
# 6. Implementation audits
# ----------------------------------------------------------------------------------

def random_book(rng: random.Random, n: int, grid=GRID) -> List[Order]:
    return [Order(rng.choice((1, -1)), rng.randint(1, 12), rng.choice(grid), f"o{i}") for i in range(n)]


def exp_sequence_audit(n_books: int = 25, seed: int = SEED, cap: int = 5040) -> Dict:
    rng = random.Random(seed)
    rows = []
    total = 0
    diffs = 0
    for b in range(n_books):
        n = rng.randint(3, 9)
        orders = random_book(rng, n)
        collar = rng.choice([None, None, 2])
        canon = clear(orders, GRID, P_REF, f"b{b}", collar)
        if math.factorial(n) <= cap:
            perms = itertools.permutations(orders)
            mode = "exhaustive"
            count = math.factorial(n)
        else:
            perms = (rng.sample(orders, n) for _ in range(cap))
            mode = "sampled"
            count = cap
        d = 0
        for p in perms:
            if clear(list(p), GRID, P_REF, f"b{b}", collar) != canon:
                d += 1
        total += count
        diffs += d
        rows.append({"book": b, "n": n, "mode": mode, "permutations": count, "differences": d})
    return {"books": n_books, "exhaustive": sum(r["mode"] == "exhaustive" for r in rows),
            "sampled": sum(r["mode"] == "sampled" for r in rows), "permutations": total, "differences": diffs, "rows": rows}


def exp_differential(n_cases: int = 10_000, seed: int = SEED) -> Dict:
    rng = random.Random(seed)
    mism = 0
    e2e = 0
    for i in range(n_cases):
        g = list(range(1, rng.randint(3, 40) + 1))
        pref = rng.choice(g)
        orders = random_book(rng, rng.randint(1, 10), g)
        collar = rng.choice([None, None, 1, 3])
        if clear(orders, g, pref, "d", collar) != clear_reference(orders, g, pref, "d", collar):
            mism += 1
        # end-to-end: the identity pi = G + C - dF holds atom by atom on a coupled run
        if i % 10 == 0:
            victim = Order(rng.choice((1, -1)), rng.randint(1, 10), rng.choice(g), "VICTIM")
            adv = Order(rng.choice((1, -1)), rng.randint(1, 10), rng.choice(g), "ADV")
            mk = Market(g, pref, Fr(1), fee_rate=Fr(1, 1000), collar=collar)
            s = run_coupled(Book(orders), victim, adv, mk)
            if not s.check_identity(BASELINE_LAW):
                e2e += 1
    return {"cases": n_cases, "mismatches": mism, "end_to_end_cases": n_cases // 10, "identity_failures": e2e}


# ----------------------------------------------------------------------------------
# 7. Worked example of Sect. 3 (ten-tick grid)
# ----------------------------------------------------------------------------------

def exp_worked_example() -> Dict:
    grid = list(range(1, 11))
    mk = Market(grid, 5)
    E = Book([buy(3, 5, "B1"), sell(3, 5, "S1")])
    v = buy(4, 8, "V")
    law = PriceLaw([(1, Fr(1, 2)), (-1, Fr(1, 2))])
    adv = sell(1, 8, "ADV")
    rows = {}
    for name, book, collar in [("E", E, None), ("E+A", E.with_orders(adv), None), ("E+v", E.with_orders(v), None),
                               ("E+v+A", E.with_orders(v, adv), None), ("E+v+A collar 1", E.with_orders(v, adv), 1)]:
        o = clear(book, grid, 5, collar=collar)
        rows[name] = {"price": o.price, "fills": o.fills}
    s = run_coupled(E, v, adv, mk)
    opt = optimise(E, v, ActionSpace(2, grid), mk, law)
    optc = optimise(E, v, ActionSpace(2, grid), mk.replace(collar=1), law)
    return {"clearing": rows, "dQ": s.dQ, "C_imm": fr(s.C_imm), "E_pi": fr(s.expected_pi(law)),
            "E_G": fr(s.expected_G(law)), "E_G_pos": fr(s.expected_G_pos(law)), "L_v": fr(s.victim_loss),
            "optimum": {"action": [opt.action.side, opt.action.qty, opt.action.limit], "E_pi": fr(opt.value)},
            "optimum_collar1": {"action": [optc.action.side, optc.action.qty, optc.action.limit], "E_pi": fr(optc.value),
                                "C_imm": fr(optc.surplus.C_imm), "L_v": fr(optc.surplus.victim_loss)},
            "delta_E": displacement(E, 2, grid, 5), "delta_Ev": displacement(E.with_orders(v), 2, grid, 5),
            "Delta_v": victim_displacement(E, v, grid, 5),
            "resilience_bound": fr(resilience_bound(2, 0, displacement(E, 2, grid, 5), displacement(E.with_orders(v), 2, grid, 5), victim_displacement(E, v, grid, 5), 1)),
            "collar_bound_B1": fr(collar_bound(2, 0, 1, 1))}


ALL = {
    "worked_example": exp_worked_example,
    "regimes": lambda: exp_regimes(seeds=(SEED, SEED + 1, SEED + 2, SEED + 3, SEED + 4)),
    "tightness": exp_tightness,
    "channel": exp_channel,
    "channel_robustness": exp_channel_robustness,
    "sensitivity_scale": exp_sensitivity_scale,
    "collateral": exp_collateral,
    "harm": exp_harm,
    "leakage": exp_leakage,
    "assumption_boundary": exp_assumption_boundary,
    "near_binding": exp_near_binding,
    "matched_input": exp_matched_input,
    "calibration": exp_calibration,
    "sequence_audit": exp_sequence_audit,
    "differential": exp_differential,
}
