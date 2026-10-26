"""Exhaustive search over the adversary's finite action family.

The baseline adversary chooses the null action or one directional order
(side, qty <= q_max, limit on the grid) subject to a collateral cap on notional.
Because the family is finite we enumerate it completely, so every reported optimum is an
attained maximum on the family (a lower bound on the supremum over richer classes).
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterator, List, Optional, Sequence, Tuple

from .coupled import Market, Surplus, run_coupled, run_leaky
from .orders import Book, Order
from .pricelaw import F, PriceLaw

ADV_ID = "ADV"


@dataclass(frozen=True)
class ActionSpace:
    q_max: int
    grid: Sequence[int]
    tick: Fraction = Fraction(1)
    c_max: Optional[Fraction] = None      # collateral cap on notional qty * limit * tick
    sides: Tuple[int, ...] = (+1, -1)

    def actions(self) -> Iterator[Optional[Order]]:
        yield None
        for s in self.sides:
            for q in range(1, self.q_max + 1):
                for l in self.grid:
                    if self.c_max is not None and F(q) * F(l) * self.tick > self.c_max:
                        continue
                    yield Order(s, q, l, ADV_ID)

    def size(self) -> int:
        return sum(1 for _ in self.actions())


@dataclass
class OptResult:
    action: Optional[Order]
    surplus: Surplus
    value: Fraction                        # E[pi_v] at the optimum
    n_actions: int
    second_best: Optional[Fraction] = None  # for the optimality-gap report
    all_values: Optional[List[Tuple[Optional[Order], Fraction]]] = None


def _tie_key(a: Optional[Order]) -> Tuple:
    # deterministic preference among equal-valued actions: null, then small, then low
    return (0,) if a is None else (1, a.qty, a.side, a.limit)


def _pref(a: Optional[Order]) -> Tuple:
    """Higher is preferred among equal-valued actions: null, then smaller quantity, then a
    buy before a sell, then the limit closest to the low end of the grid."""
    return (1,) if a is None else (0, -a.qty, a.side, -a.limit)


def _value(s: Surplus, law: PriceLaw, objective: str) -> Fraction:
    if objective == "pi":
        return s.expected_pi(law)
    if objective == "pi_pos":
        return s.expected_pi_pos(law)
    if objective == "G":
        return s.expected_G(law)
    if objective == "G_pos":
        return s.expected_G_pos(law)
    raise ValueError(objective)


def optimise(honest: Book, victim: Optional[Order], space: ActionSpace, market: Market,
             law: PriceLaw, objective: str = "pi", keep_all: bool = False) -> OptResult:
    """Maximise E[pi_v] ("pi"), E[(pi_v)^+] ("pi_pos"), E[G_v] ("G") or E[(G_v)^+] ("G_pos")
    over the finite action family, with the same order in both worlds."""
    best = None            # (value, pref, action, surplus)
    second: Optional[Fraction] = None
    values = []
    n = 0
    for a in space.actions():
        n += 1
        s = run_coupled(honest, victim, a, market)
        v = _value(s, law, objective)
        if keep_all:
            values.append((a, v))
        if best is None or (v, _pref(a)) > (best[0], best[1]):
            if best is not None:
                second = best[0] if second is None else max(second, best[0])
            best = (v, _pref(a), a, s)
        else:
            second = v if second is None else max(second, v)
    assert best is not None
    return OptResult(best[2], best[3], best[0], n, second, values if keep_all else None)


def optimise_wealth(honest: Book, victim: Optional[Order], space: ActionSpace, market: Market,
                    law: PriceLaw, world: str) -> Tuple[Optional[Order], Fraction]:
    """Maximise expected terminal wealth in one world ("+" with the victim, "-" without)."""
    best = None
    for a in space.actions():
        s = run_coupled(honest, victim, a, market)
        w = s.expected_wealth(law, world)
        if best is None or (w, _pref(a)) > (best[0], best[1]):
            best = (w, _pref(a), a)
    assert best is not None
    return best[2], best[0]


def value_of_victim(honest: Book, victim: Optional[Order], space: ActionSpace, market: Market,
                    law: PriceLaw) -> Tuple[Fraction, Optional[Order], Optional[Order]]:
    """V_v = max_a E[W^{+v}(a)] - max_a E[W^{-v}(a)]: the gain in the adversary's optimal
    value from the victim's presence, for an adversary that is rational in each world."""
    ap, wp = optimise_wealth(honest, victim, space, market, law, "+")
    am, wm = optimise_wealth(honest, victim, space, market, law, "-")
    return wp - wm, ap, am


def optimise_leaky(honest: Book, victim: Optional[Order], space: ActionSpace, market: Market,
                   law: PriceLaw, objective: str = "G") -> Tuple[Optional[Order], Optional[Order], Surplus]:
    """Leakage stress test: drop the same-order constraint of Lemma 3 and maximise the
    objective over pairs (a^{+v}, a^{-v}). For objective "G" (E[G_v] = dQ_v * E[dP]) the
    problem separates into maximising X^+ and minimising X^- (for positive drift)."""
    if objective != "G":
        raise NotImplementedError("only the exposure objective is supported")
    mu = law.mean()
    best_plus = best_minus = None
    for a in space.actions():
        s = run_coupled(honest, victim, a, market)
        vp = s.plus.adversary_position * mu
        vm = -s.minus.adversary_position * mu
        if best_plus is None or (vp, _pref(a)) > (best_plus[0], best_plus[1]):
            best_plus = (vp, _pref(a), a)
        if best_minus is None or (vm, _pref(a)) > (best_minus[0], best_minus[1]):
            best_minus = (vm, _pref(a), a)
    assert best_plus is not None and best_minus is not None
    s = run_leaky(honest, victim, best_plus[2], best_minus[2], market)
    return best_plus[2], best_minus[2], s
