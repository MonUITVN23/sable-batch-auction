"""Deterministic uniform-price clearing on a finite tick grid.

Clearing price selection (lexicographic, Sect. 4 of the paper):
  1. maximise executable volume  V(p) = min(D(p), S(p));
  2. minimise imbalance          I(p) = |D(p) - S(p)|;
  3. minimise distance to the precommitted reference tick ``p_ref``;
  4. directional tie rule: prefer the tick above the reference;
  5. smallest remaining tick.

Allocation: strictly in-the-money orders fill before marginal ones; marginal quantity is
rationed pro rata with largest-remainder rounding, remainders ordered by a hash of
(batch id, order id), so the outcome is a deterministic function of the multiset.

An optional *collar* of width ``B`` restricts the candidate ticks to ``|p - p_ref| <= B``.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .orders import Book, Order


@dataclass(frozen=True)
class Outcome:
    price: int
    volume: int
    imbalance: int
    fills: Dict[str, int]          # order id -> filled quantity (>= 0)

    def fill(self, order_id: str) -> int:
        return self.fills.get(order_id, 0)


def _rank_key(batch_id: str, order_id: str) -> str:
    return hashlib.sha256(f"{batch_id}|{order_id}".encode()).hexdigest()


def demand_supply(book: Iterable[Order], p: int) -> Tuple[int, int]:
    d = s = 0
    for o in book:
        if o.side == +1 and o.limit >= p:
            d += o.qty
        elif o.side == -1 and o.limit <= p:
            s += o.qty
    return d, s


def candidate_ticks(grid: Sequence[int], p_ref: int, collar: Optional[int] = None) -> List[int]:
    if collar is None:
        return list(grid)
    return [p for p in grid if abs(p - p_ref) <= collar]


def select_price(book: Iterable[Order], grid: Sequence[int], p_ref: int,
                 collar: Optional[int] = None) -> Tuple[int, int, int]:
    """Return (price, volume, imbalance) under the lexicographic rule."""
    orders = list(book)
    best_key = None
    best = None
    for p in candidate_ticks(grid, p_ref, collar):
        d, s = demand_supply(orders, p)
        v = min(d, s)
        i = abs(d - s)
        # directional tie rule: tick above the reference wins (0 sorts before 1)
        key = (-v, i, abs(p - p_ref), 0 if p >= p_ref else 1, p)
        if best_key is None or key < best_key:
            best_key, best = key, (p, v, i)
    if best is None:
        raise ValueError("empty candidate set")
    return best


def _pro_rata(orders: List[Order], quantity: int, batch_id: str) -> Dict[str, int]:
    """Largest-remainder pro-rata allocation of ``quantity`` among ``orders``."""
    total = sum(o.qty for o in orders)
    if total == 0 or quantity == 0:
        return {o.id: 0 for o in orders}
    if quantity >= total:
        return {o.id: o.qty for o in orders}
    raw = {o.id: Fraction(o.qty * quantity, total) for o in orders}
    floors = {k: math.floor(v) for k, v in raw.items()}
    left = quantity - sum(floors.values())
    # remainders descending; ties broken by the keyed hash so the result is deterministic
    order_ids = sorted(raw, key=lambda k: (-(raw[k] - floors[k]), _rank_key(batch_id, k)))
    for k in order_ids[:left]:
        floors[k] += 1
    return floors


def allocate(book: Iterable[Order], price: int, volume: int, batch_id: str = "batch") -> Dict[str, int]:
    fills: Dict[str, int] = {}
    orders = list(book)
    for side in (+1, -1):
        if side == +1:
            strict = [o for o in orders if o.side == +1 and o.limit > price]
            marginal = [o for o in orders if o.side == +1 and o.limit == price]
        else:
            strict = [o for o in orders if o.side == -1 and o.limit < price]
            marginal = [o for o in orders if o.side == -1 and o.limit == price]
        strict_total = sum(o.qty for o in strict)
        if strict_total >= volume:
            fills.update(_pro_rata(strict, volume, batch_id))
            fills.update({o.id: 0 for o in marginal})
        else:
            fills.update({o.id: o.qty for o in strict})
            fills.update(_pro_rata(marginal, volume - strict_total, batch_id))
    for o in orders:
        fills.setdefault(o.id, 0)
    return fills


def clear(book: Book | Iterable[Order], grid: Sequence[int], p_ref: int,
          batch_id: str = "batch", collar: Optional[int] = None) -> Outcome:
    """Fast clearing: one pass over candidate ticks, then allocation."""
    orders = list(book)
    price, volume, imbalance = select_price(orders, grid, p_ref, collar)
    fills = allocate(orders, price, volume, batch_id)
    _check_conservation(orders, fills, volume)
    return Outcome(price, volume, imbalance, fills)


def clear_reference(book: Book | Iterable[Order], grid: Sequence[int], p_ref: int,
                    batch_id: str = "batch", collar: Optional[int] = None) -> Outcome:
    """Slow, independent reference implementation used by the differential test.

    It enumerates every tick, recomputes demand/supply from scratch with explicit loops,
    sorts candidates with a plain comparison function, and allocates by repeated
    single-unit rounds instead of the closed-form largest-remainder formula.
    """
    orders = list(book)
    cands = []
    for p in grid:
        if collar is not None and abs(p - p_ref) > collar:
            continue
        d = 0
        s = 0
        for o in orders:
            if o.side == +1:
                if o.limit >= p:
                    d = d + o.qty
            else:
                if o.limit <= p:
                    s = s + o.qty
        v = d if d < s else s
        i = d - s if d >= s else s - d
        cands.append((p, v, i))
    # lexicographic selection by explicit pairwise comparison
    best = cands[0]
    for c in cands[1:]:
        if _better(c, best, p_ref):
            best = c
    price, volume, _ = best
    fills = {o.id: 0 for o in orders}
    for side in (+1, -1):
        if side == +1:
            strict = [o for o in orders if o.side == +1 and o.limit > price]
            marginal = [o for o in orders if o.side == +1 and o.limit == price]
        else:
            strict = [o for o in orders if o.side == -1 and o.limit < price]
            marginal = [o for o in orders if o.side == -1 and o.limit == price]
        remaining = volume
        groups = [strict, marginal]
        for g in groups:
            if not g or remaining == 0:
                continue
            total = sum(o.qty for o in g)
            if total <= remaining:
                for o in g:
                    fills[o.id] = o.qty
                remaining -= total
                continue
            # proportional shares, then hand out leftover units by remainder then hash
            shares = {o.id: Fraction(o.qty * remaining, total) for o in g}
            given = {k: int(v) for k, v in shares.items()}
            left = remaining - sum(given.values())
            ranked = sorted(g, key=lambda o: (-(shares[o.id] - given[o.id]), _rank_key(batch_id, o.id)))
            for o in ranked[:left]:
                given[o.id] += 1
            fills.update(given)
            remaining = 0
    return Outcome(price, volume, best[2], fills)


def _better(a: Tuple[int, int, int], b: Tuple[int, int, int], p_ref: int) -> bool:
    pa, va, ia = a
    pb, vb, ib = b
    if va != vb:
        return va > vb
    if ia != ib:
        return ia < ib
    da, db = abs(pa - p_ref), abs(pb - p_ref)
    if da != db:
        return da < db
    above_a, above_b = pa >= p_ref, pb >= p_ref
    if above_a != above_b:
        return above_a
    return pa < pb


def _check_conservation(orders: List[Order], fills: Dict[str, int], volume: int) -> None:
    bought = sum(fills[o.id] for o in orders if o.side == +1)
    sold = sum(fills[o.id] for o in orders if o.side == -1)
    if bought != sold or bought != volume:
        raise AssertionError(f"conservation violated: bought={bought} sold={sold} volume={volume}")
    for o in orders:
        if fills[o.id] < 0 or fills[o.id] > o.qty:
            raise AssertionError(f"overfill on {o.id}")
