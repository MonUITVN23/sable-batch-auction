"""Clearing-displacement quantities of Theorem 7 (all in ticks).

delta_E(Q)  = sup_{A: qty(A) <= Q} |p*(E ∪ A) - p*(E)|      (single directional order A)
Delta_v     = |p*(E ∪ {o_v}) - p*(E)|
beta        = |P_k - p*(E)|                                   (price units)
"""

from __future__ import annotations

from fractions import Fraction
from typing import Optional, Sequence

from .clearing import select_price
from .orders import Book, Order


def displacement(book: Book, Q: int, grid: Sequence[int], p_ref: int,
                 collar: Optional[int] = None) -> int:
    base = select_price(book, grid, p_ref, collar)[0]
    worst = 0
    orders = list(book)
    for s in (+1, -1):
        for q in range(1, Q + 1):
            for l in grid:
                p = select_price(orders + [Order(s, q, l, "_probe")], grid, p_ref, collar)[0]
                worst = max(worst, abs(p - base))
    return worst


def switch_cap(book: Book, grid: Sequence[int], p_ref: int, q_limit: int,
               collar: Optional[int] = None) -> Optional[int]:
    """q* = min{Q : delta_E(Q) > 0}, or None if no order of size <= q_limit moves the price."""
    for Q in range(1, q_limit + 1):
        if displacement(book, Q, grid, p_ref, collar) > 0:
            return Q
    return None


def victim_displacement(book: Book, victim: Order, grid: Sequence[int], p_ref: int,
                        collar: Optional[int] = None) -> int:
    base = select_price(book, grid, p_ref, collar)[0]
    with_v = select_price(book.with_orders(victim), grid, p_ref, collar)[0]
    return abs(with_v - base)


def reference_gap(book: Book, grid: Sequence[int], p_ref: int, tick: Fraction,
                  P_k: Fraction, collar: Optional[int] = None) -> Fraction:
    base = select_price(book, grid, p_ref, collar)[0]
    return abs(P_k - Fraction(base) * tick)
