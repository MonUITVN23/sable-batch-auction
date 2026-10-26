"""Limit orders and eligible multisets.

An eligible order is ``(side, qty, limit, id)``: side +1 buys, -1 sells; ``qty`` is an
integer number of atomic base units; ``limit`` is an integer tick on the protocol grid.
The ciphertext, collateral and nonce of the paper are not needed by the clearing
arithmetic and are modelled only through the caps in :mod:`sable.optimizer`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True, order=True)
class Order:
    side: int
    qty: int
    limit: int
    id: str

    def __post_init__(self) -> None:
        if self.side not in (+1, -1):
            raise ValueError("side must be +1 (buy) or -1 (sell)")
        if self.qty <= 0:
            raise ValueError("qty must be positive")


def buy(qty: int, limit: int, id: str) -> Order:
    return Order(+1, qty, limit, id)


def sell(qty: int, limit: int, id: str) -> Order:
    return Order(-1, qty, limit, id)


class Book:
    """An eligible multiset of orders. Order of insertion is irrelevant to clearing."""

    def __init__(self, orders: Iterable[Order] = ()):
        self.orders: List[Order] = list(orders)
        ids = [o.id for o in self.orders]
        if len(ids) != len(set(ids)):
            raise ValueError("order ids must be unique within a book")

    def __iter__(self):
        return iter(self.orders)

    def __len__(self) -> int:
        return len(self.orders)

    def with_orders(self, *extra: Order) -> "Book":
        return Book(self.orders + list(extra))

    def ids(self) -> List[str]:
        return [o.id for o in self.orders]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Book({self.orders!r})"
