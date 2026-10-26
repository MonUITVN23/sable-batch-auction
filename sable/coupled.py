"""Coupled victim / no-victim worlds and exact terminal-wealth accounting.

World W^{+v} clears E ∪ {o_v} ∪ A, world W^{-v} clears E ∪ A, with the same adversary
order A (or none) in both. For a realised external increment dP the adversary's
victim-attributable surplus is

    pi_v(dP) = W^{A,+v} - W^{A,-v} = G_v(dP) + C_imm - dF_v,

with G_v = dQ_v * dP, dQ_v = X^+ - X^-, C_imm = (P_k - p^+) X^+ - (P_k - p^-) X^-,
and dF_v the fee difference. All quantities are exact Fractions.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, Sequence

from .clearing import Outcome, clear
from .orders import Book, Order
from .pricelaw import F, PriceLaw


@dataclass(frozen=True)
class Market:
    grid: Sequence[int]            # protocol tick grid (ints)
    p_ref: int                     # precommitted reference tick
    tick: Fraction = Fraction(1)   # price units per tick
    P_k: Optional[Fraction] = None # external price at batch time (default: p_ref * tick)
    fee_rate: Fraction = Fraction(0)   # proportional fee on executed notional
    unwind_cost: Fraction = Fraction(0)  # per-unit cost of the mandatory unwind at tau_{k+1}
    collar: Optional[int] = None   # clearing-price collar width in ticks
    batch_id: str = "batch"

    @property
    def external_price(self) -> Fraction:
        return F(self.P_k) if self.P_k is not None else F(self.p_ref) * self.tick

    def price_of(self, tick: int) -> Fraction:
        return F(tick) * self.tick

    def replace(self, **kw) -> "Market":
        d = self.__dict__.copy()
        d.update(kw)
        return Market(**d)


@dataclass(frozen=True)
class WorldResult:
    outcome: Outcome
    adversary_fill: int            # phi^w >= 0
    adversary_position: int        # X^w = s^A * phi^w (0 if no adversary order)
    clearing_price: Fraction       # p^w in price units
    cash: Fraction                 # cash flow at the batch (negative when buying)
    fees: Fraction                 # F^w: auction fee + unwind cost


def _world(book: Book, adversary: Optional[Order], market: Market) -> WorldResult:
    orders = list(book)
    if adversary is not None:
        orders.append(adversary)
    out = clear(orders, market.grid, market.p_ref, market.batch_id, market.collar)
    price = market.price_of(out.price)
    if adversary is None:
        return WorldResult(out, 0, 0, price, Fraction(0), Fraction(0))
    phi = out.fill(adversary.id)
    x = adversary.side * phi
    notional = price * phi
    fee = market.fee_rate * notional + market.unwind_cost * phi
    return WorldResult(out, phi, x, price, -price * x, fee)


@dataclass(frozen=True)
class Surplus:
    """The coupled outcome for one (book, victim, adversary action) triple."""
    plus: WorldResult              # world with the victim
    minus: WorldResult             # world without the victim
    P_k: Fraction
    victim: Optional[Order]

    # --- building blocks -------------------------------------------------------------
    @property
    def dQ(self) -> int:
        return self.plus.adversary_position - self.minus.adversary_position

    @property
    def C_imm(self) -> Fraction:
        p, m = self.plus, self.minus
        return (self.P_k - p.clearing_price) * p.adversary_position - (self.P_k - m.clearing_price) * m.adversary_position

    @property
    def dF(self) -> Fraction:
        return self.plus.fees - self.minus.fees

    def G(self, dP) -> Fraction:
        return F(self.dQ) * F(dP)

    def pi(self, dP) -> Fraction:
        return self.G(dP) + self.C_imm - self.dF

    def wealth(self, dP, world: str) -> Fraction:
        w = self.plus if world == "+" else self.minus
        return w.cash + (self.P_k + F(dP)) * w.adversary_position - w.fees

    # --- victim side -----------------------------------------------------------------
    @property
    def victim_fill(self) -> int:
        return self.plus.outcome.fill(self.victim.id) if self.victim is not None else 0

    @property
    def victim_loss(self) -> Fraction:
        """L_v = s_v (p^+ - P_k) x_v: execution loss against the reference price."""
        if self.victim is None:
            return Fraction(0)
        return self.victim.side * (self.plus.clearing_price - self.P_k) * self.victim_fill

    # --- exact expectations over a finite price law ----------------------------------
    def expected_pi(self, law: PriceLaw) -> Fraction:
        return law.expect(self.pi)

    def expected_pi_pos(self, law: PriceLaw) -> Fraction:
        return law.expect(lambda d: max(self.pi(d), 0))

    def expected_G(self, law: PriceLaw) -> Fraction:
        return F(self.dQ) * law.mean()

    def expected_G_pos(self, law: PriceLaw) -> Fraction:
        return law.expect(lambda d: max(self.G(d), 0))

    def expected_wealth(self, law: PriceLaw, world: str) -> Fraction:
        return law.expect(lambda d: self.wealth(d, world))

    def check_identity(self, law: PriceLaw) -> bool:
        """pi_v == (W^+ - W^-) atom by atom, and == G + C_imm - dF."""
        for d, _ in law.atoms:
            if self.wealth(d, "+") - self.wealth(d, "-") != self.pi(d):
                return False
        return True


def run_coupled(honest: Book, victim: Optional[Order], adversary: Optional[Order],
                market: Market) -> Surplus:
    """Run both coupled worlds with the same adversary order."""
    plus_book = honest.with_orders(victim) if victim is not None else honest
    plus = _world(plus_book, adversary, market)
    minus = _world(honest, adversary, market)
    return Surplus(plus, minus, market.external_price, victim)


def run_leaky(honest: Book, victim: Optional[Order], adv_plus: Optional[Order],
              adv_minus: Optional[Order], market: Market) -> Surplus:
    """Leakage baseline: the adversary may submit different orders in the two worlds."""
    plus_book = honest.with_orders(victim) if victim is not None else honest
    plus = _world(plus_book, adv_plus, market)
    minus = _world(honest, adv_minus, market)
    return Surplus(plus, minus, market.external_price, victim)


def expected_surplus(honest: Book, victim: Optional[Order], adversary: Optional[Order],
                     market: Market, law: PriceLaw) -> Fraction:
    return run_coupled(honest, victim, adversary, market).expected_pi(law)
