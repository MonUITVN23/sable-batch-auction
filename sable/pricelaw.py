"""Finite laws for the external price increment and the declared moment envelope.

All moments are exact rationals. Prices are expressed in quote units per base unit; the
tick size converts ticks to price units.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Sequence, Tuple


def F(x) -> Fraction:
    return x if isinstance(x, Fraction) else Fraction(x)


@dataclass(frozen=True)
class Envelope:
    """Declared bounds |E[dP|H]| <= mu_h and Var(dP|H) <= sigma2_h (with h folded in)."""
    mu_h: Fraction
    sigma2_h: Fraction

    @property
    def sigma_sqrt_h(self) -> float:
        return math.sqrt(self.sigma2_h)

    def contains(self, law: "PriceLaw") -> bool:
        return abs(law.mean()) <= self.mu_h and law.variance() <= self.sigma2_h


class PriceLaw:
    """A finite probability law for dP_{k+1} = P_{k+1} - P_k."""

    def __init__(self, atoms: Sequence[Tuple[Fraction | int, Fraction | int]]):
        self.atoms: List[Tuple[Fraction, Fraction]] = [(F(v), F(p)) for v, p in atoms]
        total = sum(p for _, p in self.atoms)
        if total != 1:
            raise ValueError(f"probabilities sum to {total}, not 1")
        if any(p < 0 for _, p in self.atoms):
            raise ValueError("negative probability")

    # exact moments ---------------------------------------------------------------
    def expect(self, f) -> Fraction:
        return sum(p * F(f(v)) for v, p in self.atoms)

    def mean(self) -> Fraction:
        return self.expect(lambda v: v)

    def second_moment(self) -> Fraction:
        return self.expect(lambda v: v * v)

    def variance(self) -> Fraction:
        m = self.mean()
        return self.second_moment() - m * m

    def positive_part_mean(self) -> Fraction:
        return self.expect(lambda v: max(v, 0))

    def abs_mean(self) -> Fraction:
        return self.expect(abs)

    def scaled(self, c) -> "PriceLaw":
        return PriceLaw([(v * F(c), p) for v, p in self.atoms])

    def shifted(self, c) -> "PriceLaw":
        return PriceLaw([(v + F(c), p) for v, p in self.atoms])

    # sampling (Monte-Carlo consistency check only) ---------------------------------
    def sample(self, rng: random.Random, n: int) -> List[Fraction]:
        vals = [v for v, _ in self.atoms]
        weights = [float(p) for _, p in self.atoms]
        return rng.choices(vals, weights=weights, k=n)

    def __repr__(self) -> str:  # pragma: no cover
        return "PriceLaw(" + ", ".join(f"{v}:{p}" for v, p in self.atoms) + ")"


# The baseline law of the paper: mean 1/4, variance 19/16, E[(dP)^+] = 1/2.
BASELINE_LAW = PriceLaw([(1, Fraction(1, 2)), (0, Fraction(5, 12)), (-3, Fraction(1, 12))])
BASELINE_ENVELOPE = Envelope(Fraction(1, 4), Fraction(19, 16))


def estimate_moments(samples: Sequence[Fraction]) -> Dict[str, float]:
    n = len(samples)
    m = sum(samples) / n
    v = sum((x - m) ** 2 for x in samples) / (n - 1)
    return {"mu_h": float(m), "sigma2_h": float(v), "n": n}
