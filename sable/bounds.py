"""Theoretical envelopes of the paper.

Prop. 4 (exposure):      E[G_v]      <= q_max * mu_h
                          E[(G_v)^+]  <= q_max/2 * (mu_h + sqrt(mu_h^2 + sigma2_h))   (sharp two-moment)
                                      <= q_max * (mu_h + sqrt(sigma2_h))              (earlier, loose)
Thm. 7(a) (resilience):  |C_imm| <= q_max * (beta + 2 delta_E + delta_{E+v} + Delta_v)
Thm. 7(b) (collar):      |C_imm| <= q_max * (|P_k - p_ref| + 3 B)
"""

from __future__ import annotations

import math
from fractions import Fraction

from .pricelaw import Envelope, F


def signed_bound(q_max: int, env: Envelope) -> Fraction:
    return F(q_max) * env.mu_h


def tail_bound(q_max: int, env: Envelope) -> float:
    mu = float(env.mu_h)
    return q_max / 2 * (mu + math.sqrt(mu * mu + float(env.sigma2_h)))


def tail_bound_loose(q_max: int, env: Envelope) -> float:
    return q_max * (float(env.mu_h) + math.sqrt(float(env.sigma2_h)))


def collar_bound(q_max: int, ref_gap, B: int, tick) -> Fraction:
    """|C_imm| <= q_max (|P_k - p_ref| + 3B), with B in ticks and the gap in price units."""
    return F(q_max) * (F(ref_gap) + 3 * F(B) * F(tick))


def resilience_bound(q_max: int, beta, delta_E: int, delta_Ev: int, Delta_v: int, tick) -> Fraction:
    return F(q_max) * (F(beta) + (2 * delta_E + delta_Ev + Delta_v) * F(tick))
