"""Serial price-time-priority matching (continuous order-book baseline).

Orders arrive one at a time. An incoming order matches against the best resting orders of
the opposite side while the prices cross, trading at the resting order's limit; the
remainder rests. This is the discipline of on-chain order-book exchanges and the
comparison baseline of the paper's matched-input experiment.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from .orders import Order


def serial_match(arrivals: Sequence[Order]) -> Dict[str, int]:
    bids: List[List] = []   # [limit, seq, order, remaining]
    asks: List[List] = []
    fills: Dict[str, int] = {o.id: 0 for o in arrivals}
    for seq, o in enumerate(arrivals):
        remaining = o.qty
        if o.side == +1:
            asks.sort(key=lambda r: (r[0], r[1]))          # lowest ask, earliest first
            for r in asks:
                if remaining == 0 or r[0] > o.limit:
                    break
                t = min(remaining, r[3])
                r[3] -= t
                remaining -= t
                fills[o.id] += t
                fills[r[2].id] += t
            asks[:] = [r for r in asks if r[3] > 0]
            if remaining:
                bids.append([o.limit, seq, o, remaining])
        else:
            bids.sort(key=lambda r: (-r[0], r[1]))         # highest bid, earliest first
            for r in bids:
                if remaining == 0 or r[0] < o.limit:
                    break
                t = min(remaining, r[3])
                r[3] -= t
                remaining -= t
                fills[o.id] += t
                fills[r[2].id] += t
            bids[:] = [r for r in bids if r[3] > 0]
            if remaining:
                asks.append([o.limit, seq, o, remaining])
    return fills
