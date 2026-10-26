# SABLE: victim-attributable surplus in encrypted batch auctions

Reference implementation and experiment code for the paper
*SABLE: Decomposing and Bounding Victim-Attributable Surplus in Encrypted Batch Auctions*.
Every number, table and figure in the paper is produced by this repository.

The model is a single-pair, finite-tick, uniform-price batch call auction whose eligible
set is frozen before disclosure. Two coupled worlds differ only by one victim order, the
same adversary order is submitted in both, and the adversary's victim-attributable surplus

    pi_v = W^{+v} - W^{-v} = G_v + C_imm - dF_v

is split into an inventory-exposure term, a clearing-price term and a fee term.
All money is exact rational arithmetic (`fractions.Fraction`); expectations over the
finite external-price law are exact. Monte-Carlo appears only as a consistency check.

## Quick start

```
python -m pip install -r requirements.txt
python -m pytest -q                  # unit tests (differential, identity, bounds, fixtures)
python run_all.py                    # all experiments -> results/*.json   (~1 min)
python scripts/make_tables.py        # -> tables/*.tex
python scripts/make_figures.py       # -> figures/*.pdf and *.png
```

`python run_all.py channel harm` runs a subset. Nothing needs a GPU or network access;
the only dependency beyond the standard library is `matplotlib` for the figures.

## Layout

| path | content |
|---|---|
| `sable/orders.py` | limit orders and eligible multisets |
| `sable/clearing.py` | lexicographic clearing (max volume, min imbalance, closest to reference, directional tie, smallest tick), strict-before-marginal allocation with largest-remainder pro-rata, optional price collar; plus an independent slow reference implementation |
| `sable/pricelaw.py` | finite laws for the external increment, exact moments, the declared moment envelope |
| `sable/coupled.py` | coupled worlds, terminal-wealth accounting, `pi_v`, `G_v`, `C_imm`, `dF_v`, victim loss `L_v` |
| `sable/optimizer.py` | exhaustive search over the adversary's finite action family (side, quantity, limit) under position and collateral caps; the leakage variant over pairs of orders; the value-of-victim `V_v = max W^{+v} - max W^{-v}` |
| `sable/displacement.py` | clearing-displacement quantities `delta_E(Q)`, `Delta_v`, `beta` of Theorem 7 |
| `sable/bounds.py` | the envelopes of Proposition 4 and Theorem 7 |
| `sable/serial.py` | serial price-time-priority matcher (comparison baseline) |
| `sable/experiments.py` | every experiment of the paper, one function each |
| `run_all.py`, `scripts/` | drivers that write `results/`, `tables/`, `figures/` |
| `tests/` | pytest suite |

## Experiments and where they appear in the paper

| `results/` file | paper |
|---|---|
| `worked_example.json` | Sect. 3, Table 2 (ten-tick example, collar, displacement quantities) |
| `regimes.json` | Table 1 (attained exposure vs envelopes, five Monte-Carlo seeds) |
| `tightness.json` | Fig. 2(a) (single-victim fixture, signed bound attained, tail slack) |
| `channel.json` | Fig. 2(b), Table 3 (channel switch, `L_v`, `V_v`) |
| `channel_robustness.json` | Fig. 2(d), Table 4 (switch cap vs tick size and near depth) |
| `harm.json` | Fig. 2(c), Table 5 (aggressive victim, collar bound of Theorem 7(b)) |
| `collateral.json`, `sensitivity_scale.json` | sensitivity paragraph |
| `leakage.json` | leakage stress test |
| `assumption_boundary.json`, `near_binding.json` | assumption-boundary paragraph |
| `matched_input.json` | matched-input baseline |
| `calibration.json` | calibration feasibility |
| `sequence_audit.json`, `differential.json` | implementation audits |

Results are deterministic: the only random choices are the Monte-Carlo consistency
check and the random books of the audits, all seeded with 20260729.

## Reproducing a single number by hand

```python
from fractions import Fraction as F
from sable import Book, buy, sell, run_coupled, PriceLaw
from sable.coupled import Market

grid = range(1, 11)
E = Book([buy(3, 5, "B1"), sell(3, 5, "S1")])      # honest book
v = buy(4, 8, "V")                                  # victim: urgent buy of 4, limit 8
s = run_coupled(E, v, sell(1, 8, "ADV"), Market(list(grid), p_ref=5))
law = PriceLaw([(1, F(1, 2)), (-1, F(1, 2))])
print(s.dQ, s.C_imm, s.expected_pi(law), s.victim_loss)   # -1 3 3 12
```

## License

MIT.
