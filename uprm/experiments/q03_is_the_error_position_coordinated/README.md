# step_coherence: is the error position coordinated, even if wrong?

The scorer cannot say *where* a chain breaks (localisation at chance). But
ICM never asks whether a label is *right*, only whether labels are mutually
consistent. So: do chains of the same problem agree on where they break,
more than chance would give?

| file | was | what it does | result |
|---|---|---|---|
| `signature_cache.py` | stage43_sigcache | one-time cache of every chain's equation signature (`data/uPRM_work_/runs/stage43_signatures.jsonl`, 51 MB) | |
| `step_coherence.py` | stage43_step_coherence | for pairs of chains on the same problem, how often their error positions agree, against a permutation null that shuffles positions among chains with the same step count | same-answer pairs: exact 0.393 vs 0.354 null, binary (error / no error) 0.888 vs 0.805. Different-answer pairs agree *less* than null |

Result: `data/uPRM_work_/recorded_results/stage43_results_full.json` (`_smoke` is the small
dry run). The coordination is real but small; it was not built on further.
