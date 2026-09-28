"""uprm/scoring.py must equal the functions in the stage-32 experiment file.

    python -m unittest tests/test_scoring_matches_experiments.py
"""
import random
import sys
import unittest
from pathlib import Path


class _BlockHeavyImports:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("torch", "vllm", "transformers"):
            raise ImportError(f"blocked heavy import: {name}")


sys.meta_path.insert(0, _BlockHeavyImports())
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uprm import scoring                                                    # noqa: E402
from uprm.experiments.q01_does_step_scoring_work import toy1_first_broken_step as s32     # noqa: E402


def random_chain(rng):
    lines = []
    for i in range(rng.randint(0, 16)):
        r = rng.random()
        if r < 0.1:
            lines.append("")
        elif r < 0.2:
            lines.append("   ")
        elif r < 0.3:
            lines.append(f"  #### {rng.randint(1, 99)}")
        else:
            lines.append(f"  step {i}: {rng.randint(1, 9)} * {rng.randint(1, 9)} = "
                         f"<<{rng.randint(1, 9)}*{rng.randint(1, 9)}={rng.randint(1, 81)}>>{rng.randint(1, 81)}  ")
    return "\n".join(lines)


class ScoringMatchesExperiment(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(scoring.MAX_STEPS, s32.MAX_STEPS)
        self.assertEqual(scoring.SYSTEM, s32.SYSTEM)

    def test_functions(self):
        rng = random.Random(3)
        for _ in range(300):
            chain = random_chain(rng)
            q = f"  Question {rng.randint(1, 999)}?  "
            steps = scoring.split_steps(chain)
            self.assertEqual(steps, s32.split_steps(chain))
            for j in range(1, len(steps) + 2):
                self.assertEqual(scoring.build_transcript(q, steps, j),
                                 s32.build_transcript(q, steps, j))
                self.assertEqual(scoring.marker_slots(q, steps, j),
                                 s32.marker_slots(q, steps, j))
                lps = [(-rng.random() * 3, -rng.random() * 3) for _ in steps]
                self.assertEqual(scoring.score_from_logprobs(lps, j, len(steps)),
                                 s32.score_from_logprobs(lps, j, len(steps)))


if __name__ == "__main__":
    unittest.main()
