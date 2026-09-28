"""The uPRM step scorer, in four functions (uPRM = arXiv 2605.10158, Eq. 6).

A chain is split into steps. The model reads a grading transcript in which
each step is followed by a marker, ``+`` (correct) or ``-`` (incorrect), and
at each marker slot we read its probability of ``+`` vs ``-``. From those
per-step log-probabilities:

    S(j) = log P(step j is the FIRST wrong one)
         = sum_{t<j} log p+_t   +   log p-_j        (j = n_steps+1: no error)

    p_clean = S(n_steps + 1) - logsumexp_j S(j)     the model's probability
                                                    that no step is wrong

Copied verbatim from experiments/q01_does_step_scoring_work/toy1_first_broken_step.py
(stage 32). tests/test_scoring_matches_experiments.py proves it.
"""

MAX_STEPS = 12
SYSTEM = ("You are a strict mathematical reasoning judge. After each step you "
          "respond with + if the step is correct and - if it is incorrect.\n\n")


def split_steps(chain_text):
    """Reasoning steps = non-empty lines, excluding the '#### N' answer line."""
    steps = [line.strip() for line in chain_text.split("\n")
             if line.strip() and not line.lstrip().startswith("####")]
    return steps[:MAX_STEPS]


def build_transcript(question, steps, j, pos=" +", neg=" -"):
    """Grading transcript asserting 'first error at step j' (j = len+1 means
    no error). Ends right after the final marker so every marker slot is a
    next-token position we can read."""
    lines = [SYSTEM + "Problem: " + question.strip()]
    for index, step in enumerate(steps, start=1):
        marker = neg if index == j else pos
        lines.append(f"Step {index}: {step}\nJudge:{marker}")
        if index == j:
            break                      # nothing after the first error matters
    return "\n\n".join(lines)


def marker_slots(question, steps, j):
    """Prompts whose NEXT token is the marker for each graded step, plus the
    marker we are asserting there. One forward pass per slot."""
    slots = []
    prefix = SYSTEM + "Problem: " + question.strip()
    for index, step in enumerate(steps, start=1):
        prompt = prefix + f"\n\nStep {index}: {step}\nJudge:"
        slots.append((prompt, index == j))
        marker = " -" if index == j else " +"
        prefix = prefix + f"\n\nStep {index}: {step}\nJudge:{marker}"
        if index == j:
            break
    return slots


def score_from_logprobs(step_logprobs, j, n_steps):
    """uPRM Eq. 6 from cached per-step (logp_plus, logp_minus).
    step_logprobs[t] is for step t+1 under the all-correct prefix."""
    total = 0.0
    for t in range(min(j - 1, n_steps)):
        total += step_logprobs[t][0]
    if j <= n_steps:
        total += step_logprobs[j - 1][1]
    return total
