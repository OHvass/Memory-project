"""
Pilot Experiment: Human Memory (Free Recall + Serial Recall)
--------------------------------------------------------------
DRAFT v3 for the pilot run described in the project notes / assignment.

Purpose of this pilot (per assignment):
  1. Discover things to improve in the design before the real experiment.
  2. Verify the experiment saves data correctly (check the CSV after running).
  3. Help decide how many repetitions are needed for the actual experiment
     (the pilot itself is not expected to show significant effects).

Each group member should run this once as a participant (data collection
section). Then the group pools all participants' CSVs and runs the
analysis section (analyze_pilot_data) -- the same pipeline intended for
the full experiment.

Guessing-avoidance design choice:
  Free recall risks a fixed-response-set problem: if you always show all
  10 digits (0-9), the "correct answer" is the same every trial and a
  participant could learn to just type 0-9 without remembering the actual
  sequence. To avoid this, each free-recall trial draws a RANDOM SUBSET
  from a larger item pool, so the correct set differs every trial and
  can't be memorized once. Serial recall is less exposed to this specific
  issue (order still has to be recalled correctly), but item sets are
  still drawn from a larger pool and randomized per trial as good practice.

v3 additions:
  - Paired statistical comparisons between conditions (paired t-test where
    applicable, plus a distribution-free permutation test, since pilot n
    is tiny and normality can't be assumed).
  - Sound-alike vs. look-alike error classification for serial recall,
    using small phonetic/visual similarity tables over the letter pool.
  - Multi-file pooling: analyze_pilot_data() accepts a list of CSV paths
    (one per participant) and pools them, matching the "treat as one
    participant run N times" simplification from the assignment.
  - Matplotlib figures for the serial position curve, the free-recall
    region comparison, the serial-recall span comparison, and the
    error-type breakdown -- so the statistical results are visualized,
    not just printed.

v4 changes (addressing pilot review before the main experiment):
  - Block order is now counterbalanced across participants instead of
    fixed. Each participant enters an "order index"; block order is a
    cyclic rotation of that index (a simple Latin-square-style scheme),
    or randomized if no index is given. The resulting order is logged
    per trial (block_position) so order effects can be checked/controlled
    for later instead of being an unexamined confound.
  - Distractor-task numbers are now generated fresh each trial instead of
    once at import time (previously every participant saw the same 5
    numbers on every distractor trial).
  - Free- and serial-recall scoring now flags trials whose response
    contains a token that isn't in the relevant item pool (flagged_for_
    review column) -- almost always a typo -- so it can be checked by
    hand instead of being silently scored as simply wrong.
  - Serial recall now also reports an order-free "any position" accuracy
    (multiset overlap between recalled and shown items) alongside the
    standard strict in-place accuracy. Comparing the two flags cases where
    a single early omission/insertion cascades into misleadingly low
    in-place scores.
  - The primary, hypothesis-driven cross-condition comparisons are now
    corrected for multiple comparisons (Benjamini-Hochberg FDR) and
    reported separately from the within-condition region checks, which
    are explicitly labeled exploratory/uncorrected.
  - Rest breaks are inserted between blocks, and total session duration is
    printed at the end, to make fatigue/session-length visible rather than
    silent.
"""

import csv
import itertools
import math
import os
import random
import string
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # no display available in this environment
import matplotlib.pyplot as plt
from scipy import stats

# ---------------------------------------------------------------------------
# Config -- tune these after piloting; repetition count should be revisited
# once pilot data is analyzed (see report_outline_v2.md, Sec. 3)
# ---------------------------------------------------------------------------
PRESENTATION_RATE_SEC = 4.0      # seconds each item is shown
FAST_RATE_SEC = 2.0              # for the presentation-rate manipulation
FREE_RECALL_LIST_LEN = 12        # length of list shown per trial (>10)
SERIAL_BASE_LEN = 7              # classic span starting point
TRIALS_PER_CONDITION = 3         # keep pilot brief; full exp count TBD from pilot

# Large pools so each trial's correct set differs (avoids fixed-answer guessing)
LETTER_POOL = [c for c in string.ascii_uppercase if c not in "AEIOU"]  # 21 letters
THREE_LETTER_WORDS = [
    "CAT", "DOG", "SUN", "BOX", "PEN", "CUP", "HAT", "RUG",
    "BAT", "MAP", "JAR", "LOG", "FAN", "KEY", "NET", "OWL",
]
DISTRACTOR_TASK_N = 5             # number of distractor items; regenerated fresh each trial

assert FREE_RECALL_LIST_LEN < len(LETTER_POOL), (
    "Free-recall list must be a strict subset of the pool, "
    "otherwise every trial's correct answer is identical (guessing risk)."
)

# ---------------------------------------------------------------------------
# Block definitions + counterbalancing
# ---------------------------------------------------------------------------
# Each participant used to see blocks in this exact fixed order, which
# confounds any condition effect with practice/fatigue over the session.
# These lists are now rotated per participant (see counterbalanced_order)
# instead of being run in a hardcoded sequence.
FREE_RECALL_CONDITIONS = [
    ("baseline", dict(rate_sec=PRESENTATION_RATE_SEC, post_task=None)),
    ("fast_rate", dict(rate_sec=FAST_RATE_SEC, post_task=None)),
    ("distractor_after", dict(rate_sec=PRESENTATION_RATE_SEC, post_task="distractor")),
    ("pause_after", dict(rate_sec=PRESENTATION_RATE_SEC, post_task="pause")),
]

SERIAL_RECALL_CONDITIONS = [
    ("baseline_letters", dict(item_pool=LETTER_POOL, suppression=None)),
    ("chunking_words", dict(item_pool=THREE_LETTER_WORDS, suppression=None)),
    ("articulatory_suppression", dict(item_pool=LETTER_POOL, suppression="articulatory")),
    ("finger_tapping", dict(item_pool=LETTER_POOL, suppression="finger_tapping")),
]


def get_participant_order_index():
    """Ask for an integer that determines this participant's block order.
    Group members should each use a DIFFERENT number (0, 1, 2, ...) so that,
    across the group, every condition appears about equally often in every
    serial position -- a simple Latin-square-style counterbalance. Leaving
    it blank (or entering something invalid) falls back to a random order."""
    raw = input(
        "Participant order index (0, 1, 2, ... -- use a DIFFERENT number for "
        "each group member to counterbalance block order; leave blank to "
        "randomize): "
    ).strip()
    if raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        print("  (not a valid integer -- falling back to a randomized order)")
        return None


def counterbalanced_order(conditions, participant_index):
    """Cyclically rotate `conditions` based on participant_index so block
    order varies systematically across the group instead of being identical
    for everyone. Falls back to a random shuffle if participant_index is None."""
    conditions = list(conditions)
    if participant_index is None:
        random.shuffle(conditions)
        return conditions
    shift = participant_index % len(conditions)
    return conditions[shift:] + conditions[:shift]


def take_a_break(message="Take a short break, then press Enter to continue..."):
    input(f"\n{message}\n")

# --- Sound-alike / look-alike tables over LETTER_POOL (B C D F G H J K L M N P Q R S T V W X Y Z) ---
# Rough, commonly-cited English letter-confusion groupings (Conrad, 1964-style).
# These are simplifications for pilot-scale classification, not a validated phonetic model.
SOUND_ALIKE_GROUPS = [
    {"B", "D", "G", "P", "T", "V"},   # "ee"-family rhymes: bee/dee/gee/pea/tee/vee -- BDGPT V share stop+vowel confusion
    {"M", "N"},
    {"S", "X", "F"},
    {"C", "Z"},
]
LOOK_ALIKE_GROUPS = [
    {"M", "N", "W"},
    {"C", "G", "O"} & set(LETTER_POOL),
    {"P", "R", "B"},
    {"V", "Y"},
    {"F", "T"},
]


def letters_confusable(a, b, groups):
    return any(a in g and b in g for g in groups)


class _Tee:
    """Writes to multiple streams at once. Used to mirror every printed
    line (prompts, scores, statistical output) into a saved log file as
    well as the terminal, so the console text isn't lost once the terminal
    or session closes."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


# Registry of every statistical test run in one analysis session, used to
# build a single summary figure (_plot_all_statistical_tests) of every test
# performed -- both the pre-specified/BH-corrected comparisons and the
# exploratory ones -- rather than only having the numbers scroll past in
# the console.
_STAT_TEST_LOG = []


def _reset_stat_test_log():
    _STAT_TEST_LOG.clear()


def _log_stat_test(label, family, raw_p):
    _STAT_TEST_LOG.append({
        "label": label,
        "family": family,
        "raw_p": raw_p,
        "adjusted_p": None,
        "significant": None,
    })


# ---------------------------------------------------------------------------
# Presentation / response helpers
# ---------------------------------------------------------------------------
def clear():
    os.system("cls" if os.name == "nt" else "clear")


def present_sequence(items, rate_sec):
    """Show items one at a time at the given rate (simulates paced presentation)."""
    for item in items:
        clear()
        print(item)
        time.sleep(rate_sec)
    clear()
    print("(sequence ended)")


def get_response(prompt):
    return input(prompt).strip().upper()


def log_trial(writer, **fields):
    writer.writerow(fields)


def score_free_recall(list_shown, response_str):
    """Rough scoring: which positions were correctly recalled (order-free).
    Also flags trials containing a token not found in LETTER_POOL at all --
    almost always a typo -- so they can be checked by hand rather than
    silently scored as a plain miss."""
    recalled = [w.strip() for w in response_str.split(",") if w.strip()]
    correct_positions = [i for i, item in enumerate(list_shown) if item in recalled]
    flagged = any(w not in LETTER_POOL for w in recalled)
    return correct_positions, len(recalled), flagged


def score_serial_recall(list_shown, response_str, item_pool):
    """Position-by-position exact-order scoring (the standard serial-recall
    measure), plus error classification.

    NOTE on a known limitation: strict positional scoring means a single
    early omission or inserted item shifts every later comparison ("cascading"
    misalignment), which can look like a string of wrong answers even though
    most items were actually remembered. Rather than silently living with
    that, we also compute an order-free "any position" accuracy (multiset
    overlap of recalled vs. shown items) below, so analyze_serial_recall()
    can flag conditions/trials where the two measures diverge a lot.
    """
    recalled = [w.strip() for w in response_str.split(",") if w.strip()]
    n_correct_in_place = 0
    sound_alike_errors = 0
    look_alike_errors = 0
    other_errors = 0
    for i, target in enumerate(list_shown):
        given = recalled[i] if i < len(recalled) else None
        if given == target:
            n_correct_in_place += 1
        elif given is not None and len(given) == 1 and len(target) == 1:
            if letters_confusable(given, target, SOUND_ALIKE_GROUPS):
                sound_alike_errors += 1
            elif letters_confusable(given, target, LOOK_ALIKE_GROUPS):
                look_alike_errors += 1
            else:
                other_errors += 1
        elif given is not None:
            other_errors += 1
    fully_correct = recalled == list_shown

    shown_counts = Counter(list_shown)
    recalled_counts = Counter(recalled)
    n_correct_any_position = sum(min(shown_counts[k], recalled_counts[k]) for k in shown_counts)

    flagged = any(w not in item_pool for w in recalled)

    return (n_correct_in_place, fully_correct, sound_alike_errors, look_alike_errors,
            other_errors, n_correct_any_position, flagged)


# ---------------------------------------------------------------------------
# Free Recall Experiment
# ---------------------------------------------------------------------------
def run_free_recall_block(writer, condition, rate_sec, post_task=None, block_position=None):
    """
    condition: label for logging, e.g. 'baseline', 'fast_rate',
               'distractor_after', 'pause_after'
    post_task: None | 'distractor' | 'pause'
    block_position: this block's position (1-based) in this participant's
                     running order -- logged so order effects can be
                     checked/controlled for rather than being an invisible
                     confound (see counterbalanced_order).
    """
    for trial in range(TRIALS_PER_CONDITION):
        items = random.sample(LETTER_POOL, FREE_RECALL_LIST_LEN)  # random subset each trial
        print(f"\n[Free Recall | {condition} | trial {trial+1}] Get ready...")
        time.sleep(1)
        present_sequence(items, rate_sec)

        if post_task == "distractor":
            # Fresh numbers every trial (previously a fixed set generated once
            # at import time, so every participant saw the same 5 numbers
            # every single time).
            distractor_items = [str(random.randint(100, 999)) for _ in range(DISTRACTOR_TASK_N)]
            print("Quick task: type back these numbers as you see them.")
            for n in distractor_items:
                input(f"  Type {n}: ")
        elif post_task == "pause":
            print("(brief pause...)")
            time.sleep(5)

        response = get_response("Recall the letters, in ANY order, comma-separated: ")
        correct_positions, n_recalled, flagged = score_free_recall(items, response)

        log_trial(
            writer,
            experiment="free_recall",
            condition=condition,
            block_position=block_position,
            trial=trial + 1,
            list_shown="|".join(items),
            response=response,
            n_recalled=n_recalled,
            correct_positions="|".join(map(str, correct_positions)),
            rate_sec=rate_sec,
            list_len=len(items),
            flagged_for_review=flagged,
        )


# ---------------------------------------------------------------------------
# Serial Recall Experiment
# ---------------------------------------------------------------------------
def run_serial_recall_block(writer, condition, item_pool, seq_len=SERIAL_BASE_LEN,
                             suppression=None, block_position=None):
    """
    condition: label for logging, e.g. 'baseline_letters', 'chunking_words',
               'articulatory_suppression', 'finger_tapping'
    suppression: None | 'articulatory' | 'finger_tapping'
    block_position: this block's position (1-based) in this participant's
                     running order -- see counterbalanced_order.
    """
    for trial in range(TRIALS_PER_CONDITION):
        items = random.sample(item_pool, min(seq_len, len(item_pool)))
        print(f"\n[Serial Recall | {condition} | trial {trial+1}] Get ready...")
        if suppression == "articulatory":
            print("Repeat 'the-the-the' out loud continuously while items are shown and while you type.")
        elif suppression == "finger_tapping":
            print("Tap your finger on the table continuously while items are shown and while you type.")
        time.sleep(1)
        present_sequence(items, PRESENTATION_RATE_SEC)

        response = get_response("Recall the items IN ORDER, comma-separated: ")
        (n_correct_in_place, fully_correct, sound_alike_errors, look_alike_errors,
         other_errors, n_correct_any_position, flagged) = score_serial_recall(items, response, item_pool)

        log_trial(
            writer,
            experiment="serial_recall",
            condition=condition,
            block_position=block_position,
            trial=trial + 1,
            list_shown="|".join(items),
            response=response,
            seq_len=len(items),
            n_correct_in_place=n_correct_in_place,
            n_correct_any_position=n_correct_any_position,
            fully_correct=fully_correct,
            sound_alike_errors=sound_alike_errors,
            look_alike_errors=look_alike_errors,
            other_errors=other_errors,
            flagged_for_review=flagged,
        )


# ---------------------------------------------------------------------------
# Analysis (same pipeline for pilot and full experiment; pools across
# participants as a simplification -- see report_outline_v2.md, Sec. 4)
# ---------------------------------------------------------------------------
def wilson_ci(n_correct, n_total, z=1.96):
    """Wilson score confidence interval for a proportion (better than normal
    approximation for small n, which pilot data will have)."""
    if n_total == 0:
        return (float("nan"), float("nan"))
    p = n_correct / n_total
    denom = 1 + z**2 / n_total
    centre = (p + z**2 / (2 * n_total)) / denom
    margin = (z * math.sqrt((p * (1 - p) + z**2 / (4 * n_total)) / n_total)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def load_rows(csv_paths, experiment):
    """Load and pool rows from one or more participant CSVs.
    Pooling multiple files here is the 'treat as one participant run N
    times' simplification described in the assignment: it merges all rows
    together and does not track between-participant variability."""
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]
    rows = []
    for path in csv_paths:
        with open(path) as f:
            for row in csv.DictReader(f):
                if row["experiment"] == experiment:
                    rows.append(row)
    return rows


def permutation_test(sample_a, sample_b, n_perm=5000, seed=0):
    """Distribution-free two-sample permutation test on the difference of
    means. Preferred over a t-test when n is small (pilot-scale) and
    normality can't be assumed; reported alongside the t-test for comparison."""
    rng = random.Random(seed)
    combined = sample_a + sample_b
    n_a = len(sample_a)
    observed = abs(sum(sample_a) / len(sample_a) - sum(sample_b) / len(sample_b))
    count = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        perm_a = combined[:n_a]
        perm_b = combined[n_a:]
        diff = abs(sum(perm_a) / len(perm_a) - sum(perm_b) / len(perm_b))
        if diff >= observed:
            count += 1
    return count / n_perm  # p-value


def benjamini_hochberg(pvalues, alpha=0.05):
    """Benjamini-Hochberg false-discovery-rate correction.
    Returns (adjusted_pvalues, reject_flags) in the same order as `pvalues`.
    Used for the primary, hypothesis-driven comparisons: running many tests
    without correction inflates the chance of calling noise 'significant'."""
    m = len(pvalues)
    if m == 0:
        return [], []
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running_min = 1.0
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        candidate = pvalues[idx] * m / rank
        running_min = min(running_min, candidate)
        adjusted[idx] = min(running_min, 1.0)
    reject = [p <= alpha for p in adjusted]
    return adjusted, reject


def _report_bh_correction(results, family_label, alpha=0.05):
    """results: list of (label, p_value) tuples for one family of PRIMARY
    (confirmatory) comparisons. Prints raw + BH-adjusted p-values so a
    single, clearly-labeled correction is applied where it matters most,
    rather than treating every printed p-value in this script as
    confirmatory."""
    if not results:
        return
    labels = [r[0] for r in results]
    pvals = [r[1] for r in results]
    adjusted, reject = benjamini_hochberg(pvals, alpha=alpha)
    print(f"\n  --- Benjamini-Hochberg FDR correction: '{family_label}' "
          f"({len(pvals)} tests, alpha={alpha}) ---")
    print("  (Only these targeted, pre-specified comparisons are corrected here;")
    print("   the within-condition checks above are exploratory/diagnostic.)")
    for lbl, p, adj, rej in zip(labels, pvals, adjusted, reject):
        verdict = "SIGNIFICANT after correction" if rej else "not significant after correction"
        print(f"    {lbl:50s} raw p={p:.3f}  adj p={adj:.3f}  -> {verdict}")
        for entry in _STAT_TEST_LOG:
            if entry["family"] == family_label and entry["label"] == lbl:
                entry["adjusted_p"] = adj
                entry["significant"] = rej
                break


def compare_conditions(label_a, sample_a, label_b, sample_b, log_as=None):
    """Run both a paired t-test (when equal length: paired by trial index)
    and a permutation test, print results, and return them.

    log_as: optional (label, family) tuple. If given, this comparison's raw
    p-value is added to the module-level statistical test registry, which
    _plot_all_statistical_tests() turns into one summary figure of every
    test run. `family` groups tests that belong together (e.g. all the
    free-recall primary hypotheses), matching the family_label later passed
    to _report_bh_correction so the adjusted p-value can be filled in."""
    print(f"\n  Comparing {label_a} vs {label_b} (n={len(sample_a)} vs n={len(sample_b)}):")
    if len(sample_a) < 2 or len(sample_b) < 2:
        print("    Not enough trials for a statistical test (expected at pilot scale).")
        return None

    perm_p = permutation_test(list(sample_a), list(sample_b))
    print(f"    Permutation test p-value: {perm_p:.3f}")

    if len(sample_a) == len(sample_b):
        t_stat, t_p = stats.ttest_rel(sample_a, sample_b)
        print(f"    Paired t-test:            t={t_stat:.2f}, p={t_p:.3f}")
    else:
        t_stat, t_p = stats.ttest_ind(sample_a, sample_b)
        print(f"    Independent t-test:       t={t_stat:.2f}, p={t_p:.3f} "
              f"(unequal n -- paired test not applicable)")

    print("    Note: with small per-condition n, expect wide confidence intervals; "
          "for the PRIMARY hypothesis comparisons this raw p-value is corrected for "
          "multiple comparisons below (Benjamini-Hochberg FDR) before drawing conclusions.")

    if log_as is not None:
        label, family = log_as
        _log_stat_test(label, family, perm_p)

    return perm_p, t_p


def analyze_free_recall(csv_paths, out_dir="."):
    """
    Serial position curve per condition, with Wilson CIs.
    Compares recency-region and primacy-region proportions against the
    middle-of-sequence baseline (as required), and statistically compares
    matching regions ACROSS conditions (e.g. recency in baseline vs.
    recency in distractor_after).
    """
    rows = load_rows(csv_paths, "free_recall")

    n_flagged = sum(1 for row in rows if row.get("flagged_for_review") == "True")
    if n_flagged:
        print(f"\n  NOTE: {n_flagged}/{len(rows)} free-recall trials contained a response "
              f"token not in the letter pool (likely a typo) -- flagged_for_review=True "
              f"in the CSV. Worth a manual look before trusting those trials' scores.")

    # position -> condition -> [n_correct, n_total]
    counts = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    # condition -> region -> list of per-trial proportions (for stats/plots)
    region_trial_scores = defaultdict(lambda: defaultdict(list))

    for row in rows:
        condition = row["condition"]
        list_len = int(row["list_len"])
        correct_positions = set(int(p) for p in row["correct_positions"].split("|") if p)

        primacy_positions = set(range(0, 2))
        recency_positions = set(range(list_len - 2, list_len))
        middle_positions = set(range(2, list_len - 2))

        for pos in range(list_len):
            counts[condition][pos][1] += 1
            if pos in correct_positions:
                counts[condition][pos][0] += 1

        for region_name, positions in [
            ("primacy", primacy_positions),
            ("middle", middle_positions),
            ("recency", recency_positions),
        ]:
            if positions:
                score = len(correct_positions & positions) / len(positions)
                region_trial_scores[condition][region_name].append(score)

    print("\n--- Free Recall: serial position curve (proportion correct, 95% CI) ---")
    region_summary = {}  # condition -> region -> (p, lo, hi, n)
    for condition, pos_counts in counts.items():
        print(f"\nCondition: {condition}")
        list_len = max(pos_counts) + 1
        primacy_positions = range(0, 2)
        recency_positions = range(list_len - 2, list_len)
        middle_positions = range(2, list_len - 2)

        region_summary[condition] = {}
        for label, positions in [
            ("primacy", primacy_positions),
            ("middle", middle_positions),
            ("recency", recency_positions),
        ]:
            n_correct = sum(pos_counts[p][0] for p in positions if p in pos_counts)
            n_total = sum(pos_counts[p][1] for p in positions if p in pos_counts)
            if n_total == 0:
                continue
            p = n_correct / n_total
            lo, hi = wilson_ci(n_correct, n_total)
            region_summary[condition][label] = (p, lo, hi, n_total)
            print(f"  {label:20s}: p={p:.2f}  95% CI=[{lo:.2f}, {hi:.2f}]  (n={n_total})")

        # Within-condition: recency/primacy vs middle baseline
        # (exploratory/diagnostic -- not corrected for multiple comparisons;
        # the targeted cross-condition tests below are the confirmatory ones.)
        if "recency" in region_trial_scores[condition] and "middle" in region_trial_scores[condition]:
            compare_conditions(f"{condition}: recency", region_trial_scores[condition]["recency"],
                                f"{condition}: middle", region_trial_scores[condition]["middle"],
                                log_as=(f"{condition}: recency vs middle",
                                        "free recall region checks (exploratory)"))
        if "primacy" in region_trial_scores[condition] and "middle" in region_trial_scores[condition]:
            compare_conditions(f"{condition}: primacy", region_trial_scores[condition]["primacy"],
                                f"{condition}: middle", region_trial_scores[condition]["middle"],
                                log_as=(f"{condition}: primacy vs middle",
                                        "free recall region checks (exploratory)"))

    # Across-condition comparisons matching the design's hypotheses
    print("\n--- Free Recall: cross-condition comparisons (targeted hypotheses) ---")
    hypothesis_pairs = [
        ("baseline", "fast_rate", "primacy",
         "H: faster rate reduces primacy"),
        ("baseline", "fast_rate", "recency",
         "H: faster rate does NOT change recency"),
        ("baseline", "distractor_after", "recency",
         "H: distractor after sequence reduces recency"),
        ("baseline", "distractor_after", "primacy",
         "H: distractor after sequence does NOT change primacy"),
        ("baseline", "pause_after", "recency",
         "H: brief pause has little effect on recency"),
        ("baseline", "pause_after", "primacy",
         "H: brief pause has little effect on primacy"),
    ]
    primary_results = []
    for cond_a, cond_b, region, hypothesis in hypothesis_pairs:
        if (cond_a in region_trial_scores and region in region_trial_scores[cond_a]
                and cond_b in region_trial_scores and region in region_trial_scores[cond_b]):
            print(f"\n  {hypothesis}")
            label = f"{cond_a} vs {cond_b} ({region})"
            result = compare_conditions(f"{cond_a} ({region})", region_trial_scores[cond_a][region],
                                         f"{cond_b} ({region})", region_trial_scores[cond_b][region],
                                         log_as=(label, "free recall primary hypotheses"))
            if result is not None:
                perm_p, _t_p = result
                primary_results.append((label, perm_p))
    _report_bh_correction(primary_results, family_label="free recall primary hypotheses")

    _plot_serial_position_curves(counts, out_dir)
    _plot_region_comparison(region_summary, out_dir)
    return region_summary


def analyze_serial_recall(csv_paths, out_dir="."):
    """
    Span / accuracy per condition, pooled across participants, with CIs.
    Also compares baseline vs. articulatory_suppression vs. finger_tapping,
    and classifies confusion errors as sound-alike vs. look-alike.
    """
    rows = load_rows(csv_paths, "serial_recall")

    n_flagged = sum(1 for row in rows if row.get("flagged_for_review") == "True")
    if n_flagged:
        print(f"\n  NOTE: {n_flagged}/{len(rows)} serial-recall trials contained a response "
              f"token not in that trial's item pool (likely a typo) -- flagged_for_review=True "
              f"in the CSV. Worth a manual look before trusting those trials' scores.")

    # condition -> [n_fully_correct, n_trials]
    counts = defaultdict(lambda: [0, 0])
    # condition -> list of per-trial proportion-correct-in-place (for stats)
    trial_scores = defaultdict(list)
    # condition -> list of per-trial proportion-correct regardless of position
    # (diagnostic for cascading misalignment -- see score_serial_recall)
    trial_scores_any_position = defaultdict(list)
    # condition -> [sound_alike, look_alike, other]
    error_counts = defaultdict(lambda: [0, 0, 0])

    for row in rows:
        condition = row["condition"]
        counts[condition][1] += 1
        if row["fully_correct"] == "True":
            counts[condition][0] += 1
        seq_len = int(row["seq_len"])
        trial_scores[condition].append(int(row["n_correct_in_place"]) / seq_len)
        if row.get("n_correct_any_position"):
            trial_scores_any_position[condition].append(int(row["n_correct_any_position"]) / seq_len)
        error_counts[condition][0] += int(row.get("sound_alike_errors", 0) or 0)
        error_counts[condition][1] += int(row.get("look_alike_errors", 0) or 0)
        error_counts[condition][2] += int(row.get("other_errors", 0) or 0)

    print("\n--- Serial Recall: in-place vs. order-free accuracy (diagnostic) ---")
    for condition in counts:
        if trial_scores[condition] and trial_scores_any_position.get(condition):
            in_place_mean = sum(trial_scores[condition]) / len(trial_scores[condition])
            any_pos_mean = (sum(trial_scores_any_position[condition])
                             / len(trial_scores_any_position[condition]))
            gap = any_pos_mean - in_place_mean
            note = ("  <- items often present but misordered/misaligned, not simply forgotten"
                    if gap > 0.15 else "")
            print(f"  {condition:25s}: in-place={in_place_mean:.2f}  order-free={any_pos_mean:.2f}{note}")

    print("\n--- Serial Recall: proportion of fully-correct trials (95% CI) ---")
    summary = {}
    for condition, (n_correct, n_total) in counts.items():
        if n_total == 0:
            continue
        p = n_correct / n_total
        lo, hi = wilson_ci(n_correct, n_total)
        summary[condition] = (p, lo, hi, n_total)
        print(f"  {condition:25s}: p={p:.2f}  95% CI=[{lo:.2f}, {hi:.2f}]  (n={n_total})")

    print("\n--- Serial Recall: targeted comparisons ---")
    comparisons = [
        ("baseline_letters", "chunking_words",
         "H: chunking (words) keeps span near baseline (~7 units)"),
        ("baseline_letters", "articulatory_suppression",
         "H: articulatory suppression reduces span"),
        ("baseline_letters", "finger_tapping",
         "H: finger tapping does NOT reduce span"),
        ("articulatory_suppression", "finger_tapping",
         "H: suppression effect is specific to articulatory task, not general dual-tasking"),
    ]
    primary_results = []
    for cond_a, cond_b, hypothesis in comparisons:
        if cond_a in trial_scores and cond_b in trial_scores:
            print(f"\n  {hypothesis}")
            label = f"{cond_a} vs {cond_b}"
            result = compare_conditions(cond_a, trial_scores[cond_a], cond_b, trial_scores[cond_b],
                                         log_as=(label, "serial recall primary hypotheses"))
            if result is not None:
                perm_p, _t_p = result
                primary_results.append((label, perm_p))
    _report_bh_correction(primary_results, family_label="serial recall primary hypotheses")

    print("\n--- Serial Recall: error-type breakdown (sound-alike vs. look-alike) ---")
    for condition, (sound_alike, look_alike, other) in error_counts.items():
        total_errors = sound_alike + look_alike + other
        if total_errors == 0:
            print(f"  {condition:25s}: no classifiable single-letter substitution errors observed")
            continue
        print(f"  {condition:25s}: sound-alike={sound_alike}, look-alike={look_alike}, "
              f"other={other}  (total={total_errors})")
    print("  --> Prediction: look-alike share should rise specifically under "
          "'articulatory_suppression' relative to 'baseline_letters'.")

    _plot_span_comparison(summary, out_dir)
    _plot_error_types(error_counts, out_dir)
    return summary, error_counts


def _plot_all_statistical_tests(out_dir):
    """One figure summarizing every statistical test run this session: each
    comparison as a row, its raw permutation-test p-value as a circle, and
    -- for the pre-specified primary/confirmatory hypotheses -- its
    Benjamini-Hochberg-adjusted p-value as a diamond, connected by a line.
    This is meant as the at-a-glance graphical companion to the printed
    (and now logged) console statistics, not a replacement for them."""
    if not _STAT_TEST_LOG:
        return

    def sort_key(entry):
        is_exploratory = "exploratory" in entry["family"]
        return (is_exploratory, entry["family"], entry["raw_p"])

    entries = sorted(_STAT_TEST_LOG, key=sort_key)
    labels = [e["label"] for e in entries]

    fig_height = max(4.0, 0.35 * len(entries) + 1.5)
    fig, ax = plt.subplots(figsize=(9.5, fig_height))
    ys = range(len(entries))

    unique_families = list(dict.fromkeys(e["family"] for e in entries))
    cmap = plt.get_cmap("tab10")
    family_colors = {fam: cmap(i % 10) for i, fam in enumerate(unique_families)}

    for yi, e in zip(ys, entries):
        color = family_colors[e["family"]]
        ax.scatter(e["raw_p"], yi, marker="o", facecolors="none", edgecolors=color,
                   s=60, zorder=3)
        if e["adjusted_p"] is not None:
            ax.scatter(e["adjusted_p"], yi, marker="D", color=color, s=45, zorder=3)
            ax.plot([e["raw_p"], e["adjusted_p"]], [yi, yi], color=color, alpha=0.35, linewidth=1)

    ax.axvline(0.05, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(list(ys))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("p-value")
    ax.set_title("All Statistical Tests\ncircle = raw permutation p, diamond = BH-adjusted p (where corrected)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor="none",
                   markeredgecolor=family_colors[fam], label=fam, markersize=7)
        for fam in unique_families
    ]
    legend_handles.append(plt.Line2D([0], [0], color="black", linestyle="--", label="alpha = 0.05"))
    ax.legend(handles=legend_handles, fontsize=7, loc="lower right")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "all_statistical_tests_summary.png"), dpi=150)
    plt.close(fig)


def analyze_pilot_data(csv_paths, out_dir="."):
    """Run both analyses -- same pipeline intended for the full experiment.
    csv_paths: a single path, or a list of paths (one per participant) to
    pool together, matching the assignment's pooling simplification."""
    os.makedirs(out_dir, exist_ok=True)
    _reset_stat_test_log()
    analyze_free_recall(csv_paths, out_dir)
    analyze_serial_recall(csv_paths, out_dir)
    _plot_all_statistical_tests(out_dir)
    print(f"\nFigures saved to: {os.path.abspath(out_dir)}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _plot_serial_position_curves(counts, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for condition, pos_counts in counts.items():
        positions = sorted(pos_counts)
        props = [pos_counts[p][0] / pos_counts[p][1] if pos_counts[p][1] else 0 for p in positions]
        ax.plot([p + 1 for p in positions], props, marker="o", label=condition)
    ax.set_xlabel("Serial position")
    ax.set_ylabel("Proportion correct")
    ax.set_title("Free Recall: Serial Position Curve by Condition")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "free_recall_serial_position_curve.png"), dpi=150)
    plt.close(fig)


def _plot_region_comparison(region_summary, out_dir):
    conditions = list(region_summary.keys())
    regions = ["primacy", "middle", "recency"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.25
    x = range(len(conditions))
    for i, region in enumerate(regions):
        vals, errs = [], []
        for cond in conditions:
            if region in region_summary[cond]:
                p, lo, hi, n = region_summary[cond][region]
                vals.append(p)
                # max(0, ...) guards against tiny negative values from
                # floating-point rounding in wilson_ci (e.g. hi can come out
                # as 0.9999999999999999 when p=1.0, making hi - p a very
                # small negative number instead of exactly 0). matplotlib's
                # yerr rejects any negative value, even -1e-16.
                errs.append((max(0.0, p - lo), max(0.0, hi - p)))
            else:
                vals.append(0)
                errs.append((0, 0))
        err_lo = [e[0] for e in errs]
        err_hi = [e[1] for e in errs]
        positions = [xi + (i - 1) * width for xi in x]
        ax.bar(positions, vals, width=width, yerr=[err_lo, err_hi], capsize=3, label=region)
    ax.set_xticks(list(x))
    ax.set_xticklabels(conditions, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Proportion correct (95% CI)")
    ax.set_title("Free Recall: Primacy / Middle / Recency by Condition")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "free_recall_region_comparison.png"), dpi=150)
    plt.close(fig)


def _plot_span_comparison(summary, out_dir):
    conditions = list(summary.keys())
    vals = [summary[c][0] for c in conditions]
    # max(0, ...) guards against tiny negative values from floating-point
    # rounding in wilson_ci -- see the matching note in _plot_region_comparison.
    err_lo = [max(0.0, summary[c][0] - summary[c][1]) for c in conditions]
    err_hi = [max(0.0, summary[c][2] - summary[c][0]) for c in conditions]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(len(conditions))
    ax.bar(x, vals, yerr=[err_lo, err_hi], capsize=4, color="steelblue")
    ax.set_xticks(list(x))
    ax.set_xticklabels(conditions, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Proportion fully-correct trials (95% CI)")
    ax.set_title("Serial Recall: Accuracy by Condition")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "serial_recall_span_comparison.png"), dpi=150)
    plt.close(fig)


def _plot_error_types(error_counts, out_dir):
    conditions = list(error_counts.keys())
    sound_alike = [error_counts[c][0] for c in conditions]
    look_alike = [error_counts[c][1] for c in conditions]
    other = [error_counts[c][2] for c in conditions]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(len(conditions))
    ax.bar(x, sound_alike, label="sound-alike")
    ax.bar(x, look_alike, bottom=sound_alike, label="look-alike")
    bottom2 = [s + l for s, l in zip(sound_alike, look_alike)]
    ax.bar(x, other, bottom=bottom2, label="other")
    ax.set_xticks(list(x))
    ax.set_xticklabels(conditions, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Number of substitution errors")
    ax.set_title("Serial Recall: Error Type by Condition")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "serial_recall_error_types.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def collect_data():
    participant_id = input("Participant ID (e.g. initials): ").strip() or "anon"
    order_index = get_participant_order_index()
    output_file = f"pilot_data_{participant_id}_{datetime.now():%Y%m%d_%H%M%S}.csv"

    fieldnames = [
        "experiment", "condition", "block_position", "trial",
        "list_shown", "response", "rate_sec", "list_len",
        "n_recalled", "correct_positions",
        "seq_len", "n_correct_in_place", "n_correct_any_position", "fully_correct",
        "sound_alike_errors", "look_alike_errors", "other_errors",
        "flagged_for_review",
    ]

    # Counterbalance block order across the group instead of running every
    # participant through the identical fixed sequence (which previously
    # confounded any condition effect with practice/fatigue over the
    # session). Free and serial recall are rotated with different offsets
    # so the two blocks' orders aren't perfectly correlated with each other.
    free_order = counterbalanced_order(FREE_RECALL_CONDITIONS, order_index)
    serial_order = counterbalanced_order(
        SERIAL_RECALL_CONDITIONS, None if order_index is None else order_index + 1
    )
    print("\nBlock order for this participant:")
    print("  Free recall:  ", [c for c, _ in free_order])
    print("  Serial recall:", [c for c, _ in serial_order])
    print("(This is logged per-trial as 'block_position' so order effects can be checked later.)")

    start_time = time.time()

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        print("=" * 60)
        print("PILOT EXPERIMENT -- Human Memory")
        print("=" * 60)

        print("\n--- Free recall block ---")
        for i, (condition, kwargs) in enumerate(free_order, start=1):
            if i > 1:
                take_a_break()
            run_free_recall_block(writer, condition, block_position=i, **kwargs)

        take_a_break("Free recall done. Take a longer break if you'd like, "
                      "then press Enter to start serial recall...")

        print("\n--- Serial recall block ---")
        for i, (condition, kwargs) in enumerate(serial_order, start=1):
            if i > 1:
                take_a_break()
            run_serial_recall_block(writer, condition, seq_len=SERIAL_BASE_LEN,
                                     block_position=i, **kwargs)

    elapsed_min = (time.time() - start_time) / 60
    print(f"\nDone. Data saved to: {output_file}")
    print(f"Total session time: {elapsed_min:.1f} minutes.")
    print("Verify the file above looks correct (this checks 'data saves correctly').")
    return output_file


def main():
    log_path = f"pilot_console_log_{datetime.now():%Y%m%d_%H%M%S}.txt"
    real_stdout = sys.stdout
    with open(log_path, "w") as logf:
        sys.stdout = _Tee(real_stdout, logf)
        try:
            output_file = collect_data()

            # To pool multiple participants once the group has run this script
            # several times, replace the line below with:
            #   analyze_pilot_data(["pilot_data_p1_....csv", "pilot_data_p2_....csv", ...])
            analyze_pilot_data(output_file, out_dir="pilot_figures")

            print("\nUse these pilot results (and the p-values above) to judge whether "
                  "TRIALS_PER_CONDITION is enough, or needs to be increased for the "
                  "full experiment.")
        finally:
            sys.stdout = real_stdout

    print(f"\nFull console output (every prompt, score, and statistic printed above) "
          f"was also saved to: {log_path}")


if __name__ == "__main__":
    main()
