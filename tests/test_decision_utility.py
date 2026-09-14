"""Tests for the C2 analysis path (Plan v2, contribution C2).

This code produces the paper's headline number, so the failure modes worth pinning down
are the quiet ones — the ones that yield a plausible number rather than an exception.

The one that would actually have bitten: flips must be compared by feature NAME, not by
answer letter. Letters are positional, and the attacked condition re-ranks the list, so
comparing letters would count a flip every time the ordering moved even when the agent
picked the same feature. That inflates the headline in exactly the direction we want it
to go, which is why it gets a test.
"""
from __future__ import annotations

import numpy as np

from avert.eval.decision_utility import (
    PROMPT_VARIANTS,
    DecisionCase,
    build_cases,
    decide_from_logits,
    summarize_cells,
)


def _case(cond="clean", candidates=None, causal=("f0",)):
    candidates = candidates or [f"f{i}" for i in range(4)]
    return DecisionCase(
        sample_id="s1", dataset="d", predicted_class="UDPFlood", condition=cond,
        attack="none", candidates=candidates, values=[1.0] * len(candidates),
        importances=[0.9, 0.5, 0.3, 0.1][: len(candidates)], causal=list(causal),
    )


def test_correctness_is_membership_in_the_causal_set():
    c = _case(causal=("f2",))
    assert c.is_correct(2)
    assert not c.is_correct(0)


def test_causal_visible_flags_when_the_answer_is_off_screen():
    assert _case(causal=("f1",)).causal_visible
    assert not _case(causal=("f99",)).causal_visible


def test_deterministic_decision_is_the_argmax():
    c = _case()
    d = decide_from_logits(c, np.array([0.1, 3.0, 0.2, 0.0]), temperature=None)
    assert d.choice_idx == 1 and d.choice == "f1"
    assert abs(sum(d.probs) - 1.0) < 1e-6


def test_sampling_is_reproducible_and_seed_sensitive():
    c = _case()
    logits = np.array([1.0, 1.0, 1.0, 1.0])          # flat: sampling must actually vary
    a = [decide_from_logits(c, logits, 1.0, seed=7).choice for _ in range(3)]
    assert len(set(a)) == 1, "same seed must give the same decision"
    across = {decide_from_logits(c, logits, 1.0, seed=s).choice for s in range(40)}
    assert len(across) > 1, "different seeds must be able to give different decisions"


def test_flips_compare_feature_names_not_letter_positions():
    """The attacked arm re-ranks the list. Comparing letters would score a flip whenever
    the ordering moved, even though the agent chose the same feature."""
    clean = _case("clean", candidates=["f0", "f1", "f2", "f3"])
    # same features, reversed order: 'f0' is now at position 3 instead of 0
    attacked = _case("attacked", candidates=["f3", "f2", "f1", "f0"])
    dc = decide_from_logits(clean, np.array([5.0, 0, 0, 0]), None)          # picks f0 (letter A)
    da = decide_from_logits(attacked, np.array([0, 0, 0, 5.0]), None)       # picks f0 (letter D)
    assert dc.choice == da.choice == "f0"
    assert dc.choice_idx != da.choice_idx, "the letter moved even though the feature did not"

    rows = [
        {"judge": "j", "dataset": "d", "class": "c", "attack": "A1_displacement", "seed": 0,
         "sample_id": "s1", "condition": "clean", "det_choice": dc.choice, "det_correct": True,
         "det_confidence": 1.0, "smpa_choice": dc.choice, "smpb_choice": dc.choice},
        {"judge": "j", "dataset": "d", "class": "c", "attack": "A1_displacement", "seed": 0,
         "sample_id": "s1", "condition": "attacked", "det_choice": da.choice, "det_correct": True,
         "det_confidence": 1.0, "smpa_choice": da.choice, "smpb_choice": da.choice},
    ]
    cell = summarize_cells(rows)[0]
    assert cell["attacked__flip"] == 0.0, "same feature chosen: this is not a flip"


def test_control_arm_is_zero_when_the_judge_does_not_waver():
    rows = [{"judge": "j", "dataset": "d", "class": "c", "attack": "A1_displacement", "seed": 0,
             "sample_id": f"s{i}", "condition": "clean", "det_choice": "f0", "det_correct": True,
             "det_confidence": 1.0, "smpa_choice": "f0", "smpb_choice": "f0"} for i in range(5)]
    assert summarize_cells(rows)[0]["control__flip_smp"] == 0.0


def test_ranking_withheld_arm_hides_importances_but_keeps_the_features():
    names = [f"f{i}" for i in range(12)]
    vals = np.arange(12, dtype=float)
    clean_attr = np.linspace(1.0, 0.0, 12)
    attacked_attr = np.linspace(0.0, 1.0, 12)
    cases = build_cases("s1", "d", "UDPFlood", "A1_displacement", names, vals,
                        clean_attr, attacked_attr, causal=["f0"], n_shown=10)

    assert set(cases) == {"clean", "attacked", "no_explanation"}
    withheld = cases["no_explanation"]
    assert withheld.importances is None, "the ranking must be removed"
    # ...but the same features stay on screen, which is why it is a conservative floor
    assert set(withheld.candidates) == set(cases["clean"].candidates)
    assert len(withheld.candidates) == 10


def test_repaired_arm_appears_only_when_a_repaired_attribution_is_given():
    names = [f"f{i}" for i in range(12)]
    vals = np.arange(12, dtype=float)
    attr = np.linspace(1.0, 0.0, 12)
    with_repair = build_cases("s1", "d", "c", "A3_scaffolding", names, vals, attr, attr,
                              causal=["f0"], repaired_attr=attr, n_shown=10)
    assert "repaired" in with_repair


def test_prompt_variants_change_wording_but_not_the_candidate_list():
    c = _case()
    rendered = {v: c.prompt(v) for v in PROMPT_VARIANTS}
    assert len({*rendered.values()}) == len(PROMPT_VARIANTS), "variants must differ"
    for text in rendered.values():
        for i, name in enumerate(c.candidates):
            assert f"{'ABCD'[i]}) {name}" in text, "same options, same letters, same order"


def test_summarize_pairs_within_a_cell_not_across_cells():
    """Two seeds share a sample_id. Pairing across them would compare unrelated decisions."""
    def row(seed, cond, choice):
        return {"judge": "j", "dataset": "d", "class": "c", "attack": "A1_displacement",
                "seed": seed, "sample_id": "s1", "condition": cond, "det_choice": choice,
                "det_correct": True, "det_confidence": 1.0,
                "smpa_choice": choice, "smpb_choice": choice}
    cells = summarize_cells([row(0, "clean", "f0"), row(0, "attacked", "f0"),
                             row(1, "clean", "f1"), row(1, "attacked", "f9")])
    by_seed = {c["seed"]: c for c in cells}
    assert by_seed[0]["attacked__flip"] == 0.0
    assert by_seed[1]["attacked__flip"] == 1.0


def test_paired_effect_counts_broken_and_fixed_separately():
    """A mean delta of zero can mean nothing happened, or that the attack broke as many
    decisions as it accidentally fixed. Those are different findings."""
    from avert.eval.decision_utility import paired_effect

    def row(i, cond, correct):
        return {"judge": "j", "dataset": "d", "class": "c", "seed": 0, "attack": "A1",
                "sample_id": f"s{i}", "condition": cond, "det_correct": correct}

    rows = []
    for i in range(10):                       # 5 broken, 5 fixed -> mean 0, discordant 10
        rows += [row(i, "clean", i < 5), row(i, "attacked", i >= 5)]
    e = paired_effect(rows, "attacked")
    assert e["n"] == 10 and abs(e["mean_delta"]) < 1e-9
    assert e["broke"] == 5 and e["fixed"] == 5 and e["discordant"] == 10
    assert not e["harms"], "a zero mean is not harm, however much churn there is"


def test_paired_effect_flags_harm_only_when_the_interval_clears_zero():
    from avert.eval.decision_utility import paired_effect

    def pair(i, clean_ok, atk_ok):
        return [{"judge": "j", "dataset": "d", "class": "c", "seed": 0, "attack": "A1",
                 "sample_id": f"s{i}", "condition": "clean", "det_correct": clean_ok},
                {"judge": "j", "dataset": "d", "class": "c", "seed": 0, "attack": "A1",
                 "sample_id": f"s{i}", "condition": "attacked", "det_correct": atk_ok}]

    strong = [r for i in range(40) for r in pair(i, True, i >= 30)]   # 30/40 broken
    assert paired_effect(strong, "attacked")["harms"]

    weak = [r for i in range(40) for r in pair(i, True, i >= 1)]      # 1/40 broken
    assert not paired_effect(weak, "attacked")["harms"]


def test_paired_effect_interval_widens_when_observations_are_clustered():
    """The interval must be a cluster bootstrap, not a paired t-CI.

    Construct data where every cell is internally identical but cells disagree with each
    other. Independent-observation statistics see a huge n and a tiny standard error;
    the truth is that there are only a handful of independent units. If someone swaps the
    cluster bootstrap back out for a t-CI, this test fails.
    """
    from avert.eval.decision_utility import paired_effect

    rows = []
    for cell_i in range(6):
        broke_all = cell_i < 3                    # 3 cells fully broken, 3 untouched
        for i in range(50):                       # 50 flows per cell -> n=300, clusters=6
            base = {"judge": "j", "dataset": "d", "class": "c", "seed": cell_i,
                    "attack": "A1", "sample_id": f"s{i}"}
            rows += [{**base, "condition": "clean", "det_correct": True},
                     {**base, "condition": "attacked", "det_correct": not broke_all}]
    e = paired_effect(rows, "attacked")
    assert e["n"] == 300 and e["n_clusters"] == 6
    assert e["ci_half_width"] > 2 * e["ci_half_width_naive"], (
        "clustered interval must be materially wider than the naive one when all the "
        "variation lives between cells rather than within them")


def test_floor_sweep_reports_every_threshold_asked_for():
    from avert.eval.decision_utility import competent_cell_set, floor_sweep

    per_cell = [{"judge": "j", "dataset": "d", "class": "c", "attack": "A1_displacement",
                 "seed": s, "clean__acc": 0.9 if s < 4 else 0.1} for s in range(6)]
    assert len(competent_cell_set(per_cell, 0.5)) == 4
    assert len(competent_cell_set(per_cell, 0.0)) == 6

    rows = []
    for s in range(6):
        for i in range(20):
            base = {"judge": "j", "dataset": "d", "class": "c", "seed": s,
                    "attack": "A1_displacement", "sample_id": f"s{i}"}
            rows += [{**base, "condition": "clean", "det_correct": True},
                     {**base, "condition": "attacked", "det_correct": i >= 5}]
    sweep = floor_sweep(rows, per_cell, floors=(0.0, 0.5))
    assert [r["floor"] for r in sweep] == [0.0, 0.5]
    assert all(r["n_clusters"] > 0 for r in sweep)


def test_seed_clustered_ci_is_wider_than_the_naive_one_when_seeds_disagree():
    """The matrix runs 3 classes per seed sharing one detector. Treating them as 9
    independent observations understates the interval; this fails if anyone reverts to
    a flat t-interval over all cells."""
    import numpy as np

    from avert.eval.metrics import mean_ci, seed_clustered_ci

    # three seeds, three classes each; classes within a seed agree, seeds disagree
    vals = np.array([0.30, 0.31, 0.29,   0.50, 0.51, 0.49,   0.70, 0.71, 0.69])
    seeds = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    mu_flat, hw_flat = mean_ci(vals)
    mu_clu, hw_clu = seed_clustered_ci(vals, seeds)

    assert abs(mu_flat - mu_clu) < 1e-9, "the point estimate must not move"
    assert hw_clu > hw_flat, "clustering by seed must widen the interval, not narrow it"


def test_seed_clustered_ci_degenerates_safely_on_one_seed():
    import numpy as np

    from avert.eval.metrics import seed_clustered_ci

    mu, hw = seed_clustered_ci(np.array([0.4, 0.5, 0.6]), np.array([0, 0, 0]))
    assert abs(mu - 0.5) < 1e-9 and hw == 0.0


def test_shuffling_permutes_the_options_but_keeps_each_feature_with_its_own_numbers():
    """The shuffled arm exists to break position bias. It must move the ORDER without
    detaching a feature from its value or its attribution, or it would be measuring
    something else entirely."""

    c = DecisionCase(
        sample_id="s1", dataset="d", predicted_class="UDPFlood", condition="clean",
        attack="none", candidates=[f"f{i}" for i in range(8)],
        values=[float(i) for i in range(8)],
        importances=[1.0 - 0.1 * i for i in range(8)], causal=["f3"],
    )
    sh = c.shuffled(seed=1)

    assert set(sh.candidates) == set(c.candidates)
    assert sh.candidates != c.candidates, "the order must actually change"
    original = dict(zip(c.candidates, zip(c.values, c.importances)))
    for name, val, imp in zip(sh.candidates, sh.values, sh.importances):
        assert original[name] == (val, imp), f"{name} lost its value or attribution"
    assert sh.causal == c.causal

    # correctness must follow the new positions, not the old ones
    assert sh.is_correct(sh.candidates.index("f3"))
    assert not sh.is_correct(sh.candidates.index("f0"))


def test_shuffling_is_reproducible_and_seed_sensitive():
    c = DecisionCase(sample_id="s", dataset="d", predicted_class="c", condition="clean",
                     attack="none", candidates=[f"f{i}" for i in range(10)],
                     values=[0.0] * 10, importances=[0.0] * 10, causal=["f0"])
    assert c.shuffled(3).candidates == c.shuffled(3).candidates
    orders = {tuple(c.shuffled(s).candidates) for s in range(20)}
    assert len(orders) > 1, "different seeds must give different orders"


def test_shuffling_preserves_the_ranking_withheld_arm_having_no_importances():
    c = DecisionCase(sample_id="s", dataset="d", predicted_class="c",
                     condition="no_explanation", attack="A1", candidates=["a", "b", "c"],
                     values=[1.0, 2.0, 3.0], importances=None, causal=["a"])
    assert c.shuffled(0).importances is None


def test_shuffle_offset_is_stable_across_processes():
    """The per-case shuffle offset must not use Python's hash(): string hashing is
    randomised per process, so the shuffled arm would permute differently on every run and
    could never be reproduced from the cache."""
    import subprocess
    import sys

    prog = ("import sys; sys.path.insert(0, 'scripts');"
            "from run_decision_utility import stable_offset;"
            "print(stable_offset('s12345'), stable_offset('flow-7'))")
    outs = {subprocess.run([sys.executable, "-c", prog], capture_output=True,
                           text=True, cwd=".").stdout.strip() for _ in range(3)}
    assert len(outs) == 1, f"offset differs across processes: {outs}"
    assert outs != {""}, "the helper did not run"


def test_stratified_slice_covers_every_attack_class():
    """Stride sampling aliased onto one attack class and silently dropped the rest.

    The cache lists attacks on a repeating period, so any stride that is a multiple of that
    period selects a single class. It happened: stride 9 over 5400 case-sets produced 4800
    A1_misdirection rows and zero A1_displacement -- the attack carrying the entire result --
    and the arms built on that slice looked entirely plausible.
    """
    import sys
    sys.path.insert(0, "scripts")
    from run_decision_utility import stratified_slice

    attacks = ["A1_misdirection", "A1_displacement", "A3_scaffolding"]
    cached = [{"attack": attacks[i % 3], "sample_id": f"s{i}"} for i in range(5400)]

    for n in (600, 300, 90, 30):
        sl = stratified_slice(cached, n)
        seen = {r["attack"] for r in sl}
        assert seen == set(attacks), f"n={n} covered only {seen}"

    # the old approach, kept here as the thing that must never come back
    aliased = {r["attack"] for r in cached[:: len(cached) // 600][:600]}
    assert len(aliased) == 1, "the regression this test exists for"


def test_stratified_slice_is_deterministic():
    import sys
    sys.path.insert(0, "scripts")
    from run_decision_utility import stratified_slice

    cached = [{"attack": ["a", "b", "c"][i % 3], "sample_id": f"s{i}"} for i in range(300)]
    a = [r["sample_id"] for r in stratified_slice(cached, 60)]
    b = [r["sample_id"] for r in stratified_slice(cached, 60)]
    assert a == b


# ---------------------------------------------------------------- pre-registered rule
# The rule that decides which sentence C2 gets to make used to be a literal string in
# `make_c2_report.py` saying "outcome B". It was true of the three-seed run and kept printing
# through the corrected five-seed re-run, where the all-cells shuffled intervals contain zero
# -- the table above the sentence contradicted the sentence and nothing failed. These pin the
# evaluator that replaced it, including the two verdicts it had never once returned.

def _eff(judge, lo, hi, mean, ranked):
    return {"judge": judge, "ci_lo": lo, "ci_hi": hi, "mean_delta": mean, "ranked": ranked}


def _decide(*args):
    import sys
    sys.path.insert(0, "scripts")
    from make_c2_report import decide_outcome
    return decide_outcome(*args)


def test_outcome_b_needs_every_shuffled_interval_to_clear_zero():
    """Retention alone is not outcome B. This is the exact shape that was asserted wrongly."""
    effects = [_eff("q", -0.231, -0.045, -0.135, -0.149), _eff("p", -0.176, -0.030, -0.099, -0.128)]
    assert _decide(0.705, 0.658, 0.349, effects)[0] == "B"

    # one interval crossing zero is enough to drop to C, however small the crossing
    effects[1] = _eff("p", -0.109, +0.043, -0.035, -0.080)
    outcome, lines = _decide(0.705, 0.658, 0.349, effects)
    assert outcome == "C"
    assert any("mediated share" in ln for ln in lines), "outcome C must report the share"


def test_outcome_a_is_reachable():
    """Clean accuracy at chance with no surviving effect: pure rank mediation."""
    effects = [_eff("q", -0.09, +0.05, -0.02, -0.14), _eff("p", -0.07, +0.06, -0.01, -0.11)]
    outcome, lines = _decide(0.705, 0.36, 0.349, effects)
    assert outcome == "A"
    assert any("No claim about judgment" in ln for ln in lines)


def test_low_retention_is_not_outcome_b_even_when_effects_clear_zero():
    effects = [_eff("q", -0.20, -0.05, -0.12, -0.15)]
    assert _decide(0.705, 0.45, 0.349, effects)[0] == "C"


def test_mediated_share_is_the_registered_formula():
    """(ranked - shuffled) / ranked, printed as the share position carries."""
    effects = [_eff("q", -0.10, +0.02, -0.050, -0.100)]
    _outcome, lines = _decide(0.705, 0.658, 0.349, effects)
    assert any("mediated share 50%" in ln for ln in lines)
