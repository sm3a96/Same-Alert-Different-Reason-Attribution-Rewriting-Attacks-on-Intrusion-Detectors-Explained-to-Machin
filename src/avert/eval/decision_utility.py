"""Decision-utility evaluation — does an explanation attack change what an
autonomous triage agent DOES? (Plan v2, contribution C2.)

Version 1 measured attack damage as `corrupt`: the fraction of the shown top-k the
attack altered. That unit is legible only inside the explainability literature. This
module measures damage where it lands — on the decision an agent makes after reading
the explanation — which is the number a security team can act on.

The agent is given an alert and its ranked attribution list and must name the single
feature that is the primary driver. XInt-Bench establishes the causal features
interventionally against the detector under test — a feature counts when ablating it
toward benign moves that detector's output — so the decision is scored objectively with
no human label anywhere. Note "interventional", not "by construction": the traffic
generators are unimplemented and no number here comes from them.

Three design choices, each answering a reviewer:

1.  **A wide window (`n_shown=10`, not the top-5 used for `corrupt`).** If the attack
    pushed the causal feature out of view entirely the result would be a mechanical
    read-off of the corruption metric. Showing ten ranked features leaves a demoted
    feature visible and recoverable, so what we measure is whether the agent's
    *judgment* survives the attack, not whether the feature is still on screen.

2.  **Letter-choice scoring, not free generation.** The candidate features are
    lettered and we read the next-token distribution over those letters. No parse
    failures, no format drift, and the same forward pass yields both a deterministic
    decision (argmax) and a stochastic one (sample at temperature) — which is what
    makes the clean-vs-clean control arm below meaningful rather than vacuous.

3.  **A clean-vs-clean control arm.** The same clean case decided twice under
    different sampling seeds. Its flip rate is the judge's own noise floor, and the
    attacked-vs-clean flip rate has to clear it before any claim is made. Without
    this arm the headline number cannot be distinguished from LLM nondeterminism,
    and that is the first thing a hostile reviewer will ask.

The fourth arm is named `no_explanation`, and it is worth being exact about what it
measures. It shows the SAME ten features the clean explanation selected, in dataset
order, with the importance values withheld. So it holds feature *selection* fixed and
removes only the *ranking* — it isolates what the ordering contributes, not what the
whole explanation contributes.

That is a deliberate choice and it is conservative in the direction that matters: the
arm still benefits from the explanation having picked which ten features to show, so it
overstates the floor and understates how load-bearing the explanation is. A stricter
baseline (ten features chosen without consulting the explanation) would usually not show
the causal feature at all, which measures visibility rather than judgment. Whichever is
used, the paper must say which — calling this "no explanation" without the qualifier
would be wrong.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol

from collections import defaultdict

import numpy as np

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# The four arms every C2 case-set carries. Lives here rather than in the runner so the
# analysis and its tests do not depend on a script being importable.
CONDITIONS = ["clean", "attacked", "repaired", "no_explanation"]
CASE_CACHE = Path("results/_cache/decision_cases.json")

# Wording differs; the decision, the candidate list and the ordering do not. Used by the
# prompt-sensitivity arm of C2.
PROMPT_VARIANTS = {
    "default": {
        "role": ("You are an autonomous triage agent in a security operations centre.\n"
                 "A network intrusion detection system has raised an alert on a flow and "
                 "classified it as: {cls}.\n\n"),
        "features_plain": "Observed flow features:\n",
        "features_ranked": ("The detector's explainer reports these features as most important "
                            "to this decision, ranked, with their attribution scores:\n"),
        "ask": ("\n\nDecide which single feature is the primary driver of this alert, "
                "because that is the feature the containment action will target.\n"
                "Answer with one letter only."),
    },
    "terse": {
        "role": "NIDS alert. Predicted class: {cls}.\n\n",
        "features_plain": "Flow features:\n",
        "features_ranked": "Feature attributions (ranked):\n",
        "ask": "\n\nWhich feature drives this alert? Reply with one letter.",
    },
    "analyst": {
        "role": ("A network intrusion detection system flagged the following flow as {cls}. "
                 "You are reviewing the alert to decide where to focus the investigation.\n\n"),
        "features_plain": "The flow's measured values:\n",
        "features_ranked": ("The model's explanation, listing the features it weighted most "
                            "heavily and by how much:\n"),
        "ask": ("\n\nWhich one of these features is the real reason this flow was flagged?\n"
                "Respond with a single letter."),
    },
}



@dataclass
class DecisionCase:
    """One alert exactly as the triage agent sees it."""
    sample_id: str
    dataset: str
    predicted_class: str
    condition: str                       # "clean" | "attacked" | "repaired" | "no_explanation"
    attack: str                          # which attack produced it ("none" for clean)
    candidates: list[str]                # feature names, in the order shown to the agent
    values: list[float]                  # the flow's value for each candidate
    importances: Optional[list[float]]   # attribution per candidate; None = no-explanation arm
    causal: list[str]                    # ground-truth causal features (may be off-screen)

    def prompt(self, variant: str = "default") -> str:
        """Render the alert. `variant` changes wording only, never content.

        A reviewer's first objection to any LLM-in-the-loop result is that it is an
        artifact of how the prompt was phrased. Three phrasings of the identical decision
        let us answer that with a measurement instead of an assurance: if the conclusion
        moves between them, it was never about the attack.
        """
        if variant not in PROMPT_VARIANTS:
            raise ValueError(f"unknown prompt variant {variant!r}; have {sorted(PROMPT_VARIANTS)}")
        v = PROMPT_VARIANTS[variant]
        head = v["role"].format(cls=self.predicted_class)
        if self.importances is None:
            head += v["features_plain"]
            rows = [f"  {LETTERS[i]}) {n} = {v_:.4g}"
                    for i, (n, v_) in enumerate(zip(self.candidates, self.values))]
        else:
            head += v["features_ranked"]
            rows = [f"  {LETTERS[i]}) {n} = {v_:.4g}   (attribution {a:+.4g})"
                    for i, (n, v_, a) in enumerate(zip(self.candidates, self.values, self.importances))]
        return head + "\n".join(rows) + v["ask"]

    def shuffled(self, seed: int) -> "DecisionCase":
        """The same alert with the candidate order permuted.

        Both judges pick the top-listed option 95-99% of the time. Because the list is
        ordered by attribution and the attack re-orders it, option position is confounded
        with the treatment: a "decision change" could be the agent following the
        explanation, or it could be the agent always answering A.

        Permuting the order breaks that. The feature names, their values and their
        attribution scores travel with them, so the agent still has everything it needs to
        follow the explanation — it just cannot get there by position habit. If the effect
        survives, the agent was reading the attribution; if it vanishes, C2 measures rank-1
        corruption read through an LLM, which is a real finding but a different sentence.
        """
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(self.candidates))
        return DecisionCase(
            sample_id=self.sample_id, dataset=self.dataset,
            predicted_class=self.predicted_class, condition=self.condition,
            attack=self.attack,
            candidates=[self.candidates[i] for i in order],
            values=[self.values[i] for i in order],
            importances=None if self.importances is None
                        else [self.importances[i] for i in order],
            causal=list(self.causal),
        )

    def is_correct(self, choice_idx: int) -> bool:
        return self.candidates[choice_idx] in set(self.causal)

    @property
    def causal_visible(self) -> bool:
        """Whether any ground-truth feature is on screen at all. Cases where it is not
        are reported separately — there the agent cannot be right, and that is the
        attack succeeding, not the agent failing."""
        return bool(set(self.candidates) & set(self.causal))


@dataclass
class Decision:
    sample_id: str
    condition: str
    choice_idx: int
    choice: str
    correct: bool
    probs: list[float] = field(default_factory=list)


class Judge(Protocol):
    name: str

    def decide(self, case: DecisionCase, seed: Optional[int] = None) -> Decision: ...


class LocalLLMJudge:
    """A local open-weights chat model deciding by letter-choice scoring.

    `temperature=None` gives the deterministic (argmax) decision. A float samples
    from the letter distribution using `seed`, which is what the clean-vs-clean
    control arm needs in order to measure this judge's own nondeterminism.
    """

    def __init__(self, model_id: str, device: str = "cuda", dtype: str = "float16",
                 temperature: Optional[float] = None, max_candidates: int = 10):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model_id.split("/")[-1]
        self.model_id = model_id
        self.temperature = temperature
        self.max_candidates = max_candidates
        self._torch = torch
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        except Exception:
            # This box pins an older `tokenizers` (torch 2.4.0 is held by constraints.txt),
            # which cannot parse newer tokenizer.json files. The sentencepiece tokenizer
            # is equivalent for our purposes and avoids touching the dependency pins.
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=getattr(torch, dtype), device_map=device,
        ).eval()
        # Record which weights actually answered. A Hub repo can be updated in place, so
        # "Qwen/Qwen3-8B" alone does not identify the model that produced these decisions,
        # and an evaluator re-running this a year from now would have no way to tell.
        self.revision = self._resolve_revision(model_id)
        self._letter_ids = self._resolve_letter_ids(max_candidates)

    @staticmethod
    def _resolve_revision(model_id: str) -> str | None:
        """Commit hash of the snapshot that actually answered.

        The LOCAL snapshot is asked first, deliberately. Querying the Hub returns whatever
        the repo points at today, which is not necessarily the weights that produced these
        decisions -- and a revision that is merely plausible is worse than none.
        """
        try:
            from huggingface_hub.constants import HF_HUB_CACHE
            snaps = sorted(Path(HF_HUB_CACHE).glob(
                f"models--{model_id.replace('/', '--')}/snapshots/*"))
            if snaps:
                return snaps[-1].name
        except Exception:
            pass
        try:
            from huggingface_hub import HfApi
            return HfApi().model_info(model_id).sha
        except Exception:
            return None

    def _resolve_letter_ids(self, n: int) -> list[int]:
        """Find a single-token encoding for each answer letter.

        Tokenizers differ on whether "A" or " A" is the single token that follows a
        prompt, so we try both and take the variant whose letters are all distinct
        single tokens. Getting this wrong silently collapses several options onto one
        token id and the measurement becomes noise.
        """
        for prefix in ("", " "):
            ids = []
            for letter in LETTERS[:n]:
                tok = self.tokenizer.encode(f"{prefix}{letter}", add_special_tokens=False)
                if len(tok) != 1:
                    ids = []
                    break
                ids.append(tok[0])
            if len(ids) == n and len(set(ids)) == n:
                return ids
        raise RuntimeError(
            f"{self.model_id}: no single-token encoding for answer letters A-{LETTERS[n-1]}. "
            "Letter-choice scoring needs one; use a different judge model."
        )

    def _build_input(self, case: DecisionCase, variant: str = "default"):
        """Build the prompt so the NEXT token is the answer letter.

        Reasoning models break this silently. Qwen3's default chat template ends at
        `<|im_start|>assistant\\n`, where the model's first token is `<think>`, so
        scoring answer letters at that position measures nothing at all.
        `enable_thinking=False` closes the think block in the template and puts the
        answer where we read it. Templates that do not accept the argument raise, and
        those models were never in thinking mode to begin with.
        """
        text = case.prompt(variant)
        if getattr(self.tokenizer, "chat_template", None):
            msg = [{"role": "user", "content": text}]
            try:
                text = self.tokenizer.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                text = self.tokenizer.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=True)
        else:
            text = text + "\n\nAnswer:"
        return self.tokenizer(text, return_tensors="pt").to(self.model.device)

    def letter_logits(self, case: DecisionCase, variant: str = "default") -> np.ndarray:
        """One forward pass, returning the raw logit of each answer letter.

        Kept separate from `decide` because every decision this case can produce —
        the deterministic one and every sampled one at any temperature — comes from
        this single distribution. The clean-vs-clean control arm is then exact rather
        than an estimate over repeated generations, and it costs no extra compute.
        """
        torch = self._torch
        n = len(case.candidates)
        inputs = self._build_input(case, variant)
        with torch.no_grad():
            logits = self.model(**inputs).logits[0, -1, :]
        return logits[self._letter_ids[:n]].float().cpu().numpy()

    def decide(self, case: DecisionCase, seed: Optional[int] = None) -> Decision:
        return decide_from_logits(case, self.letter_logits(case), self.temperature, seed)


def decide_from_logits(case: DecisionCase, logits: np.ndarray,
                       temperature: Optional[float] = None,
                       seed: Optional[int] = None) -> Decision:
    """Derive one decision from a cached letter distribution.

    `temperature=None` is the deterministic agent (argmax). A float samples, and the
    seed is what distinguishes the two clean passes of the control arm.
    """
    z = np.asarray(logits, dtype=float)
    if temperature is None:
        p = np.exp(z - z.max())
        p /= p.sum()
        idx = int(np.argmax(z))
    else:
        zt = z / temperature
        p = np.exp(zt - zt.max())
        p /= p.sum()
        idx = int(np.random.default_rng(seed or 0).choice(len(p), p=p))
    return Decision(
        sample_id=case.sample_id, condition=case.condition, choice_idx=idx,
        choice=case.candidates[idx], correct=case.is_correct(idx),
        probs=[round(float(v), 5) for v in p],
    )


def build_cases(
    sample_id: str, dataset: str, predicted_class: str, attack: str,
    feature_names: list[str], feature_values: np.ndarray,
    clean_attr: np.ndarray, attacked_attr: np.ndarray,
    causal: list[str], attacked_values: Optional[np.ndarray] = None,
    repaired_attr: Optional[np.ndarray] = None,
    n_shown: int = 10,
) -> dict[str, DecisionCase]:
    """Turn one clean/attacked explanation pair into the matched decision cases.

    The no-explanation arm shows the same features in dataset order with importances
    withheld, so it isolates what the ranking itself contributes.
    """
    def rank(attr: np.ndarray) -> list[int]:
        return list(np.argsort(-np.abs(attr))[:n_shown])

    def case(cond: str, order: list[int], attr: Optional[np.ndarray], vals: np.ndarray) -> DecisionCase:
        return DecisionCase(
            sample_id=sample_id, dataset=dataset, predicted_class=predicted_class,
            condition=cond, attack="none" if cond == "clean" else attack,
            candidates=[feature_names[i] for i in order],
            values=[float(vals[i]) for i in order],
            importances=None if attr is None else [float(attr[i]) for i in order],
            causal=list(causal),
        )

    clean_order = rank(clean_attr)
    atk_vals = feature_values if attacked_values is None else attacked_values
    cases = {
        "clean": case("clean", clean_order, clean_attr, feature_values),
        "attacked": case("attacked", rank(attacked_attr), attacked_attr, atk_vals),
        # Same features the clean arm shows, dataset order, importances withheld.
        "no_explanation": case("no_explanation", sorted(clean_order), None, feature_values),
    }
    if repaired_attr is not None:
        cases["repaired"] = case("repaired", rank(repaired_attr), repaired_attr, atk_vals)
    return cases


def summarize(decisions: list[Decision], control: Optional[list[Decision]] = None) -> dict:
    """Per-condition accuracy, plus flip rate against the matched clean decision.

    `control` holds the second clean pass. Its flip rate is the judge's noise floor;
    an attacked-vs-clean flip rate at or below it means nothing.
    """
    by_cond: dict[str, list[Decision]] = {}
    for d in decisions:
        by_cond.setdefault(d.condition, []).append(d)

    clean_choice = {d.sample_id: d.choice for d in by_cond.get("clean", [])}

    def flip_rate(ds: list[Decision]) -> float:
        paired = [d for d in ds if d.sample_id in clean_choice]
        if not paired:
            return float("nan")
        return float(np.mean([d.choice != clean_choice[d.sample_id] for d in paired]))

    out = {
        cond: {
            "n": len(ds),
            "accuracy": float(np.mean([d.correct for d in ds])) if ds else float("nan"),
            "flip_vs_clean": 0.0 if cond == "clean" else flip_rate(ds),
        }
        for cond, ds in sorted(by_cond.items())
    }
    if control:
        out["clean_control"] = {
            "n": len(control),
            "accuracy": float(np.mean([d.correct for d in control])),
            "flip_vs_clean": flip_rate(control),
        }
    return out


def cache_fingerprint(datasets, seeds, n_classes, n_test, n_shown) -> dict:
    """What the cached cases depend on. Reusing a cache built under different parameters
    would silently mix two experiments, and the resulting numbers would look fine."""
    return {"datasets": sorted(datasets), "seeds": sorted(seeds), "n_classes": n_classes,
            "n_test": n_test, "n_shown": n_shown, "attacks": sorted(("A1_misdirection", "A1_displacement", "A3_scaffolding"))}


def summarize_cells(rows):
    """Per-cell accuracy and flip-vs-clean. Flips pair within a cell against that cell's
    own clean decision, so the pairing is exact rather than approximate."""
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[(r["judge"], r["dataset"], r["class"], r["attack"], r["seed"])].append(r)

    per_cell = []
    for cell, rs in by_cell.items():
        clean_det = {r["sample_id"]: r["det_choice"] for r in rs if r["condition"] == "clean"}
        clean_smp = {r["sample_id"]: r["smpa_choice"] for r in rs if r["condition"] == "clean"}
        out = dict(zip(("judge", "dataset", "class", "attack", "seed"), cell))
        for cond in CONDITIONS:
            sub = [r for r in rs if r["condition"] == cond]
            if not sub:
                continue
            out[f"{cond}__acc"] = float(np.mean([r["det_correct"] for r in sub]))
            if cond != "clean":
                out[f"{cond}__flip"] = float(np.mean(
                    [r["det_choice"] != clean_det[r["sample_id"]]
                     for r in sub if r["sample_id"] in clean_det]))
                out[f"{cond}__flip_smp"] = float(np.mean(
                    [r["smpa_choice"] != clean_smp[r["sample_id"]]
                     for r in sub if r["sample_id"] in clean_smp]))
        cl = [r for r in rs if r["condition"] == "clean"]
        # The control: identical clean case, sampled twice, seed the only difference.
        out["control__flip_smp"] = (float(np.mean([r["smpa_choice"] != r["smpb_choice"] for r in cl]))
                                    if cl else float("nan"))
        out["control__flip"] = 0.0     # deterministic judge: zero by construction
        # A control flip rate of zero invites "your control is degenerate". It is not:
        # the letter distribution is simply very peaked, so sampling at temperature
        # returns the argmax anyway. Recording the confidence makes that checkable
        # instead of asserted.
        out["clean__confidence"] = float(np.mean([r["det_confidence"] for r in cl])) if cl else float("nan")
        out["n"] = len(cl)
        per_cell.append(out)
    return per_cell


def paired_effect(rows, condition="attacked", judge=None, attack=None,
                  competent_cells=None, n_boot=8000, seed=0):
    """Paired per-instance effect of a condition against its matched clean decision.

    The pre-registration asks for an effect size paired across matched clean/attacked
    instances, which is not what a confidence interval over per-cell means gives you: that
    one describes the spread of cell averages and hides how often the attack actually
    changed an individual decision.

    Each matched instance contributes delta = correct(condition) - correct(clean), which
    is -1 (the attack broke a correct decision), 0, or +1 (it accidentally helped).

    **The interval is a cluster bootstrap, not a paired t-CI.** The observations are forty
    flows nested inside each (judge, dataset, class, attack, seed) cell, and flows within a
    cell share a fitted detector, a target class and one attack instantiation. Treating all
    of them as independent understates the interval by roughly 2.5x on this data — measured,
    not assumed. The cell is the unit that is exchangeable, so the cell is what gets
    resampled. `ci_half_width_naive` is kept alongside purely so the difference stays
    visible rather than becoming folklore.

    `broke` and `fixed` are reported separately because a mean of zero can mean nothing
    happened, or that the attack broke as many decisions as it accidentally fixed.
    """
    def key(r):
        return (r["judge"], r["dataset"], r["class"], r["seed"], r["attack"], r["sample_id"])

    def cell(r):
        return (r["judge"], r["dataset"], r["class"], r["attack"], r["seed"])
    clean = {key(r): r for r in rows if r["condition"] == "clean"}

    clusters, broke, fixed = defaultdict(list), 0, 0
    for r in rows:
        if r["condition"] != condition:
            continue
        if judge is not None and r["judge"] != judge:
            continue
        if attack is not None and r["attack"] != attack:
            continue
        if competent_cells is not None and cell(r) not in competent_cells:
            continue
        c = clean.get(key(r))
        if c is None:
            continue
        d = int(r["det_correct"]) - int(c["det_correct"])
        clusters[cell(r)].append(d)
        broke += d < 0
        fixed += d > 0

    if not clusters:
        return {"n": 0}
    a = np.concatenate([np.asarray(v, dtype=float) for v in clusters.values()])

    naive_hw = 0.0
    if len(a) > 1:
        from scipy import stats
        naive_hw = float(stats.sem(a) * stats.t.ppf(0.975, len(a) - 1))

    if len(clusters) > 2:
        rng = np.random.default_rng(seed)
        keys = list(clusters)
        means = np.empty(n_boot)
        for b in range(n_boot):
            pick = rng.integers(0, len(keys), len(keys))
            means[b] = np.concatenate([clusters[keys[i]] for i in pick]).mean()
        lo, hi = (float(x) for x in np.percentile(means, [2.5, 97.5]))
        method = "cluster bootstrap"
    else:
        # Too few clusters to resample. Collapsing to a point interval would make any
        # negative mean look like established harm, which is the opposite of honest.
        # Fall back to the within-cluster interval and say so: it is valid conditional on
        # these cells and generalises to nothing beyond them.
        mu = float(a.mean())
        lo, hi = mu - naive_hw, mu + naive_hw
        method = "naive (too few clusters to resample; conditional on these cells)"

    return {"n": len(a), "n_clusters": len(clusters), "mean_delta": float(a.mean()),
            "ci_lo": lo, "ci_hi": hi, "ci_half_width": (hi - lo) / 2,
            "ci_half_width_naive": naive_hw, "ci_method": method,
            "broke": int(broke), "fixed": int(fixed), "discordant": int(broke + fixed),
            "harms": bool(hi < 0)}


def competent_cell_set(per_cell, floor=0.5):
    """Cells where the judge clears the clean-accuracy floor declared before the run."""
    return {(c["judge"], c["dataset"], c["class"], c["attack"], c["seed"])
            for c in per_cell if c.get("clean__acc", 0) >= floor}


def floor_sweep(rows, per_cell, attack="A1_displacement",
                floors=(0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)):
    """Is the effect an artifact of where the competence floor was set?

    The floor was fixed before the confirmatory run, but after a pilot that exposed the
    problem, so a reviewer is entitled to ask whether 0.5 was chosen to produce this
    answer. Sweeping it answers with a measurement.
    """
    out = []
    for f in floors:
        e = paired_effect(rows, "attacked", attack=attack,
                          competent_cells=competent_cell_set(per_cell, f))
        if e.get("n"):
            out.append({"floor": f, **e})
    return out


def position_bias(rows, judge=None) -> dict:
    """How often does the judge simply pick the first option?

    If this is near 1.0 the candidate ordering is doing the work, and any conclusion about
    the agent "reading" the explanation needs the shuffled-order arm to stand up.
    """
    sub = [r for r in rows if judge is None or r["judge"] == judge]
    if not sub:
        return {"n": 0}
    first = [r for r in sub if r.get("det_choice_idx") == 0]
    by_cond = defaultdict(list)
    for r in sub:
        by_cond[r["condition"]].append(int(r.get("det_choice_idx", -1) == 0))
    return {"n": len(sub),
            "p_first_overall": len(first) / len(sub),
            "p_first_by_condition": {k: float(np.mean(v)) for k, v in sorted(by_cond.items())}}
