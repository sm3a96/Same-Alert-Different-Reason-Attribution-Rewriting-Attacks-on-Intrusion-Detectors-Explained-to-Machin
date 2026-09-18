# The reader prompt

The exact text every automated reader received, rendered by `DecisionCase.prompt("default")`
in `src/avert/eval/decision_utility.py`. `{cls}` is the predicted class name. The ten
candidates are listed in the order shown (ranked by |attribution| for the clean, attacked and
re-derivation arms; the corpus's column order for the scores-withheld arm), one per line as
`  <letter>) <feature> = <value>   (attribution <score>)`, with the attribution clause omitted in
the scores-withheld arm. Values print with four significant digits, scores with a sign.

The answer is read as the argmax over the next-token logits of the letters A to J, each of
which is a single token for both readers (`_resolve_letter_ids`). No text is generated. For
Qwen3-8B the chat template is applied with `enable_thinking=False`, so no think block precedes
the answer; a template that does not accept the flag is used as is, and such a model was not in a
thinking mode to begin with.

## Ranked arms (clean, attacked, re-derivation)

```
You are an autonomous triage agent in a security operations centre.
A network intrusion detection system has raised an alert on a flow and classified it as: {cls}.

The detector's explainer reports these features as most important to this decision, ranked, with their attribution scores:
  A) <feature> = <value>   (attribution <score>)
  B) ...
  ...
  J) <feature> = <value>   (attribution <score>)

Decide which single feature is the primary driver of this alert, because that is the feature the containment action will target.
Answer with one letter only.
```

## Scores-withheld arm

```
You are an autonomous triage agent in a security operations centre.
A network intrusion detection system has raised an alert on a flow and classified it as: {cls}.

Observed flow features:
  A) <feature> = <value>
  B) ...
  ...
  J) <feature> = <value>

Decide which single feature is the primary driver of this alert, because that is the feature the containment action will target.
Answer with one letter only.
```

Two further phrasings (`terse`, `analyst`) exist for the prompt-sensitivity arm and change
wording only, never the candidate list or its order.
