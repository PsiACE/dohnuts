# Data and evaluation

Use public labels and programmatic scoring. The training mixture includes
public teacher-derived soft labels from typed-decisions; no new teacher calls
or LLM judges are required. Deterministic conversion means reproducible examples
and scores, not infallible annotations or known per-example true probabilities.

## Training sources

| Source | Decision tasks | Protocol |
| --- | --- | --- |
| [MASSIVE 1.1](https://huggingface.co/datasets/AmazonScience/massive) | English/Chinese `choice` | All 60 intents; fixed human-readable label mapping |
| [BoolQ](https://github.com/google-research-datasets/boolean-questions) | Text `noul` | Original passage, question, and Boolean answer |
| [CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/) | Visual `choice`, `noul`, `score` | Attributes, existence, and ordered counts |
| [ScreenQA](https://github.com/google-research-datasets/screen_qa) | Screenshot `choice` and candidate-level `noul` | Original human annotations in `answers_and_bboxes` |
| AG News / emotion / BANKING77 | Text `choice` | Full label vocabularies and official labeled test sets |
| XNLI | English/Chinese `choice` | Three labels; aligned translations and duplicate premises stay grouped |
| A-OKVQA / ScienceQA | Image `choice` | Four-choice A-OKVQA; image-only ScienceQA subset with official test retained |
| VQAv2 yes/no | Image `noul` | Soft answer votes; image-grouped split of labeled official validation |
| typed-decisions | All three decision types | Public soft teacher distributions; state-grouped partitions |

CLEVR scene graphs and programs verify labels but never enter model inputs.
Use complete answer domains per question family rather than answer-dependent
candidate sampling. Count performance does not establish subjective scoring
ability. English visual benchmarks do not establish Chinese visual competence.

### ScreenQA conversion

Build candidate regions from the screenshot's Rico View Hierarchy using a fixed,
answer-independent rule. Present region IDs and geometry consistently; do not
feed annotator descriptions or ground-truth answers into the input.

Retain questions whose annotations agree on one answer element that maps
unambiguously to a candidate. Record exclusion counts for multi-element answers,
disagreement, missing assets, and unmapped regions. Do not inject a target region
to rescue an otherwise invalid example.

- `choice`: select the matching region from the candidate set.
- `noul`: determine whether a specified candidate is the answer region.

A negative candidate is not evidence that the answer is absent from the screen.
Publish the negative-sampling rule and class balance; calibration is specific to
that distribution. Report these as derived decision tasks, not official ScreenQA
question-answering scores. The inference contract requires candidate regions;
automatic UI detection is outside scope.

ScreenQA Short and ComplexQA are excluded from the data mix. The former
contains model-produced answers, and the latter uses model-generated questions
and answers with human validation.

## Splits and reproducibility

- MASSIVE and ScreenQA retain official train/dev/test partitions. Reserve a
  group-based calibration subset from train.
- BoolQ and CLEVR use public labeled dev/val as frozen final evaluation sets,
  not as official test results. Split train into fitting, development, and
  calibration partitions.
- Keep all questions from one image, all language variants of one source
  utterance, and questions sharing a passage in the same partition. Audit
  official splits for overlap and document any exclusions.
- Store source revisions and hashes, sample IDs, split assignments, templates,
  filters, candidate permutations, sampling weights, and seeds in manifests.
- Pin preprocessing and metric implementations. Report cross-split duplicates
  and possible backbone pretraining contamination.

Select checkpoints and hyperparameters using development data only. Fit
temperatures on calibration data only. Freeze the recipe before final evaluation.
Seeded runs improve repeatability but do not guarantee bitwise equivalence across
GPU kernels or software versions.

## Evaluation

Report results by dataset, language, decision type, and candidate count:

- Accuracy and macro-F1 for decision quality.
- NLL and explicitly normalized Brier score for distributions.
- Ranked probability score and MAE for ordered counts.
- Fixed-bin ECE and reliability diagrams for calibration.
- Candidate-order sensitivity and correct-image/missing-image/mismatched-image
  checks for vision dependence; these are quality checks, not competing trainers.
- End-to-end p50/p95 latency, throughput, and peak VRAM on the local GPU, including
  preprocessing and with image, sequence, candidate, and batch sizes disclosed.

Complete calibration, held-out evaluation, and benchmarks for seed 42 before
replication. Report variation only when multiple seeds have completed evaluation;
for one seed, leave sample standard deviation unmeasured. All probabilities
must be finite, in range, and normalized where appropriate. Report filtered
coverage alongside accuracy. Do not equate a hard target with a known conditional
probability, or a concentrated distribution with justified confidence.

## Release assets and terms

Release the recipe, manifests, inference configuration, calibration parameters,
metrics, and model card alongside any weights. Keep raw data and checkpoints out
of Git. Retain source attribution and audit data and image terms before release.

MASSIVE and CLEVR identify CC BY 4.0 terms; BoolQ identifies CC BY-SA 3.0 terms.
ScreenQA annotations identify CC BY 4.0, but Rico screenshot terms must also be
checked. Do not infer a weight license solely from an annotation or code license.
ScienceQA is included in the research mixture. Existing Laya Vision
weights remain benchmark references. See [upstream alignment](upstream-alignment.md)
for source protocols and comparison limits.

## Business decision sources

| Source | Decision supervision | Evidence and limits |
| --- | --- | --- |
| [Amazon ESCI](https://github.com/amazon-science/esci-data) | Query/product `choice`: exact, substitute, complement, irrelevant; English, Spanish and Japanese | Actual customer search queries with manual relevance judgments. Use explicit E/S/C/I labels. The categories are not assumed to be an ordinal scale. Apache-2.0 project release. |
| [WikiQA](https://www.microsoft.com/en-us/research/publication/wikiqa-a-challenge-dataset-for-open-domain-question-answering/) | Query/passage answerability `noul` | Real Bing queries, Wikipedia sentences and crowdsourced binary judgments. Explicit negative labels; never equate missing judgments with negatives. Microsoft Research Data License: research/technology development restrictions apply. |
| [ShARC](https://sharc-data.github.io/data.html) | Agent `choice`: yes, no, irrelevant, ask for missing information | Public rule documents with crowdsourced scenarios/conversations. Real policy tasks, not a log of actual customer interactions. Evidence annotations and the target follow-up wording never enter model state. CC BY-SA 3.0. |
| [ContractNLI](https://stanfordnlp.github.io/contract-nli/) | Full-contract `choice`: entailment, contradiction, not mentioned | Human annotations on real contracts. Keep the entire document; exclude over-budget documents instead of selecting gold evidence spans. CC BY 4.0. |
| [SpamAssassin](https://spamassassin.apache.org/old/publiccorpus/readme.html) | Email spam `noul`; known ham supplies negative phishing examples | Public collected messages. Includes hard ham. Sender copyright remains with senders; the software's Apache license is not a blanket data license. This is a research corpus. |
| [Nazario phishing corpus](https://monkey.org/~jose/phishing/README.txt) | Positive phishing `noul`, paired with separately identified ham | Hand-classified personal-inbox messages; 2023 for training/calibration, 2024 development, 2025 test. CC BY 4.0. Positive/negative source and age differ; report cross-corpus and temporal limitations. |
| [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) | Message spam `noul` | Public collected SMS with spam/ham labels. Adds a short-message domain; it does not substitute for email evaluation. UCI release is CC BY 4.0. |

The mixture includes research-use sources, including ScienceQA. Dataset annotations, raw documents and mixed trained
weights have distinct rights; this document does not certify a commercial
checkpoint license.

## Conversion and isolation

`scripts/prepare_data.py` verifies every raw-file SHA-256 against
`data/manifests/enrichment-downloads.json` before conversion. It creates the
existing `{state, question, target}` schema. One-hot targets come from source
labels. No model generates pseudo-labels, confidence scores or explanations.

Source rules are fixed in the converter:

- ESCI uses the official challenging-query subset. Select 450 training queries
  and at most 60 queries per held-out partition per locale using stable hashes.
  Keep whole queries, with official test priority. Products may occur under
  multiple queries: this is unseen-query evaluation, not unseen-product evidence.
- ShARC groups by source page and snippet. ContractNLI groups by full contract.
  WikiQA joins query and source-document aliases before resolving split conflicts.
- Mail uses decoded bodies with the same plain-text/HTML conversion for both
  classes; labels, corpus names, spam-filter headers and acquisition timestamps
  are absent from state. URL/number-normalized body signatures group simple
  campaign variants. Attachments and links are never executed or fetched.
- Independent calibration comes from training groups. Original dev/test labels
  are never moved into training. Conflicting group partitions keep the higher
  priority partition and exclude the others.
- Frozen Laya/JevBench inputs, the sampled training inputs, and its
  held-out states are checked with punctuation-insensitive text hashes.
  Checks include Laya's 3,000-character body prefixes. Matching source groups are
  excluded. This does not prove absence of semantic or paraphrase duplicates.
- The fixed 2,048-token training budget is unchanged. Exclusions are counted.
  Serving still accepts 4,096 tokens. Results on the eligible ContractNLI subset
  must not be presented as scores on its complete official test set.

The training mixture combines 17 text/image groups and nine business task/language
groups. Original dev/calibration/test rows retain their partitions. The existing uniform group
sampler therefore allocates an expected 9/26 of updates to business tasks and 17/26
to the other text/image tasks. Its fixed 6,000-row training cap per group remains in effect.
No class rebalancing is hidden in calibration or evaluation. Phishing positives
and WikiQA positives are minorities; report recall, macro-F1, PR-AUC and class
counts alongside accuracy, NLL, Brier and ECE.


The fixed 26-group mixture lives in `data/processed/v1`. Its prepared partitions
contain 185,857 training, 64,843 development, 86,897 calibration and 180,031 test
rows. The per-group training cap yields 143,238 eligible training rows.
`manifest.json` records counts, exclusions and hashes. The frozen benchmark text
fingerprints in `data/manifests/evaluation-text-sha256.json` are exclusion inputs,
never training examples. `scripts/prepare_data.py` builds this mixture without
requiring a trained checkpoint or a previous training run.
