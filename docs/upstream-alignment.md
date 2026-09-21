# Upstream protocols and the mixed experiment

References are pinned to Laya `d113dca2512fb3eaca313534bc54c7162d87c1d4`,
Laya Vision `86ccec115ef3d72d1851168fc9b7194e8dbed35a`, and Bub main
`9bf70488e22a44eeb875a9d55d9a7d352b82e381`. The training mixture includes upstream public tasks and soft teacher
labels. Existing Laya weights are benchmark references, not training targets.

## Publicly documented training coverage

Laya's pinned README and `BENCHMARKS.md` identify AG News, BoolQ, email spam,
phishing, RAG relevance and support triage as in its training mixture. Its
application builders identify the corresponding public sources as
`fancyzhx/ag_news`, `google/boolq`, `SetFit/enron_spam`,
`zefang-liu/phishing-email-dataset`, `microsoft/ms_marco`, and
`Tobi-Bueck/customer-support-tickets`. This is evidence of declared source/task
coverage, not a complete per-checkpoint training manifest, split audit or proof
of identical training records for the English and multilingual checkpoints.

The typed-decisions fine-tuning notebook explicitly loads
`LocalLLaMA/typed-decisions/all/train`: 1,200 cases, 6,000 original decisions.
The published test is 400 cases and 2,000 decisions. Teacher-derived distributions
measure agreement with those teachers rather than independently verified
business outcomes.

MASSIVE and XNLI appearing in the multilingual evaluation does not establish
that Laya trained on them. DAIR Emotion, SST-5, prompt-injections, toxic-chat
and domain-based model routing are described as held-out evaluation tasks.
The public sources inspected do not provide a complete base-training ID list.

The support dataset's current card does not substantiate verified real enterprise
ticket provenance and advertises a synthetic ticket generator. Its CC BY-NC 4.0
terms also differ from the Laya model license. Using a task in an upstream
benchmark does not certify its annotation quality or commercial training rights.

## Evaluation alignment

| Reference | Actual evaluation/training practice | Dohnuts treatment |
| --- | --- | --- |
| [Laya benchmarks](https://github.com/NandhaKishorM/laya/blob/d113dca2512fb3eaca313534bc54c7162d87c1d4/BENCHMARKS.md) | MASSIVE uses 20 candidate intents across languages; XNLI uses three entailment labels; public classifiers include AG News, emotion and BANKING77 | Train/evaluate MASSIVE en-US and zh-CN with all 60 intents; XNLI en/zh; AG News 4, emotion 6, BANKING77 77. These are not the published 51-language/20-candidate MASSIVE setting. |
| [typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) | 1,200 training cases / 6,000 decisions and 400 test cases / 2,000 decisions; labels average three teacher samples | Keep state-level groups and soft distributions. Results measure teacher agreement, not independent real-world correctness. Reserve train groups for development and calibration. |
| [Laya Vision model card](https://github.com/r33drichards/laya-vision/blob/86ccec115ef3d72d1851168fc9b7194e8dbed35a/hf_model_card.md) | A-OKVQA four-choice, ScienceQA image subset, VQAv2 yes/no soft targets; equal source sampling | Add all three to the mix. Use frozen vision plus LoRA instead of updating the entire language backbone. |
| Laya Vision VQAv2 | Published run re-splits official validation by image; 50k training / 5k evaluation examples | Download that labeled validation source, split by image hash and audit COCO overlap with A-OKVQA. Our ID list and counts differ; do not present as an exact published-result reproduction or official VQA score. |
| Laya Vision calibration | Per-type temperature, top-label accuracy, NLL, 15-bin ECE; training-tail calibration rows | Use an independent grouped calibration partition. ECE uses maximum probability, not entropy-based API confidence. |
| Laya Vision quality checks | Four cyclic candidate rotations and image-dependence tests; published `score` head untrained | Preserve these audits; add supervised CLEVR count/attribute/existence and ScreenQA region decisions. Counting alone does not validate subjective ordinal judgments. |
| Laya latency | Batch sizes 1, 5, 10, 50; published T4 numbers | Measure the local RX 7900 XTX, with model-specific token counts and all timing samples. Published Jev figures are third-party references, not a local Jev benchmark. |
| [JevBench v1.2.2](https://github.com/fstandhartinger/jevbench/tree/v1.2.2) | 534 decisions; intelligence, calibration, speed, and cost carry equal weights | Run all 231 public tasks with the frozen checkpoint and official harness. Compare reference accuracy on identical IDs. The other 303 tasks and a local provider tariff are unavailable, so no official rank or full composite is reported. |

Dataset revisions, file sizes, URLs and SHA-256 hashes are stored in
`data/manifests/upstream-downloads*.json`. Source conversion uses raw annotations
and Parquet, not downloaded executable dataset scripts. The official A-OKVQA
annotation release supplies COCO IDs missing from its HF Parquet conversion.

Train, development, calibration and final evaluation groups are disjoint after
joining source IDs, normalized text, pixel hashes, and COCO IDs. If official or
independently derived partitions conflict, retain the higher-priority partition
(test > development > calibration > train) and exclude lower-priority examples.
Report these exclusions rather than silently moving rows into the test set.

All 77 BANKING77 labels fit the API's 128-candidate limit. A separate
2,048-token training budget determines dataset eligibility. The Qwen3.5 adapter
supports 4,096-token inference. Eligibility is frozen before training; no
candidate or question is truncated to manufacture a passing result. Screening
counts and excluded IDs accompany accuracy.

ScienceQA adds noncommercial/share-alike source terms to the research mix.
The typed-decisions source explicitly warns that a better model can disagree
with its small teacher. Model quality claims must therefore separate human,
programmatic, and teacher-derived targets. Annotation terms alone do not settle
the license of a released mixed checkpoint.

ScienceQA uses the official **test** image subset (2,017 rows) for our final
evaluation. Laya Vision's reported 2,097-row set corresponds to official
validation. Our development partition starts from that validation split, with
cross-partition image overlap removed; the published accuracy is not a directly
matched final-test baseline.

Local upstream checkpoint references also have a training-exposure limitation:
our A-OKVQA development rows come from the official training set used by the
published Vision model, and our VQAv2 split is independently derived from the
same validation pool that upstream re-split for training. These scores are useful
implementation checks, but they do not establish an uncontaminated head-to-head
comparison. Official held-out partitions and any verifiable upstream training
ID exclusions must be identified before making a competitive quality claim.
Reference CPU evaluation uses the published FP32 weights and temperatures, with
2048-token total/head budgets to retain all candidates; upstream still applies
its 48-token per-option cap. Dohnuts uses BF16 with its own full-candidate template.
