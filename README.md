# Multilingual grounding pilot

Two completed external-label studies separate question language, supplied English evidence and cultural image use. The recorded experiment has 24 AfriQA questions (12 Yoruba, 12 Swahili), 12 Yoruba Afri-MCQA image groups and three interface-only units. Two local configurations completed 234 requests. The current manuscript is **Source Use and Cultural Image Questions: Small External-Label Pilots in Yoruba and Swahili**, by Saurav K. Aryal, Howard University.

[Paper PDF](paper.pdf) · [Overleaf ZIP](overleaf.zip) · [Recorded results](results/observed/) · [Future recruitment evidence](recruitment/prolific_20261101/)

## Completed results

Each table cell contains 12 paired items. AfriQA requires a verbatim **English passage span** even when the question is in Yoruba or Swahili.

| Model | Yoruba native / English question EM | Swahili native / English question EM | Empty passage correct null |
| --- | --- | --- | --- |
| qwen3.5:4b | 3/12 / 5/12 | 7/12 / 6/12 | 24/24 |
| gemma4:26b | 7/12 / 6/12 | 6/12 / 6/12 | 24/24 |

| Model | Yoruba image/native | Image/English | No image/native |
| --- | ---: | ---: | ---: |
| qwen3.5:4b | 5/12 | 8/12 | 2/12 |
| gemma4:26b | 7/12 | 8/12 | 3/12 |

All 234 outputs have valid schemas and no transport errors. The 18 interface checks are excluded from 216 locked responses over 36 paired units. Question-language effects are mixed. Image/no-image net differences are +3 and +4 correct items, including individual losses. These results establish a reproducible interface; natural native-answer quality, source freshness, contextual appropriateness and community validation remain unmeasured. The November 1 update changes future work and documentation, not these observations.

## Verify saved results

The standard-library verifier recomputes every recorded response schema, strict and normalized labels, token F1 and aggregate table cell. It also checks the publication provenance and recruitment-evidence hashes. It performs no model or participant calls.

```sh
python3 verify_results.py
```

Regenerate the same two recorded-count figures:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python generate_figures.py
```

Figure fonts may vary across operating systems; the plotted counts come from the saved numeric results.

`data/model_results.json` contains the recorded counts and original result hashes. `results/observed/` contains projected raw responses, decisions and aggregates. Input projections preserve text/image hashes without redistributing their content. `PUBLICATION_PROVENANCE.json` links the original and public observations and records the November 1 documentation revision. `verification.json` records the local checks for this revision.

## Recreate inputs and optionally repeat inference

This command downloads official source inputs and images into a fresh local directory and checks the pinned study identities. It does not run models:

```sh
python prepare_data.py --out results/reproduced-inputs
```

The source pins are AfriQA Git revision `5c29933753692c28773c34ac1f68482007f1d2bb` and Afri-MCQA dataset-server revision `d5d4db17fc2703d761799746d258e2dbb6bc4b73`. The helper checks all 117 cases and 13 processed image packets. It stops if source revision, selected content or processed image bytes differ. Upstream availability is necessary; identities and exclusions are in `protocol/`.

For optional inference, start a compatible native local chat server at `127.0.0.1:11434` with the downloaded model tags, then run sequentially:

```sh
python -m pilots.strengthening.multilingual_external_qa --stage execute --out results/reproduced-inputs --model qwen3.5:4b
# After completion and unloading the first model:
python -m pilots.strengthening.multilingual_external_qa --stage execute --out results/reproduced-inputs --model gemma4:26b
```

Each model executes 117 cases once, without retries or answer repair, using temperature 0, seed 20261031, requested context 6144, output cap 128, JSON formatting and disabled thinking. Existing output directories are rejected. The observed Qwen artifact uses NVFP4/MLX and Gemma uses Q4_K_M/llama.cpp; digests and backend metadata are recorded. Another artifact/backend is a new configuration. The study did not benchmark 48 GB memory use, runtime or energy.

## Proposed next study

The revised future work uses a **Swahili core and gated Asante-compatible Twi replication**, with English reference questions. It is separate from the completed Yoruba/Swahili benchmark diagnostic. No Twi experiment or new human study has been conducted.

The [dated Prolific snapshot](recruitment/prolific_20261101/) records 701 first-language and 1,872 fluent Swahili participants, and 70 first-language Twi participants, each active within 90 days with no other filter. Swahili pools overlap and cannot be added. These mutable self-reports identify a possible paid recruitment route, not qualified or reserved participants. No direct Yoruba option was found; that does not establish zero Yoruba speakers.

After funding, ethics determination and consent, paid qualification will assess language variety, reading/writing and source interpretation. The target is 12 raters per language, with at least six qualified independent raters and separate authorship/adjudication roles before test freezing. Authors will not rate their own scenarios. If Twi qualification or costs fail, scope will be reduced before locked outcomes, retaining Swahili and its English reference.

The planned comparison has 120 scenario families, 40 development and 80 locked, with four evidence/wording variants, four configurations and two policies: 3,840 responses including development and 2,560 locked responses. A 640-response primary audit would be rated twice. The strong baseline already filters ordinary date/location metadata; additional source qualification must justify proposition-level checks beyond it. Useful-answer coverage, unsupported propositions, disagreement, family/source grouping and repeated raters will be reported together. This design and all human assessment remain prospective.

## Sources and licenses

- [AfriQA](https://aclanthology.org/2023.findings-emnlp.997/) supplies human translations and English answer spans verified at their original offsets. This pilot measures question comprehension and English extraction, not natural native-language answer quality.
- [Afri-MCQA](https://aclanthology.org/2026.acl-long.1869/) supplies Yoruba cultural image questions. Its official test labels are hidden; the experiment uses a frozen image-group research holdout from public development records. The snapshot contains no Swahili cultural items.
- [Prolific filters](https://researcher-help.prolific.com/en/articles/445159-how-do-i-use-prolific-s-filters), [paid custom screening](https://researcher-help.prolific.com/en/articles/445155-how-to-use-custom-screening-to-recruit-specific-participants) and [country coverage](https://researcher-help.prolific.com/en/articles/445224-who-are-the-participants-on-prolific) inform the future recruitment plan. Panel availability does not establish local representativeness or community endorsement.

`ATTRIBUTION.txt` records upstream terms and differing AfriQA license statements. Afri-MCQA declares CC-BY-NC-4.0. Upstream content keeps its creators' terms. The repository publishes source identifiers, short reference labels and derived observations, excluding the image corpus, native questions, complete passages/options, model weights and full image-bearing request payloads. The original public pilot files and result hashes remain unchanged.

## Manuscript versions

The root `paper.pdf` and `overleaf.zip` are the November 1 revision. The ZIP contains `draft.tex`, `refs.bib`, an identical `references.bib` compatibility copy, both figures and upload instructions. Select `draft.tex` in Overleaf. No loose TeX or bibliography files are tracked. The [October 31 snapshot](archive/20261031/) preserves the prior manuscript and metadata. Preprints.org submission/version management is separate from these repository files.

Subject guidance remains **Computer Science and Mathematics**, topic **Artificial Intelligence and Machine Learning**. The optional MDPI Topic candidate in `preprints_metadata.json` is historical classification guidance with source URLs; it is not a record of collection selection. Code, execution, analysis and manuscript preparation used Codex assistance under the author's direction.
