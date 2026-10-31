# Multilingual grounding pilot

This repository reports two small external-label evaluation pilots that separate question language, supplied English evidence, and cultural image use. Two local configurations completed 234 requests over 24 locked AfriQA questions, 12 locked Yoruba cultural image groups, and 3 interface-only units. The artifact accompanies *Source Use and Cultural Image Questions: Small External-Label Pilots in Yoruba and Swahili* by Saurav K. Aryal, Howard University.

Repository: https://github.com/Saurav-K-Aryal/multilingual-grounding-pilot

## Recorded results

Each table cell has 12 paired items. AfriQA always requires a verbatim **English passage span**, including for native-language questions.

| Model tag | Yoruba native / English question EM | Swahili native / English question EM | Empty passage: correct null |
| --- | --- | --- | --- |
| qwen3.5:4b |3/12 /5/12|7/12 /6/12|24/24 |
| gemma4:26b |7/12 /6/12|6/12 /6/12|24/24 |

| Model tag | Yoruba image/native text | Image/English text | No image/native text |
| --- | ---: | ---: | ---: |
| qwen3.5:4b |5/12|8/12|2/12|
| gemma4:26b |7/12|8/12|3/12|

All 234 responses have valid schemas and no transport errors. The 18 interface checks are excluded from the 216 locked-response metrics. Conditions are paired within 36 locked units, not 216 independent questions. Question-language effects in AfriQA are mixed; the cultural image/no-image net differences are +3 and +4 correct items. No new community validation, natural native-answer quality, broad cultural ranking, or source-freshness result is claimed.

## Verify recorded outputs and redraw figures

The offline verifier uses only the Python standard library. It recomputes all 234 output schemas and published short-reference labels, exact-match/F1 scores and locked aggregates, and verifies public-file hashes:

```sh
python3 verify_results.py
```

Figure generation needs Pillow:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python generate_figures.py
```

`data/model_results.json` contains recorded counts and original result hashes. `results/observed/` includes projected raw responses, decisions, aggregate results, and local artifact metadata. Request projections preserve input text/image hashes without distributing those input contents. `PUBLICATION_PROVENANCE.json` records original and public hashes and every projection/path-only transformation. The original full request log was independently checked before publication, including all 52 image payloads and official label mappings.

## Recreate the input study and optionally rerun inference

This path downloads upstream inputs locally; it does not mirror source corpora or image bytes in GitHub. It uses pinned AfriQA Git revision 5c29933753692c28773c34ac1f68482007f1d2bb and checks Afri-MCQA dataset-server revision d5d4db17fc2703d761799746d258e2dbb6bc4b73. The latter is a public source revision whose cached image views may move. Preparation stops if that revision, selected case content, or processed image bytes differs, instead of silently changing the study. Upstream availability is required; original source identities are in `protocol/`.

```sh
python prepare_data.py --out results/reproduced-inputs
```

This fetches public text/metadata and the 13 selected images, freezes 117 identical cases, and verifies their identities. No model calls occur during preparation. For inference, use a compatible native local chat server at `127.0.0.1:11434` with the already downloaded model tags. The observed Qwen artifact uses NVFP4/MLX; Gemma uses Q4_K_M/llama.cpp. Weight digests are in each model's `model_provenance.json`. Other weights/backends are new configurations and should be identified as such.

```sh
python -m pilots.strengthening.multilingual_external_qa --stage execute --out results/reproduced-inputs --model qwen3.5:4b
# After this process finishes and the first model is unloaded:
python -m pilots.strengthening.multilingual_external_qa --stage execute --out results/reproduced-inputs --model gemma4:26b
```

Run the models sequentially. Each executes 117 requests once, with no retries or answer repair. Temperature 0, seed 20261031, requested context 6144, output cap 128, JSON formatting and thinking-disabled settings are frozen. Existing model run directories are rejected to preserve observations. The code calls only the loopback model endpoint. This is not a 48 GB peak-memory or runtime benchmark.

## Sources and scope

- AfriQA: https://aclanthology.org/2023.findings-emnlp.997/ and https://github.com/masakhane-io/afriqa
- Afri-MCQA: https://aclanthology.org/2026.acl-long.1869/ and https://huggingface.co/datasets/Atnafu/Afri-MCQA

AfriQA uses published human translations and gold English spans verified at their original offsets. It measures cross-lingual question comprehension and English-source extraction, not the quality of native-language answers. Empty-passage abstention is an explicit authored control, not human validation of every unsupported-answer situation.

Afri-MCQA uses **Yoruba only**. Its official test labels are hidden, so these results use an internally frozen, image-group-disjoint research holdout of public **development** records. Correct/wrong options and native/English wording come from the external resource. The selected snapshot contains no Swahili cultural items. No new translations or LLM judges provide ground truth. Public-benchmark contamination in pretrained models is not excluded.

`ATTRIBUTION.txt` records upstream license/source distinctions. The official AfriQA GitHub repository declares CC-BY-4.0; its HuggingFace metadata/card contain differing license statements, which are recorded rather than silently resolved. Afri-MCQA declares CC-BY-NC-4.0. Upstream content retains its creators' terms. This repository includes identifiers, short reference labels and derived observations, while excluding the image corpus, native questions, complete passages/options, model weights and full image-bearing request payloads.

## Manuscript and Preprints metadata

Download `overleaf.zip`, upload it to Overleaf, and select `draft.tex`. The ZIP contains the single-column ACM source, `refs.bib`, an identical `references.bib` compatibility copy, two PNG figures and a short README. There are no loose TeX or bibliography files in the repository. The manuscript uses only `refs.bib`.

Suggested Preprints subject: **Computer Science and Mathematics**. Topic within subject: **Artificial Intelligence and Machine Learning**. The optional MDPI Topic candidate is **Challenges and Solutions in Large Language Models**, a secondary fit for LLM reliability/QA, not a dedicated African-language collection. Exact labels and source URLs are in `preprints_metadata.json`; no submission or collection selection has been performed.

Code, execution, analysis and manuscript preparation used Codex assistance under the author's direction. Future work in the paper follows the proposed community-authored Yoruba/Swahili evaluation, human audit, evidence-locality/freshness checks, and cost-aware reference assistant; those interventions remain prospective.
