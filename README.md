# System Tradeoffs

This repository studies system-level tradeoffs in machine learning under real-world constraints.

We do not treat privacy, fairness, or stability as afterthoughts.  
We treat them as design variables that directly interact with model accuracy and system behavior.

This repository serves as the foundation for multiple papers and submissions, including:
- IEEE SoutheastCon, which is the annual IEEE Region 3 Technical, Professional, and Student Conference. ( https://southeastcon.ieee.org/)
- Additional workshop submissions (in preparation)
  
Each paper uses a subset of the full experimental grid defined here.

## Core Question

When deploying constraints into mission critical systems for AI in Education or AI in Healthcare, what changes?

More specifically:

- What happens to accuracy when privacy is enforced?
- What happens to subgroup fairness under decentralization?
- What happens to stability across rounds?
- What tradeoffs emerge when these constraints interact?

This repository implements controlled experiments to answer those questions.  

While grounded in AI in Education and AI in Healthcare, the insights generalize to other high-stakes domains such as FinTech and Law.

---

## System Overview

We compare the following configurations:

- **Centralized training**
- **Federated learning (FedAvg)**
- **Federated learning with Differential Privacy (DP-SGD)**

Across these settings, we vary:

- Privacy strength (ε, clipping norm, noise scale)
- Participation rate
- Number of rounds
- Data heterogeneity

We measure:

- Accuracy
- Subgroup fairness
- Privacy risk (e.g., membership inference)
- Stability across runs
- Communication overhead

---



Rather than treating privacy or fairness as afterthought evaluations, this work frames them as system properties that emerge from the interaction between:

- Task
- Data distribution
- Training configuration
- Constraint mechanism
- Evaluation protocol

This repository serves as a unified codebase supporting multiple research outputs, including conference and workshop submissions.


What you’ll find in this README
- Quick summary & how to reproduce
- Design & implementation notes (GCG attack and prompt-injection harness)
- Metrics & results (what we measured, outcomes)
- Why the local GCG attempt stopped working locally (what I observed + interpretation)
- Defenses implemented
- Limitations, open challenges, and next steps
- File layout and where to find plots / outputs

---

## Quick start — reproduce the notebook
Requirements (tested in Colab / Linux workstation):
- Python 3.8+
- pip packages (install in this order):
  - pip install --upgrade pip
  - pip install sentencepiece accelerate bitsandbytes transformers torch pillow pandas openai groq
- (Optional) Hugging Face authentication to load gated models:
  - `huggingface-cli login` (or set `HUGGINGFACE_TOKEN` / use local model path)

Run:
1. Open `CS690F_assignment6.ipynb` in Jupyter / Colab.
2. Set environment variables or provide API keys in the notebook (the notebook requests OPENAI/GROQ keys).
3. Execute the notebook cells in order.
   - Important: the notebook contains a local GCG implementation that requires a model you can run locally and gradient access.

Notes:
- The notebook shows both a GCG attempt (local gradient attack) and a prompt-injection experiment (using the Groq/OpenAI client harness).
- If you want to re-run the GCG attack locally, you must use an accessible model (see "Why the GCG stopped working" below).

---

## Design & Implementation



## Metrics & Results (summary)


---



---



---



---

## File layout (important files)
- CS690F_assignment6.ipynb — main notebook with attack, injection sweep, defenses and visualizations
- examples/
  - sample_injection_variants/ — generated snippet variants + manifest
- results_startup/
  - run_*.json, (plots/) — startup harness results
- results_variant_sweep/
  - variant_sweep_summary.csv
  - variant_sweep_rows.json
  - examples/*.png (per-variant example images)
- README.md — this file

---

## Where to find plots and artifacts
- Variant sweep summary: [`results_variant_sweep/variant_sweep_summary.csv`](./results_variant_sweep/variant_sweep_summary.csv)
- Per-variant examples (text snapshots): [`results_variant_sweep/examples/variant_13_prov_trusted.png`](./results_variant_sweep/examples/variant_13_prov_trusted.png)
- Per-variant examples (text snapshots): [`results_variant_sweep/examples/variant_12_embedded_paragraph.png`](./results_variant_sweep/examples/variant_12_embedded_paragraph.png)
- Per-variant examples (text snapshots): [`results_variant_sweep/examples/variant_16_malicious_overt.png`](./results_variant_sweep/examples/variant_16_malicious_overt.png)
- Startup experiment full JSON: [`results_startup/run_20251027_013803.json`](./results_startup/run_20251027_013803.json)
- Defense Comparison plot: [`results_startup/plots/defense_comparison.png`](./results_startup/plots/defense_comparison.png)
- Variant Success Counts plot: [`results_startup/plots/variant_success_counts.png`](./results_startup/plots/variant_success_counts.png)
- Variant Success Rates plot: [`results_startup/plots/variant_success_rates.png`](./results_startup/plots/variant_success_rates.png)

Open the notebook's "Analyze and Visualize Results" cells to render plots inline or view the PNG files listed above.

---

## How we used AI

We used ChatGPT for README structure and wording. All technical settings and results were produced by our code/notebook.
