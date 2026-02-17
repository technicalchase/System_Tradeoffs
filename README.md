# System Tradeoffs

This repository studies system-level tradeoffs in machine learning under real-world constraints.

We do not treat privacy, fairness, or stability as afterthoughts.  
We treat them as design variables that directly interact with model accuracy and system behavior.

This repository serves as the foundation for multiple papers and submissions, including:
- IEEE SoutheastCon, the annual IEEE Region 3 Technical, Professional, and Student Conference. (https://southeastcon.ieee.org/)
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

While grounded in AI in Education and AI in Healthcare, the insights generalize to other mission critical domains such as FinTech and Law.

---

## Experimental Framework

We study tradeoffs across:

### System Configurations

We compare the following configurations:

- **Centralized training**
- **Federated learning (FedAvg)**
- **Federated learning with Differential Privacy (DP-SGD)**

Across these settings, we vary:

- Privacy strength (ε, clipping norm, noise scale)
- Participation rate
- Number of rounds
- Data heterogeneity

## Evaluation Dimensions

We measure:

- **Accuracy**:  Overall and per-class performance.
- **Fairness**:  Subgroup and institutional disparity across configurations.
- **Stability**:  Variance across rounds and repeated runs.
- **Privacy risk**:   Membership inference evaluation under different constraint levels.
- **Communication overhead**:  Cost of coordination in federated settings.

Rather than optimizing a single objective, we evaluate how these dimensions shift as constraints such as privacy and decentralization are introduced.

---
## Datasets

This repository uses two primary datasets across different stages of the project.

- Synthetic gaze dataset (course prototype)  
  Used in early experiments to study controlled privacy and membership inference behavior.  
  Location: [`Dataset/`](./Dataset/)  

- SIIM COVID-19 Detection dataset (Kaggle)  
  Used in later experiments and conference submissions to evaluate system tradeoffs at realistic scale.  
  Source: https://www.kaggle.com/competitions/siim-covid19-detection  
  Note: Due to licensing, this dataset is not included in the repository.  
  Users must download it directly from Kaggle and follow the preprocessing steps described in the notebook.

Each paper or submission uses a defined subset of configurations and datasets from this repository.

---

## Quick Start

### Requirements

- Python 3.9+
- PyTorch
- torchvision
- numpy
- pandas
- scikit-learn
- matplotlib

### Run Experiments

1. Clone the repository
2. Install dependencies
3. Open the main notebook or training script
4. Execute cells or run scripts as documented in the experiment section

Results include:
- Accuracy metrics
- Fairness metrics
- Stability curves
- Privacy evaluation results

## Core implementation files

- Differential privacy utilities: [`dp.py`](./dp.py)  
  Implements clipping, noise injection, and privacy accounting logic used in DP-SGD experiments.

- Federated training + DP integration: [`fed_avg_dp.py`](./fed_avg_dp.py)  
  Implements centralized, federated, and federated + differential privacy training loops.  
  Handles client sampling, aggregation, and constraint application.

- Multi-configuration experiment runner: [`run_all_modes.py`](./run_all_modes.py)  
  Executes the full experimental grid across system configurations and privacy strengths.  
  Logs accuracy, fairness, stability, and privacy metrics.

- Shell entry point (optional local runs): [`run.sh`](./run.sh)  
  Provides a simple command-line wrapper to launch experiments reproducibly.

## Experiment Outputs

Running experiments generates structured logs, metrics, and plots.

- Training logs (CSV): [`reports/`](./reports/)  
  Contains per-round accuracy, loss, and fairness metrics.

- Model checkpoints: [`checkpoints/`](./checkpoints/)  
  Saved model states for centralized and federated runs.

- Privacy evaluation logs: [`reports/privacy/`](./reports/privacy/)  
  Membership inference metrics and attack summaries.

- Fairness summaries: [`reports/fairness/`](./reports/fairness/)  
  Institutional accuracy breakdown and fairness gap calculations.

- Stability tracking: [`reports/stability/`](./reports/stability/)  
  Variance across rounds and repeated runs.

Plots are saved as PNG files within their respective directories.  
Open the associated notebook to render visualizations inline.



---
## Authors
Andre Kennth Chase Randall, Team Lead  
PhD Student, Computer Science  
Research focus: AI in Education

Ruzan Almutairi,
Masters Student, Computer Science

Kiet Chu,
Masters Student, Computer Science

Neeladri Bhuiya,
Master Student, Computer Science

## Use of AI Tools

ChatGPT assisted with documentation drafting.  
All experiments and results were generated by the authors’ implementation.

