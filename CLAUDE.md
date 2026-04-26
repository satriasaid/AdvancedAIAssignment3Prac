# Workflow for Building the Task 3 Notebook: Hybrid CNN-Transformer Architecture with ResNet Backbone and SHAP Explainability for Multi-Class 12-Lead ECG Arrhythmia Classification

This document is a step-by-step build plan for an AI coding agent to create an `.ipynb` notebook and supporting outputs for the COMP6011 Task 3 assignment. The workflow is designed to satisfy the assignment brief and maximize marks against the marking rubric by covering the required report sections, benchmarking, methodology, explainability, AI ethics, and supporting materials [file:1][file:2]. The proposed model direction is also aligned with recent ECG literature showing strong performance for hybrid CNN-Transformer architectures with gated fusion and explainability support [page:1][web:38][web:42].

## Goal

Build a reproducible notebook that does four things well: prepares ECG data, benchmarks several candidate models, develops a final hybrid model, and produces explainable diagnostic outputs suitable for a clinician-facing report [file:1][file:2]. The notebook must also generate evidence and figures that can be reused directly in the final report, including benchmarking tables, confusion matrices, ablation results, architecture diagrams, and explanation examples [file:1][file:2].

## What the notebook must answer

The notebook should be written so that its outputs directly support these assignment questions:

- Which models were reviewed and benchmarked, and why were they selected [file:2].
- Which datasets were used, with at least two suitable ECG datasets described thoroughly [file:2].
- Which evaluation metrics were used and why they are appropriate for medical classification, including accuracy, sensitivity/recall, specificity, precision, F1-score, AUROC if possible, and confidence calibration [file:1][file:2].
- Why the final hybrid CNN-Transformer with ResNet-style backbone is the most suitable candidate for this application [page:1][file:2].
- How explainability is implemented so clinicians can trust the predictions, including SHAP and class-probability/confidence outputs [file:1][page:1].
- How low-confidence cases are flagged for manual review [file:1].
- How the solution addresses fairness, reliability, privacy, deployment practicality, UN-SDG alignment, and carbon footprint estimation [file:1][file:2].

## Recommended notebook structure

Create the notebook with the following top-level sections. Each section should map to a rubric item so the notebook becomes evidence for the final report and appendix materials [file:2].

1. Title, objectives, and experiment plan.
2. Environment setup and reproducibility controls.
3. Dataset acquisition and dataset cards.
4. Data preprocessing and label mapping.
5. Exploratory analysis of ECG signals.
6. Baseline models.
7. Advanced candidate models.
8. Benchmarking framework.
9. Final model: hybrid CNN-Transformer with ResNet backbone.
10. Training strategy and hyperparameter tuning.
11. Explainability and uncertainty handling.
12. External validation and test predictions.
13. Error analysis and ablation study.
14. Ethics, privacy, deployment, SDGs, and carbon footprint.
15. Export of figures, tables, predictions, logs, and appendix artifacts.

## Notebook outputs to generate

The coding agent should save all major outputs to an `output/` folder so they can be inserted into the report or cloud folder as compulsory supporting materials [file:2].

- `benchmark_results.csv`
- `dataset_summary.csv`
- `class_distribution.csv`
- `hyperparameter_log.csv`
- `ablation_results.csv`
- `external_validation_results.csv`
- `test_predictions.csv`
- `low_confidence_cases.csv`
- `model_card.md`
- `ethics_checklist.md`
- `carbon_footprint_estimate.md`
- PNG figures for sample ECGs, preprocessing pipeline, class distribution, training curves, ROC curves, PR curves, confusion matrices, SHAP examples, and architecture diagram [file:1][file:2]

## Step-by-step workflow

## Phase 1: Define scope exactly as the brief requires

The coding agent must lock the task definition before writing model code. The target classes in the assignment are `NORM`, `AFIB`, `AFLT`, `1dAVb`, `RBBB`, `LBBB`, and `OTHERS`, and only the first predicted disease is used for accuracy assessment if multiple diseases are returned [file:1]. The notebook should explicitly document this label policy, the class definitions, and the intended use in a clinical support setting rather than autonomous diagnosis [file:1].

Create a markdown cell titled `Problem Definition` that states the clinical goal, target label space, and decision-support constraint. Then create a markdown cell titled `Research Objectives` with 4-6 measurable objectives, for example: achieve strong macro-F1 across all target classes, compare at least four model families, implement explainability, and design a low-confidence referral mechanism [file:2].

## Phase 2: Reproducibility and setup

Start with a notebook section that fixes random seeds, logs package versions, records hardware, and stores experiment configuration as a dictionary or YAML-style object. This helps the methodology section and provides professional evidence of sound experimentation practice [file:2].

Include:

- Python version, PyTorch/TensorFlow version.
- GPU/CPU details.
- Random seed.
- File paths.
- Sampling rate assumptions.
- Window length and segmentation rules.
- Confidence threshold rules.

Export this config to `output/run_config.json`.

## Phase 3: Select datasets strategically

The rubric requires a minimum of two datasets for benchmarking, and the assignment specifically involves ECG-based disease diagnosis [file:1][file:2]. The strongest strategy is to use one dataset for broad benchmarking and another for external validation or stress-testing generalization.

Recommended dataset plan:

| Use | Dataset | Why it helps scoring |
|---|---|---|
| Main benchmarking | PhysioNet/CinC 2020 12-lead ECG dataset or PTB-XL | Closer to the assignment’s 12-lead clinical setting and supports multiple diagnostic labels relevant to conduction abnormalities and arrhythmias [file:1]. |
| Additional benchmarking / external validation | Chapman-Shaoxing, CPSC, or a second PhysioNet-compatible 12-lead dataset | Satisfies the “minimum 2 datasets” requirement and strengthens generalization claims [file:2]. |
| Optional heartbeat-level sanity benchmark | MIT-BIH | Useful for fast prototyping and literature comparison, but less aligned with the final 12-lead task than PTB-XL/CinC-style datasets [page:1][file:1]. |

For each dataset, create a dataset card section containing:

- Source and citation.
- Number of records.
- Lead count.
- Sampling frequency.
- Label ontology.
- Demographic notes if available.
- Strengths, weaknesses, and possible bias issues.

Then export `dataset_summary.csv`.

## Phase 4: Build a rigorous preprocessing pipeline

The preprocessing pipeline should be explicit because the methodology marks reward discussion of training and fine-tuning data considerations [file:2]. The coding agent should implement preprocessing as modular functions, each tested in the notebook.

Required preprocessing stages:

- Signal loading and lead selection, preserving all 12 leads when available [file:1].
- Resampling to a common frequency if datasets differ.
- Denoising or baseline wander removal if justified.
- Normalization, ideally per-record or per-lead z-score normalization.
- Segmentation or fixed-window extraction around rhythm episodes depending on the dataset format.
- Label mapping from original dataset diagnoses to the assignment labels `NORM`, `AFIB`, `AFLT`, `1dAVb`, `RBBB`, `LBBB`, `OTHERS` [file:1].
- Train/validation/test splitting by patient, not by segment, to avoid leakage.
- Class imbalance handling through weighted loss, focal loss, or augmentation.

Important: the notebook should include a dedicated section proving there is no patient leakage. This is an easy place to gain professionalism and credibility marks [file:2].

Generate figures showing raw vs processed ECGs and class distributions before and after filtering. Save these for the report.

## Phase 5: EDA that supports the literature and methodology sections

Add exploratory analysis that helps justify model design choices and makes the report look evidence-based rather than purely implementation-driven [file:2].

Include:

- Example ECGs for each target class if data permits.
- Class imbalance plots.
- Signal length distribution.
- Missing lead analysis.
- Correlation or variability summaries between leads.
- Examples of noisy/ambiguous recordings to motivate low-confidence handling.

This section should end with a short markdown cell titled `Design Implications` that connects EDA findings to model choices, such as why local waveform extraction plus long-range context modeling is needed.

## Phase 6: Benchmark simple and strong baselines first

To score highly in benchmarking, the notebook should not jump directly to the final fancy model. It should benchmark a sensible progression of candidate methods and justify each one [file:2].

Recommended benchmark set:

| Tier | Model | Why include it |
|---|---|---|
| Baseline 1 | Logistic Regression or XGBoost on handcrafted features | Shows classical ML baseline and supports literature comparison. |
| Baseline 2 | 1D CNN | Strong ECG baseline for local morphology. |
| Baseline 3 | ResNet1D | Strong residual baseline and supports the “ResNet backbone” theme. |
| Baseline 4 | CNN-LSTM | Captures temporal dependencies; useful comparison against transformer models. |
| Candidate 1 | Transformer-only or ViT-style 1D transformer | Tests whether global attention alone is enough. |
| Candidate 2 | Hybrid CNN-Transformer | Direct predecessor to the final model. |
| Final | Hybrid CNN-Transformer with ResNet backbone + SHAP + confidence flagging | Final proposed solution tied to title and rubric requirements. |

For each model, the agent should:

- Define architecture succinctly.
- Train under a consistent evaluation protocol.
- Record training time, inference time, memory use if possible.
- Save metrics to one common benchmark table.

This structure directly supports the rubric item requiring all benchmarking candidates to be listed and justified [file:2].

## Phase 7: Final model design

The final notebook model should match the title and be defensible from the literature. A strong implementation pattern is:

- **Input:** 12-lead ECG windows or recordings.
- **Backbone:** ResNet-style 1D convolutional blocks to learn local morphology and stable residual representations [page:1].
- **Context branch:** Transformer encoder layers for long-range inter-beat or inter-time-step dependencies [page:1][web:38].
- **Fusion:** concatenation plus gated fusion or attention fusion.
- **Classifier head:** fully connected layers producing the 7 assignment classes.
- **Confidence head:** softmax confidence, entropy score, or temperature-scaled confidence for referral decisions [file:1].
- **Explainability module:** SHAP on selected cases; optionally Grad-CAM-like saliency adapted for 1D signals.

Minimum implementation expectation:

1. ResNet1D feature extractor.
2. Transformer encoder on feature tokens or sequence patches.
3. Fusion layer.
4. Classification head.
5. Probability calibration step.
6. Low-confidence flagging function.

The notebook should also include an architecture figure or code-generated block diagram saved as PNG for the report [file:2].

## Phase 8: Training strategy

The training pipeline should be strong enough to support the Methodology and Benchmarking criteria [file:2]. The coding agent should implement:

- Early stopping.
- Class-weighted cross-entropy or focal loss.
- Stratified patient-level splitting where possible.
- Learning-rate scheduling.
- Optional augmentation such as random noise, scaling, time masking, or mixup if medically defensible.
- Hyperparameter search over learning rate, batch size, dropout, transformer depth, number of heads, and confidence threshold.

Store all experiments in `hyperparameter_log.csv`. The notebook should create a compact summary table of best configurations and why they were selected.

## Phase 9: Evaluation metrics that match the rubric and the medical setting

The rubric explicitly expects clear explanation of evaluation metrics [file:2]. The notebook must compute more than accuracy.

Required metrics:

- Accuracy.
- Precision.
- Recall/sensitivity.
- Specificity.
- F1-score, preferably macro and weighted.
- AUROC per class if feasible.
- AUPRC per class if feasible.
- Calibration error or reliability plot if possible.
- Confusion matrix.
- Inference time per record or batch.

Because this is medical diagnosis, the notebook should include a markdown note explaining that recall/sensitivity and false negatives matter clinically, while confidence calibration matters for safe human-in-the-loop deployment [file:1][file:2].

## Phase 10: Qualitative benchmarking

The brief explicitly asks for qualitative as well as quantitative benchmarking, including interpretability, transparency, and clinical plausibility [file:1]. The notebook should therefore create a qualitative comparison table across models.

Suggested qualitative comparison dimensions:

- Interpretability.
- Computational cost.
- Deployment complexity.
- Robustness to noisy ECGs.
- Suitability for 12-lead input.
- Clinical plausibility.
- Ease of confidence estimation.

This table is an easy way to score marks in the benchmarking section because it makes the comparison visibly complete [file:2].

## Phase 11: Explainability implementation

Explainability is essential in the brief, so this cannot be an afterthought [file:1]. The notebook should include a dedicated section titled `Explainable AI for Clinician Trust`.

Minimum explainability workflow:

1. Select representative cases from each class.
2. Use SHAP on the final model or a proxy explainer suitable for deep time-series models.
3. Visualize which lead segments contributed most to the prediction.
4. Compare a correct case and an incorrect or low-confidence case.
5. Add plain-language interpretation notes for each example.

The notebook should save:

- SHAP summary plot.
- SHAP per-class example plots.
- Saliency map examples if implemented.
- A table linking prediction, probability, true label, confidence status, and explanation artifact path.

This section should explicitly show how explanations build clinician trust and where explanations remain limited, because discussing limitations often earns stronger marks than pretending the method is perfect [file:1][file:2].

## Phase 12: Low-confidence detection and manual review policy

Your brief explicitly says the AI solution must detect and flag low-confidence cases for manual review [file:1]. The notebook therefore needs a formal uncertainty policy.

Implement one or more of these:

- Maximum softmax probability threshold.
- Predictive entropy threshold.
- Monte Carlo dropout uncertainty.
- Temperature scaling plus calibrated confidence.

Then create:

- `low_confidence_cases.csv`
- A histogram of confidence scores.
- A threshold-vs-coverage curve.
- A markdown explanation of what happens operationally when a case is flagged.

This is a high-value feature because many students will mention it in prose but not implement it.

## Phase 13: External validation and assignment test-set inference

The brief states that a separate validation set is provided to support tuning and that the final model will be applied to test ECG samples from Blackboard [file:1]. The notebook should therefore be structured to switch cleanly between training, validation, and final inference.

Required inference pipeline:

- Load final saved model.
- Load provided validation/test data.
- Apply identical preprocessing.
- Produce predicted class, top probability, and low-confidence flag.
- Save outputs to `test_predictions.csv`.
- If multiple diseases are generated internally, ensure the notebook outputs the top-1 prediction for accuracy assessment, exactly as the brief states [file:1].

Also produce a concise table in the notebook showing the final submission format expected for the report.

## Phase 14: Ablation study

To strengthen both the literature and methodology sections, run an ablation study. This gives concrete evidence for why the final architecture was selected [file:2][page:1].

Recommended ablations:

- ResNet backbone only.
- Transformer branch only.
- CNN + Transformer without gated fusion.
- Full model without SHAP, noting that SHAP affects interpretability not predictive performance.
- Full model with and without confidence flagging/calibration.

Save the results in `ablation_results.csv` and create a summary figure. This is particularly useful when justifying the final model in the report.

## Phase 15: Error analysis

A high-scoring notebook should have an `Error Analysis` section. This is where the agent inspects misclassifications between similar rhythm/conduction classes and determines whether errors are due to noise, class overlap, preprocessing issues, or confidence thresholding.

Include:

- Top confusions from the confusion matrix.
- Example false positives and false negatives.
- Whether errors cluster by dataset, class, or signal quality.
- Whether `OTHERS` is acting as a clinically safe fallback or an overused dumping class.

This analysis will help you write a much stronger discussion and conclusion section later [file:2].

## Phase 16: AI ethics, privacy, and fairness section inside the notebook

The rubric requires specific discussion of Australia’s 8 AI Ethics Principles, not generic comments [file:2]. The notebook should therefore generate evidence and notes that map the model design to those principles.

Create a markdown section or exported `ethics_checklist.md` covering at least:

- Human, social, and environmental wellbeing: earlier arrhythmia support and workflow efficiency [file:1][file:2].
- Human-centred values: decision support, not replacement of cardiologists [file:1].
- Fairness: evaluate class imbalance and, if metadata exists, subgroup performance.
- Privacy protection and security: de-identification, secure storage, restricted data sharing.
- Reliability and safety: calibration, low-confidence referral, external validation.
- Transparency and explainability: SHAP, confidence outputs, documentation.
- Contestability: clinician override and review pathway.
- Accountability: model cards, versioning, audit logs.

Even if subgroup metadata is limited, the notebook should explicitly note that limitation rather than skip fairness.

## Phase 17: Carbon footprint and deployment practicality

The rubric awards marks for carbon footprint estimation and implementation/deployment discussion [file:2]. The notebook should therefore include a short but real estimate.

Actions:

- Log approximate training time, hardware type, and number of runs.
- Use an established estimator such as ML CO2 Impact or a documented approximation method [file:2].
- Save the estimate to `carbon_footprint_estimate.md`.
- Discuss deployment scenarios: hospital server, edge device, cloud inference, and trade-offs in latency, security, and maintenance.

Also include memory footprint and inference latency if available, because practical deployment awareness is part of the methodology marks [file:2].

## Phase 18: UN-SDG alignment

The rubric expects alignment with relevant UN Sustainable Development Goals [file:2]. The notebook or exported markdown should justify this specifically.

Best choices:

- **SDG 3: Good Health and Well-Being** because the model supports earlier and more consistent cardiac diagnosis.
- **SDG 9: Industry, Innovation and Infrastructure** because it contributes to digital health infrastructure.
- Optionally **SDG 10: Reduced Inequalities** if the discussion is about extending specialist-level support to under-resourced settings.

Keep this specific to ECG diagnosis workflows rather than generic AI-for-good language.

## Phase 19: Report-ready artifacts

The notebook should finish by exporting all report-ready artifacts. This is important because the rubric gives marks for additional materials demonstrating learning and for professional presentation [file:2].

At minimum export:

- Final benchmark table as CSV.
- Final confusion matrix PNG.
- Training curve PNG.
- ROC/PR figures.
- SHAP examples.
- Architecture diagram.
- Test prediction file.
- Low-confidence review file.
- Model card.
- Ethics checklist.
- Carbon estimate.
- Brief experiment log.

Also save the final trained model checkpoint and preprocessing objects if permitted.

## Suggested notebook section headings

Use these exact or near-exact headings inside the notebook so they map cleanly into your report:

- Problem Definition
- Research Objectives
- Dataset Description
- Preprocessing Pipeline
- Exploratory Data Analysis
- Candidate Models
- Benchmarking Protocol
- Proposed Hybrid Model
- Training and Tuning
- Evaluation Metrics
- Benchmarking Results
- Explainability Analysis
- Low-Confidence Review Mechanism
- Ablation and Error Analysis
- AI Ethics and Clinical Deployment
- Carbon Footprint and SDG Alignment
- Final Test Predictions

## Recommended cell-by-cell implementation order

1. Imports and reproducibility setup.
2. Config object and paths.
3. Dataset loaders.
4. Label mapping functions.
5. Preprocessing functions.
6. EDA plots and summaries.
7. Baseline feature extraction and classical ML baseline.
8. 1D CNN baseline.
9. ResNet1D baseline.
10. CNN-LSTM baseline.
11. Transformer baseline.
12. Hybrid CNN-Transformer implementation.
13. ResNet-backed hybrid final model.
14. Training loops and validation logging.
15. Benchmark table creation.
16. Calibration and confidence-threshold module.
17. SHAP and explanation plots.
18. Ablation runs.
19. Error analysis utilities.
20. Final inference on provided validation/test files.
21. Export artifacts.

## Success criteria for the coding agent

The notebook is only finished when all of the following are true:

- At least two ECG datasets are used and documented thoroughly [file:2].
- At least four candidate models plus the final model are benchmarked [file:2].
- Results are presented quantitatively and qualitatively [file:1][file:2].
- The final model is a hybrid CNN-Transformer with a ResNet-style backbone and explicit explainability [page:1][file:1].
- A low-confidence referral mechanism is implemented, not just described [file:1].
- Outputs are exported in a clean format for the report and appendices [file:1][file:2].
- Ethics, privacy, SDGs, and carbon footprint are all addressed with assignment-specific discussion [file:2].
- The notebook is reproducible and professionally structured [file:2].

## Practical advice for maximizing marks

The best-scoring approach is not just to build the most complex model. The stronger strategy is to build a notebook that makes the report easy to write and easy to mark: every section should visibly answer one rubric criterion, every claim should be backed by an experiment or figure, and every design choice should have a short justification tied either to the brief or to recent ECG literature [file:1][file:2][page:1]. If time becomes limited, prioritize patient-safe data splitting, strong benchmarking tables, the low-confidence flagging pipeline, and explainability examples, because these are high-impact differentiators for this assignment [file:1][file:2].
