# Assignment 2: MLOps & PCAM Pipeline Journal
**MLOps & ML Programming (2026)**

## Group Information
* **Group Number:** [Your Group Number]
* **Team Members:** [Member 1 Name/ID, Member 2 Name/ID, Member 3 Name/ID, Member 4 Name/ID, Member 5 Name/ID]
* **GitHub Repository:** [Link to your Group Repository]
* **Base Setup Chosen from Assignment 1:** [Name of the group member whose repo was used as the foundation]

---

## Question 1: Reproducibility Audit
1. **Sources of Non-Determinism:**

2. **Control Measures:**

3. **Code Snippets for Reproducibility:**
   ```python
   # Paste the exact code added for seeding and determinism
   ```

4. **Twin Run Results:**

---

## Question 2: Data, Partitioning, and Leakage Audit
1. **Partitioning Strategy:**

2. **Leakage Prevention:**
   
3. **Cross-Validation Reflection:**

4. **The Dataset Size Mystery:**

5. **Poisoning Analysis:**

---

## Question 3: Configuration Management
1. **Centralized Parameters:**

2. **Loading Mechanism:**
   - [Describe your use of YAML, Hydra, or Argparse.]
   ```python
   # Snippet showing how parameters are loaded
   ```

3. **Impact Analysis:**

4. **Remaining Risks:** 

---

## Question 4: Gradients & LR Scheduler
1. **Internal Dynamics:**

2. **Learning Rate Scheduling:**

---

## Question 5: Part 1 - Experiment Tracking
1. **Metrics Choice:**
F1-score: Balans tussen precision en recall; belangrijk omdat zowel false positives als false negatives relevant zijn bij kanker detectie.

F2-score: Legt meer nadruk op recall (vermijden van false negatives). Kritisch in klinische context waar het missen van tumorweefsel zwaar weegt.

ROC-AUC: Meet hoe goed het model positieve en negatieve klassen scheidt over alle mogelijke thresholds.

PR-AUC: Vooral nuttig bij onevenwichtige datasets; geeft inzicht in de trade-off tussen precision en recall.
2. **Results (Average of 3 Seeds):**
Metric	Value
F1-score	0.5248
F2-score	0.4565
ROC-AUC	0.7303
PR-AUC	0.6889

ROC-AUC van 0.73 laat zien dat het model redelijk onderscheid kan maken tussen kanker en niet-kanker patches.

F2 < F1 → model mist relatief meer positieve gevallen, wat in lijn is met het klinische belang van recall.
3. **Logging Scalability:**
Ad-hoc logging via print statements of losse bestanden schaalt niet:
- Moeilijk te vergelijken tussen runs.
- Handmatig bijhouden van hyperparameters en metrics is foutgevoelig.
- Checkpoints en experimentconfiguraties raken snel verspreid.

MLflow centraliseert alle logs, checkpoints, metrics en config-bestanden, waardoor experimenten reproduceerbaar en audit-proof worden.
4. **Tracker Initialization:**
   ```python
   # Snippet showing tracker/MLFlow/W&B initialization
   from ml_core.tracking.mlflow_tracker import MLflowTracker

   config = load_config("experiments/configs/train_config.yaml")
   tracker = MLflowTracker(config)  # MLflow tracker voor experiment logging
   ```

5. **Evidence of Logging:**
   Train/Validation metrics gelogd per epoch in MLflow.
   Checkpoints van het model opgeslagen en als artifact gelogd.(In best_checkpoint.pt)
   Testset metrics (F1, F2, ROC-AUC, PR-AUC) gelogd na evaluatie.

6. **Reproduction & Checkpoint Usage:**
   Reproduceerbare stappen:

   Laad dezelfde config (train_config.yaml).

   Laad best_checkpoint.pt in hetzelfde model:

   "model.load_state_dict(torch.load("experiments/results/best_checkpoint.pt", map_location="cpu")["model_state"])"

   Run evaluate.py op test data → exact dezelfde metrics.

   Nieuwe data voorspellen: gebruik dezelfde evaluate.py workflow, laadt dezelfde checkpoint en maakt voorspellingen voor nieuwe samples.
7. **Deployment Issues:**
   Model drift / andere scanners: kan leiden tot lagere accuracy → mitigatie: periodieke retraining met nieuwe data.

   Data preprocessing mismatch: verschil in normalisatie, patch grootte of kleur → mitigatie: standaardiseer preprocessing in productie pipeline.

   Resource limitations: inference tijd of geheugen voor grote datasets → mitigatie: batch inference, model quantization of GPU gebruik.
---

## Question 5: Part 2 - Hyperparameter Optimization
1. **Search Space:**
Hyperparameters die je getest hebt:

Learning rate: [0.001, 0.0005]

Batch size: [16, 32]

Hidden units: [[64,32], [128,64]]
Dropout rate: [0.2, 0.3]

Waarom deze waarden?
- Learning rate: typisch startpunt voor MLP, kleiner voor fijnere optimalisatie.
- Batch size: 16 vs 32 om effect van mini-batch grootte te onderzoeken.
- Hidden units: twee configuraties voor modelcapaciteit, kleinere en grotere netwerken.
- Regelt overfitting, gekozen conservatief omdat dataset matig groot is.
2. **Visualization:**
(staan bij experiments/results)
confusion_matrix.png → laat zien hoe de voorspellingen verdeeld zijn.
roc_curve.png → voor ROC-AUC analyse.
pr_curve.png → Precision-Recall Curve.
threshold_analysis.png → laat zien welk threshold je kiest om FN te minimaliseren.
3. **The "Champion" Model:**
Confusion Matrix:
[[TN=5000, FP=3100],
 [FN=900,  TP=5000]]
 (img ook beschikbaar in results)
Accuracy : 0.6631

Precision: 0.6185

Recall : 0.8493

F1-score : 0.7158

4. **Thresholding Logic:**
ROC- en PR-curves zijn gemaakt van de champion model voorspellingen (y_true.npy, y_prob.npy).

In dit klinische scenario (detectie van carcinoma) is het minimaliseren van false negatives belangrijker dan precision.

Default threshold 0.5 zou te veel gevallen missen, daarom is gekozen voor threshold ≈ 0.3, waarmee recall hoger wordt:

Recall stijgt naar ~0.85

Precision daalt naar ~0.62 
5. **Baseline Comparison:**
Baseline (model dat altijd de meerderheid klasse voorspelt) heeft:

Accuracy ≈ 0.50

F1-score ≈ 0.0

Champion model verbetert:

Accuracy +16%

F1-score +0.72

Dit toont het duidelijke voordeel van systematisch MLOps hyperparameter search op dit dataset.
---

## Question 6: Model Slicing & Error Analysis
1. **Visual Error Patterns:**
Zie `experiments/results/slice_metrics_plot.png` voor een visuele vergelijking van globale vs slice metrics.
  
Global Performance
- Accuracy : 0.6631  
- Precision: 0.6185  
- Recall   : 0.8493  
- F1-score : 0.7158  

Slice Performance (Low-confidence tiles)
- Accuracy : 0.8839  
- Precision: 0.0000  
- Recall   : 0.0000  
- F1-score : 0.0000  
- Number of samples in slice: 11040 (8574 FP + 2466 FN)

Observatie:
1. False positives occur in gebieden met ongebruikelijke patronen.  
2. False negatives komen vaak voor bij low-confidence tiles (tiles waarop het model weinig vertrouwen heeft).  
3. De geselecteerde slice presteert slechter dan de globale metrics (precision/recall/f1=0).  
4. Alleen globale metrics monitoren is gevaarlijk in klinische context, omdat grote fouten op subgroepen verborgen blijven.

Based on the visualization and analysis of errors:  
- False positives may occur in regions with unusual or unexpected patterns.  
- False negatives predominantly happen on low-confidence tiles, where the model's predictions are less certain.  

**2. Definition of the slice**  
- **Slice:** Low-confidence tiles (tiles for which the model’s predicted probability was close to the decision threshold, i.e., uncertain predictions).  
- **Isolation method:** Tiles were filtered based on their prediction probability (`y_prob`) to select those with low confidence.  
- **Performance comparison:**  
  - Global metrics: Accuracy = 0.6631, Precision = 0.6185, Recall = 0.8493, F1-score = 0.7158  
  - Slice metrics: Accuracy = 0.8839, Precision = 0.0000, Recall = 0.0000, F1-score = 0.0000  
  The slice exhibits drastically worse performance in precision and recall, showing the model fails completely on these low-confidence tiles.  

**3. Risks of monitoring only global metrics**  
Monitoring only global metrics such as F1-score is dangerous in deployment because:  
- Global averages can hide catastrophic failures on specific subgroups or slices.  
- In a clinical context, low-confidence or misclassified tiles could correspond to carcinoma tissue, leading to missed diagnoses.  
- Without slice-level monitoring, these high-risk failure modes would go undetected, potentially compromising patient safety.

---

## Question 7: Team Collaboration and CI/CD
1. **Consolidation Strategy:** 
2. **Collaborative Flow:**

3. **CI Audit:**

4. **Merge Conflict Resolution:**

5. **Branching Discipline:**

---

## Question 8: Benchmarking Infrastructure
1. **Throughput Logic:**

2. **Throughput Table (Batch Size 1):**

| Partition | Node Type | Throughput (img/s) | Job ID |
| :--- | :--- | :--- | :--- |
| `thin_course` | CPU Only | | |
| `gpu_course` | GPU ([Type]) | | |

3. **Scaling Analysis:**

4. **Bottleneck Identification:**

---

## Question 9: Documentation & README
1. **README Link:** [Link to your Group Repo README]
2. **README Sections:** [Confirm Installation, Data Setup, Training, and Inference are present.]
3. **Offline Handover:** [List the files required on a USB stick to run the model offline.]

---

## Final Submission Checklist
- [ ] Group repository link provided?
- [ ] Best model checkpoint pushed to GitHub?
- [ ] inference.py script included and functional?
- [ ] All Slurm scripts included in the repository?
- [ ] All images use relative paths (assets/)?
- [ ] Names and IDs of all members on the first page?