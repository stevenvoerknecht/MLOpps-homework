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

2. **Results (Average of 3 Seeds):**

3. **Logging Scalability:**

4. **Tracker Initialization:**
   ```python
   # Snippet showing tracker/MLFlow/W&B initialization
   ```

5. **Evidence of Logging:**

6. **Reproduction & Checkpoint Usage:**

7. **Deployment Issues:**

---

## Question 5: Part 2 - Hyperparameter Optimization
1. **Search Space:**
2. **Visualization:**
3. **The "Champion" Model:**

4. **Thresholding Logic:**

5. **Baseline Comparison:**

---

## Question 6: Model Slicing & Error Analysis
1. **Visual Error Patterns:**

2. **The "Slice":**

3. **Risks of Silent Failure:**

---

## Question 7: Team Collaboration and CI/CD
1. **Consolidation Strategy:** 
2. **Collaborative Flow:**

3. **CI Audit:**
de standaard GPU-enabled install op de github runner zou ervoor zorgen dat de install enorm veel tijd kost en deze zou waarschijnlijk de opslag van de github runner overschrijden.

Door de CI worden er een aantal pytests uitgevoerd en wanneer ide niet slagen wordt de merge geblokkeerd. hierdoor wordt er dus niet gemerged wanneer belangrijke architectuur door een teamgenoot wordt aangepast.
4. **Merge Conflict Resolution:**

5. **Branching Discipline:**
* cbd3ac6 example.py fix
* defa664 pull fix
* dcd6225 pull test
* 28824ee (origin/dev, dev) Assignment 8 in dev
* 960e23e Adding a few new changes to the slurm job
* 5c92e68 Changing the configuration and jobscript so everyone can use the same one
* c98ebf7 removed a run result
* fd35682 Finishin trainer.py and train.py and seeding everythin
* f8d310a (origin/trainer) The first run with 3 epochs
* 6771ef7 (origin/Stevens_assignment1) Adding a working jobscript
* 05e219d first testrun with the dataset
* 74c733c changing the path to the dataset (in scratch-shared)
* d257129 slight change to .gitignore
| * 822428c (origin/feature/throughput-benchmarking, feature/throughput-benchmarking) Vraag 8 toegevoegd aan md

onze geschiedenis is non-linear. Dit is voor ons op dit moment gewenst omdat we allemaal apart en tegelijkertijd aan onze opdrachten willen werken en het later met elkaar samen willen voegen.

---

## Question 8: Benchmarking Infrastructure
1. **Throughput Logic:**
Ik heb apart gemeten. Hierbij meet je precies hoe lang de CPU en GPU bezig zijn. Ook kan je als er problemen zijn amkkelijker zien waar het model vastloopt. Bij meting tijdens training wordt wel het hele process meegenomen inclusief het ophalen van de data. Het ophalen van de data en de validation kan interferen met de throughput measurement.

start_time = time.time()
iterations = 500

for _ in range(iterations):
    _ = model(dummy_input)
    
    # Cruciaal voor GPU: wacht tot de berekening echt klaar is 
    # voordat de loop doorgaat naar de volgende iteratie of stopt.
    if device.type == 'cuda': 
        torch.cuda.synchronize()

end_time = time.time()

total_time = end_time - start_time
throughput = iterations / total_time

Ik denk dat float16 of float32 de precission zeker affect. Met float 32 worden er veel meer bits tegelijk naar de GPU verstuurd wat zeker invleod zal hebben op de precission.

Als het heel druk is op snellius kan je een aantal nodes moeten delen. Dit zal zeker effect hebben op je throughput.
2. **Throughput Table (Batch Size 1):**

| Partition | Node Type | Throughput (img/s) | Job ID |
| :--- | :--- | :--- | :--- |
| `thin_course` | CPU Only |9738.25 img/s |18476510|
| `gpu_course` | GPU ([Type]) |6804.23 img/s |18476737|

3. **Scaling Analysis:**
Batch Size	Throughput (img/s)	Scaling Factor 
8	        45.385,53	        1.0x
16	        88.488,59	        1.95x
32	        127.043,58	        2.80x
128	        229.524,22	        5.06x
256	        254.612,54	        5.61x
512	        253.402,96	        5.58x

We zien dat het van 8 tot 32 heel erg stijgt en dat het vanaf 128 tot 256 weer afzwakt en van 256 tot 512 wee lager wordt. Het plateau zot dus rond 256.

GPU Naam: NVIDIA A100-SXM4-40GB MIG 1g.5gb
Max VRAM: 209.19 M
4. **Bottleneck Identification:**
We hebben gezien dat de GPU een throughput van 253.402 img/s heeft. Dit is extreem snel. Ik denk dat de CPU dit niet evensnel kan doen dus dat de CPU prerpossesing het langzaamst is.
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
