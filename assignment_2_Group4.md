# Assignment 2: MLOps & PCAM Pipeline Journal
**MLOps & ML Programming (2026)**

## Group Information
* **Group Number:** 4
* **Team Members:** Steven Voerknecht, 14666928; Max van Aerssen; 15230015, Elia Loeb, Member 4 Name/ID, Member 5 Name/ID
* **GitHub Repository:** [Link naar onze Git repository](https://github.com/stevenvoerknecht/MLOpps-homework)
* **Base Setup Chosen from Assignment 1:** De Setup gegeven in de mail

---

## Question 1: Reproducibility Audit
1. **Sources of Non-Determinism:**   
Er zijn een aantal bronnen van non-determinisme in onze code.  
Ten eerste zijn er aantal plekken waar random functies worden aangeroepen door random, numpy en pytorch. Berekeningen die door PyTorch worden gedaan zijn uit zichzelf willekeurig, dus moeten deterministisch worden gemaakt om voorspelbaar te zijn.  
Ten tweede worden er door de dataloader workers gebruikt. Zodra er meerdere workers worden gebruikt kan dit ervoor zorgen dat dingen random uitkomsten hebben. Ook wordt een geseede generator gebruikt voor de dataloader zodat het loaden van data voorspelbaar gebeurt.  
Ten derde kunnen er verschillen plaatsvinden in de code door het gebruiken van een net andere dataset.  
Ten vierde kan er non-determinisme plaatsvinden vanwege kleine gebreken in de hardware.  
Ten vijfde kan er non-determinisme plaatsvinden als er verschillende versies van libraries worden gebruikt. 


2. **Control Measures:**  
Ten eerste worden alle random functies geseed voordat ze verder in de code worden gebruikt in de functie seed_everyting.  
Ten tweede worden de workers in de dataloader geseed en wordt er een generator aangemaakt voor de dataloader in get_dataloaders.  
Ten derde worden er geen veranderingen aangebracht aan de dataset in onze code waardoor dit tot geen problemen leidt.  
Het vierde punt van hardware is moeilijk op te lossen dus hier zijn geen aanpassingen in de code.  
Het gebruiken van dezelfde libraries wordt gedeeltelijk vastgesteld in de slurm-job en pyproject.toml. Voor een groot deel van de library wordt simpelweg de meest recente versie gebruikt, dus dit kan voor veranderingen zorgen als er een nieuwe versie uitkomt. Dit is echter niet zo'n gigantisch probleem omdat dit niet heel vaak gebeurt. 

3. **Code Snippets for Reproducibility:**
   ```python
   def seed_worker(worker_id):
      # seed every worker for reproducibility
      worker_seed = seed + worker_id
      np.random.seed(worker_seed)
      random.seed(worker_seed)
   ```
   ```python
   g = torch.Generator()
   g.manual_seed(seed)
   ```
   ```python
   def seed_everything(seed: int):
      """Ensures reproducibility across numpy, random, and torch."""
      random.seed(seed)
      np.random.seed(seed)
      torch.manual_seed(seed)
      torch.cuda.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)

      # Using deterministic execution
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False
      torch.use_deterministic_algorithms(True)
   ```

4. **Twin Run Results:**   
Na het runnen van dezelfde code op 2 verschillende computers verschilde de uitkomst van zowel de train als de validation loss nog steeds. Wij vermoeden dat dit komt doordat ze op verschillende gpu nodes waren gerund en deze net andere hardware hadden. 

---

## Question 2: Data, Partitioning, and Leakage Audit
1. **Partitioning Strategy:**  
We hebben de train/validation split gebruikt die al voorgegeven was in de dataset. Dit kan er misschien voor zorgen dat er moeilijker validation accuracy wordt verkregen, maar zelf hersplitten kan leiden dat dataleakage. Vandaar dat we deze hebben gebruikt. Er wordt wel een shuffler (met sampler) gebruikt bij de aparte train- en validation loader.  
De uiteindelijke verhouding was ~17,2% validation en ~82,8% training set. 

2. **Leakage Prevention:**  
De eerste vorm van data leakage prevention is dat we niet hebben gere-split omdat dit ervoor kan zorgen dat dezelfde datapunten of datapunten van dezelfde patienten voor kunnen komen in dezelfde dataset en dat kan leiden dat data leakage. Ook het berekenen van statistieken over de dataset gebeurt los tussen de train en validation set en alleen de resultaten van de validation set worden gebruikt om representatief te zijn. Verder gebeurt het pre-processen van data alleen in de dataloader bij __get_item__ van de PCAMDataset class. 
   
3. **Cross-Validation Reflection:**  
nested k-fold cross validation is hier niet nodig omdat er al een goede split is gegeven in de data die werkt en data leakage voorkomt. Dat hersplitten voegt niks toe en is risicovol. Ook is k-fold cross validation computationeel ongelooflijk duur en dus onpraktisch. Het gebruiken van de metrics die berekent zijn op de validation set is voldoende om hyperparamaters te tunen. 

4. **The Dataset Size Mystery:**  
De dataset was veel groter omdat de pixels in uint32 waren gegeven terwijl je aan uint8 al voldoende hebt. Hierdoor heb je veel meer bytes nodig om de plaatjes weer te geven. Dit is makkelijk op te lossen door de dataset naar uint8 te casten. 

5. **Poisoning Analysis:**  
Er zijn verschillende vormen van data poisining. Een aantal opties zijn label poisining (verkeerde labels), Backdoor poisining (iets kleins toegevoegd aan de pixels waaraan het model het correcte label kan voorspellen), Datadistribution poisining (verschillende distributie tussen train en validation) en data representation poisining (data opgeslagen met verkeerde datatype). Ik vermoede data representation poisining omdat pixels normaal tussen de 0 en 255 moeten liggen voor de verschillende kleuren, maar deze pixels hadden een veel grotere range en een veel hoger gemiddelde. Dit kwam dus omdat ze in uint32 stonden. Na het runnen van print(images.dtype) was dit vermoeden bevestigd. 

---

## Question 3: Configuration Management
1. **Centralized Parameters:**  
De paths naar bijvoorbeeld de dataset of de output directory kunnen niet gehardcode worden omdat je groepsgenoten ze dan niet kunnen gebruiken, ook heel veel model parameters wil je niet hardcoden. Een aantal voorbeelden daarvan zijn het aantal epochs, de learning rate, de dropout rate of het de hidden layer size. Ook seeds wil je kunnen aanpassen en moeten centraal kunnen veranderen. Je wil ook dingen aan het runnen van het model kunnen aanpassen zoals het gebruik van gpu of cpu, het werkgeheugen, het aantal workers of de batchsize. 

2. **Loading Mechanism:**
   - Deze parameters kunnen worden aangepast in het config.yaml of in het slurm jobscript om verschillende parameters te testen. 
   ```python
   parser = argparse.ArgumentParser(description="Train a Simple MLP on PCAM")
   parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
   args = parser.parse_args()
   ```

3. **Impact Analysis:**  
Het centraal stellen van deze parameters en vooral het centrale seeden zorgt voor enorm voorspelbare uitkomsten. Hiermee kan je ook makkelijker experimenten vergelijken omdat alle groepsleden hetzelfde config.yaml en jobscript bestandje kunnen gebruiken voor het runnen. Samenwerken werd daardoor ook veel makkelijker. Voordat alles centraal was moesten groepsleden telkens een nieuwe config.yaml en jobscript bestandje schrijven voor hun tests. 

4. **Remaining Risks:**  
Een huidig risico dat we hebben is dat de relatieve paths naar de dataset en de ouput directory alleen werken als de dataset op de juiste plek in de structuur van de folders staat. Als de dataset niet in een apart mapje in de MLOps_2026 map staat onder de naam data/surfdrive werkt de code niet meer voor iedereen. Ook moet het jobscript gerund worden vanuit de MLOps_2026 folder en niet de MLOps_2026/slurm_jobs folder want dan werkt ${PROJECT_ROOT} niet correct. 

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
We hebben de codebase van het MLOps-team als basis gebruikt, omdat niemand van ons opdracht 1 had voltooid. We hebben dit gedaan door ‘git remote add other <url>’ uit te voeren, te fetchen, samen te voegen en een paar samenvoegingsproblemen handmatig op te lossen. Dit werkte omdat we al een licht bewerkte MLOps_2026-repo in de github-repository hadden. 

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
1. **README Link:** Je kan de README.md vinden in MLOps_2026/README.md of via: [Devlink to README.md](https://vscode.dev/github/stevenvoerknecht/MLOpps-homework/blob/dev/README.md)
2. **README Sections:** Het runnen van de installatie, data setup, training en inference op een nieuwe node levert de verwachte resultaten op.
3. **Offline Handover:** Om het model offline uit te voeren, heb je de dataset, het beste checkpoint.pt-bestand en de volledige Python-code repository nodig, een venv met de juiste dependencies geïnstalleerd, het config.yaml bestand en de slurm jobscript op een USB-stick.

---

## Final Submission Checklist
- [ ] Group repository link provided?
- [ ] Best model checkpoint pushed to GitHub?
- [ ] inference.py script included and functional?
- [ ] All Slurm scripts included in the repository?
- [ ] All images use relative paths (assets/)?
- [ ] Names and IDs of all members on the first page?
