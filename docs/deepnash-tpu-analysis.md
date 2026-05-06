# Which TPU did DeepNash use?

An evidence-based analysis of the TPU version most likely used to train DeepMind's DeepNash agent (Perolat et al., June 2022, [arXiv:2206.15378](https://arxiv.org/abs/2206.15378)).

## TL;DR

**Most likely: TPU v3.** Confidence ~80%. TPU v4 ~17%, mixed ~3%.

The paper specifies the amount of TPU compute used but never names the version. The argument for v3 rests on four converging lines of indirect evidence — none individually decisive, but pointing the same way.

## What the paper actually says

From the *Infrastructure and Setup* section of arXiv:2206.15378:

> "To train the final agent we used 768 TPU nodes used for Learners and 256 TPU nodes for Actors."

The paper also explicitly says it follows the Sebulba Podracer architecture (Hessel et al., 2021, [arXiv:2104.06272](https://arxiv.org/abs/2104.06272)).

Two project-timeline anchors from the paper itself:

- "Since June 2021, we have worked on DeepNash with Vincent de Boer..."
- "DeepNash was evaluated on Gravon, beginning of April 2022..."

So the bulk of training spans roughly mid-2021 to early-2022.

## The four lines of evidence

### 1. Other DeepMind papers in the same window

The strongest single data point. Gato — same lab, published May 2022, the month before DeepNash hit arXiv — explicitly states ([arXiv:2205.06175](https://arxiv.org/abs/2205.06175)):

> "Training of the model is performed on a 16x16 TPU v3 slice for 1M steps with batch size 512..."

Wider survey of contemporary DeepMind/Google papers:

| Paper | Date | TPU used | Source |
|---|---|---|---|
| Gato | May 2022 | TPU v3 (16×16 slice) | arXiv:2205.06175 |
| DeepNash | June 2022 | unspecified | arXiv:2206.15378 |
| Flamingo | April 2022 | TPU v4 (1,536 chips × 15 days) | per PaLI paper, openreview.net/pdf?id=mWVoBz4W0u |
| Chinchilla | March 2022 | TPU v3 *and* v4 | arXiv:2203.15556 ("All models in this analysis have been trained on TPUv3/TPUv4") |
| PaLM (Google Research) | April 2022 | TPU v4 (6,144 chips) | arXiv:2204.02311 |

The pattern: flagship LLM/foundation-model work got v4 capacity; agent and RL work was on v3. DeepNash sits in the second category.

### 2. Project timeline vs. v4 availability

The TPU v4 architecture paper ([arXiv:2304.01433](https://arxiv.org/abs/2304.01433), Jouppi et al., ISCA 2023) says v4 was "deployed since 2020" — meaning racked internally for a small number of teams. It was publicly announced at Google I/O in May 2021. It became broadly available in Google Cloud only at Google I/O 2022 (May 2022, Mayes County datacenter announcement; see HPCwire coverage 2022-05-16).

DeepNash's training window (mid-2021 through early 2022) sits squarely in the period when v4 was scarce even inside Google.

### 3. Non-flagship status

PaLM frames itself as "the largest TPU-based system configuration used for training to date" (April 2022, arXiv:2204.02311). That language tells you v4 capacity was a rationed resource even within Google's own research. The papers that got it were the splashy foundation-model runs. DeepNash is a strong paper, but a bounded RL project on a board game — not a candidate for the front of that queue.

### 4. The paper's silence

Contemporary DeepMind/Google papers that did use v4 mentioned it specifically — Chinchilla, Flamingo, and PaLM all name the platform. DeepNash names neither. The boring interpretation is that the platform wasn't worth flagging because it was the standard one (v3). If DeepNash had used v4, that would have been a notable detail at the time, and silence is mild evidence against it.

## What I considered and dropped

A first pass through this question leaned heavily on a numerical argument: the paper reports 768 + 256 = 1,024 "TPU nodes," which matches a TPU v3 pod (1,024 chips) exactly. On closer inspection that argument doesn't hold up.

Per the Google Cloud glossary (developers.google.com/machine-learning/glossary/googlecloud), a "TPU node" is a *resource* of variable size — e.g., v3-8 (4 chips, 8 cores) or v3-2048 (full pod). The original Sebulba paper uses "machine" to refer to a v3-8 host: 4 chips, 8 cores per unit. If DeepNash follows that convention — which is the natural reading given the explicit Sebulba citation — then 1,024 nodes most likely means 1,024 hosts × 4 chips = 4,096 chips total. That's either a single v4 pod or four v3 pods. The number doesn't disambiguate.

So the chip-count is the only piece of *direct* numerical evidence in the paper, and it's neutral on its own. The case for v3 rests entirely on the four indirect lines above.

## Confidence and caveats

~80% TPU v3 / ~17% TPU v4 / ~3% mixed.

The mixed scenario (prototype on v3, final run on v4) is the one I'd give the most charitable reading to — Chinchilla did exactly that — but if it had happened here I'd expect the paper to mention it as Chinchilla did.

The cleanest way to resolve this would be to ask one of the authors directly. The corresponding authors listed in the paper are perolat@deepmind.com, bartdv@deepmind.com, and karltuyls@deepmind.com.

## Sources

| # | Reference | Use |
|---|---|---|
| 1 | Perolat et al., 2022. *Mastering the Game of Stratego with Model-Free Multiagent Reinforcement Learning.* arXiv:2206.15378 | Primary subject |
| 2 | Reed et al., 2022. *A Generalist Agent* (Gato). arXiv:2205.06175 | Same-lab v3 evidence |
| 3 | Hoffmann et al., 2022. *Training Compute-Optimal Large Language Models* (Chinchilla). arXiv:2203.15556 | v3/v4 mix attestation |
| 4 | Alayrac et al., 2022. *Flamingo: a Visual Language Model for Few-Shot Learning.* arXiv:2204.14198 | v4 attestation (via PaLI citation, openreview.net/pdf?id=mWVoBz4W0u) |
| 5 | Chowdhery et al., 2022. *PaLM: Scaling Language Modeling with Pathways.* arXiv:2204.02311 | v4 scarcity / "largest configuration to date" |
| 6 | Hessel et al., 2021. *Podracer architectures for scalable Reinforcement Learning.* arXiv:2104.06272 | Sebulba origin & v3 design ("2048 cores of a TPU Pod") |
| 7 | Jouppi et al., 2023. *TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning.* arXiv:2304.01433 | v4 deployment timeline ("Deployed since 2020") |
| 8 | Google I/O 2021 keynote (Sundar Pichai, May 18 2021) | Public TPU v4 announcement |
| 9 | HPCwire, "Google Cloud's New TPU v4 ML Hub Packs 9 Exaflops of AI", 2022-05-16 | Cloud preview / Mayes County |
| 10 | Google Cloud ML Glossary, developers.google.com/machine-learning/glossary/googlecloud | "TPU node" definition |
