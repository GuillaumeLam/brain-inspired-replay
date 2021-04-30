# IFT6760B-Continual Learing, UdeM, Winter 2021

Contrasting episodic and generative replay as a function of exemplar quality in continual learning

Guillaume Lam, guillaume.lam@umontreal.ca \
Shima Rastegarnia, shima.rastegarnia@umontreal.ca \
Marie St-Laurent, marie.st-laurent@umontreal.ca 

Note: The master branch contains the origin repo's code while the dev branch contains all additional code for our final project.

## Abstract

Our work investigated the use of episodic and generative replay as methods to prevent catastrophic forgetting during challenging continual learning (CL) tasks and scenarios. We contrasted the efficacy of episodic replay (ER) and brain-inspired replay (BI-R), a technique that adapts standard generative replay based on neuroscience principles and scales well to long task sequences and complex stimuli. We used permuted MNIST and split CIFAR in domain-inference and task-inference learning scenarios, respectively. For each type of replay, we also manipulated the quantity and quality of replayed exemplars. Our goal was to contrast the effectiveness of ER versus BI-R, and to explore the minimal amount of replay required by each approach to maintain performance as more tasks are learned. We observed that BI-R performs closely to ER without needing to store data from previous tasks, that both approaches prevent forgetting with very few replayed exemplars per iteration, and that BI-R is surprisingly robust to poor generated exemplar quality.

The following text is adapted from the GMvandeVen/continual-learning Repository's ReadMe
Source: https://github.com/GMvandeVen/continual-learning

---

# Brain-Inspired Replay
A PyTorch implementation of the continual learning experiments with deep neural networks described in the 
following paper:
- Brain-inspired replay for continual learning with artificial neural networks: https://www.nature.com/articles/s41467-020-17866-2

This paper proposes a new, brain-inspired version of generative replay that can scale to continual learning problems with natural images as inputs.
This is demonstrated with the Split CIFAR-100 protocol, both for task-incremental learning and for class-incremental learning.


## Installation & requirements Update:
The code runs with no issue with `Python >=3.7`. The versions that were used for Python-packages are listed in `requirements.txt`.


## Running custom experiments
Using `main_cl.py`, it is possible to run custom individual experiments. The main options for this script are:
- `--experiment`: which task protocol? (`splitMNIST`|`permMNIST`|`CIFAR100`)
- `--scenario`: according to which scenario? (`task`|`domain`|`class`)
- `--tasks`: how many tasks?

To run specific methods, use the following:
- Context-dependent-Gating (XdG): `./main_cl.py --xdg --xdg-prop=0.8`
- Elastic Weight Consolidation (EWC): `./main_cl.py --ewc --lambda=5000`
- Online EWC:  `./main_cl.py --ewc --online --lambda=5000 --gamma=1`
- Synaptic Intelligenc (SI): `./main_cl.py --si --c=0.1`
- Learning without Forgetting (LwF): `./main_cl.py --replay=current --distill`
- Generative Replay (GR): `./main_cl.py --replay=generative`
- Brain-Inspired Replay (BI-R): `./main_cl.py --replay=generative --brain-inspired`

To run additional methods on the dev branch, use the following:
- Episodic Replay (ER): `./main_cl.py --replay=exemplars`

The options available for ER are:
- `--sampling`: which sampling type? (`random`|`herding`)
- `--distortion`: which distortion to apply to replay examplars? (`fine`|`med`|`hyper`)

For information on further options: `./main_cl.py -h`.

---

## Commands to run experiments
The following table outlines the commands with some option values to get the results in the `Contrasting episodic and generative replay as a function of exemplar quality in continual learning` paper.

Dataset | Replay Type | Description | Command
:---:|:---:|:---:|:---:
Permuted-MNIST | Bi-R | Half batch size and 100 VAE units(default) | `./main_cl.py --experiment=permMNIST --scenario=domain --tasks=50 --replay=generative --batch-replay=128 --brain-inspired --no-save --pdf`
Permuted-MNIST | Bi-R | Full batch size and 50 VAE units | `./main_cl.py --experiment=permMNIST --scenario=domain --tasks=50 --replay=generative --z-dim=50 --brain-inspired --no-save --pdf`
Permuted-MNIST | ER | Half batch size and ~10 ex per class x task | `./main_cl.py --experiment=permMNIST --scenario=domain --tasks=50 --replay=exemplars --sampling=random --batch-replay=128  --budget=5000 --no-save --pdf`
CIFAR100 | Bi-R | Half batch size and 100 VAE units(default) | `./main_cl.py --experiment=CIFAR100 --scenario=class --replay=generative --batch-replay=128 --brain-inspired --no-save --pdf`
CIFAR100 | ER | Half batch size and ~5 ex per class x task | `./main_cl.py --experiment=CIFAR100 --scenario=class --freeze-convE --pre-convE --replay=exemplars --sampling=random --batch-replay=8  --budget=450 --no-save --pdf`
