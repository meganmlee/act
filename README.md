# ACT-Tokenizer: Action Chunking with Transformers + Discrete Action Tokenization

Fork of [ACT](https://tonyzhaozh.github.io/aloha/) that adds discrete action tokenization (FAST+) to the CVAE. The CVAE encoder embeds discrete tokens instead of raw actions, and the decoder predicts token logits. After decoding, tokens are detokenized back to continuous actions.

## Setup

```bash
conda create -n aloha python=3.8.10
conda activate aloha

# Core dependencies
pip install torch torchvision
pip install pyquaternion pyyaml rospkg pexpect
pip install mujoco==2.3.7 dm_control==1.0.14
pip install opencv-python matplotlib einops packaging h5py ipython
pip install transformers  # for FAST tokenizer

# Install DETR
cd detr && pip install -e . && cd ..

# Install LIBERO (needed for LIBERO tasks)
cd ..
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
```

When running, make sure LIBERO is on your path:
```bash
export PYTHONPATH=/path/to/LIBERO:$PYTHONPATH
```

## Evaluation Protocol

Per-suite multi-task training and evaluation on LIBERO, following the protocol used by OpenVLA-OFT, pi0, Dream-VLA, and MM-ACT:

- Train one policy on all 10 tasks within a single suite (50 demos per task)
- Evaluate on the same 10 tasks with different initial conditions (20 rollouts per task)
- Report per-task success rates and suite average
- Repeat for each suite, then average across suites for the headline number

| Suite | # Tasks | Episode Length | Description |
|---|---|---|---|
| LIBERO-Spatial | 10 | 300 | Spatial relationship reasoning |
| LIBERO-Object | 10 | 300 | Object recognition/manipulation |
| LIBERO-Goal | 10 | 300 | Goal-conditioned manipulation |
| LIBERO-Long | 10 | 600 | Long-horizon multi-step tasks |

## Running Experiments

### Step 0: Train FAST tokenizer (once)

```bash
python tokenizer.py \
    --dataset_path /path/to/libero_90 \
    --save_path ./fast_tokenizer \
    --chunk_size 50 --action_dim 7
```

### Per-suite training + eval

Both job scripts take a suite name as an argument and run training followed by evaluation.

```bash
# ACT baseline (continuous actions)
sbatch job.sh libero_spatial
sbatch job.sh libero_object
sbatch job.sh libero_goal
sbatch job.sh libero_10          # LIBERO-Long

# ACT + FAST tokens
sbatch job_actFAST.sh libero_spatial
sbatch job_actFAST.sh libero_object
sbatch job_actFAST.sh libero_goal
sbatch job_actFAST.sh libero_10  # LIBERO-Long
```

Checkpoints are saved to `checkpoints/{suite}_act/` and `checkpoints/{suite}_act_fast/`.

Evaluation results are written to `checkpoints/{suite}_{variant}/eval_all_tasks.txt`.

## Swapping tokenizers

The tokenizer is pluggable via the `ActionTokenizer` base class in `tokenizer.py`. To add a new one, subclass it and decorate with `@register_tokenizer`. The rest of the codebase loads tokenizers via `load_tokenizer(path)` which auto-dispatches based on a saved type marker.

## Repo Structure
- `imitate_episodes.py` — Train and evaluate ACT
- `policy.py` — ACT policy wrapper
- `tokenizer.py` — Action tokenizer interface + FAST+ implementation
- `detr/` — Model definitions (DETRVAE), modified from DETR
- `constants.py` — Task configs and constants
- `utils.py` — Data loading (continuous + tokenized)
- `job.sh` — SLURM job for ACT baseline (per-suite)
- `job_actFAST.sh` — SLURM job for ACT + FAST tokens (per-suite)
