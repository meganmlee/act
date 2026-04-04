# ACT-Tokenizer: Action Chunking with Transformers + Language Conditioning + Discrete Tokenization

Fork of [ACT](https://tonyzhaozh.github.io/aloha/) with two extensions that can be used independently or together:

1. **LAV-ACT** -- Language-conditioned ACT. A frozen CLIP text encoder extracts a task embedding (e.g. "pick up the ketchup"), which is projected and concatenated as an extra token in the transformer sequence. This lets a single model handle multiple tasks.
2. **FAST-ACT** -- Discrete action tokenization via FAST+. The CVAE encoder embeds discrete tokens instead of raw actions, and the decoder predicts token logits. After decoding, tokens are detokenized back to continuous actions.

These compose: LAV-ACT + FAST gives language-conditioned multi-task training with discrete action tokens.

## Variants

| Variant | Actions | Language | Training | Flag(s) |
|---------|---------|----------|----------|---------|
| ACT (baseline) | Continuous | No | Per-task | _(none)_ |
| LAV-ACT | Continuous | CLIP | Multi-task | `--use_language` |
| FAST-ACT | FAST discrete | No | Per-task | `--use_fast_tokens` |
| LAV-ACT + FAST | FAST discrete | CLIP | Multi-task | `--use_language --use_fast_tokens` |

## Architecture

### ACT baseline
- **CVAE encoder**: `[CLS, qpos, actions]` -> latent z
- **Decoder memory**: `[latent, proprio, image_feats]`
- Action queries cross-attend to memory -> continuous action predictions (L1 loss)

### LAV-ACT
Adds a frozen CLIP text token to both encoder and decoder:
- **CVAE encoder**: `[CLS, text_token, qpos, actions]`
- **Decoder memory**: `[latent, proprio, text_token, image_feats]`
- Text token = `Linear(CLIP_text_encoder(task_string))` (CLIP is frozen, projection is learned)
- Still continuous actions, same L1 loss

### FAST-ACT
Same architecture as ACT, but actions are discrete BPE tokens (DCT + BPE encoding):
- Encoder embeds tokens via `nn.Embedding` instead of linear projection
- Decoder predicts token logits (cross-entropy loss instead of L1)

### LAV-ACT + FAST
Combines both: language token in the sequence + discrete action tokens.
- **CVAE encoder**: `[CLS, text_token, qpos, action_tokens]`
- **Decoder memory**: `[latent, proprio, text_token, image_feats]`
- Cross-entropy loss over discrete token vocabulary

## Setup

```bash
conda create -n aloha python=3.8.10
conda activate aloha

# Core dependencies
pip install torch torchvision
pip install pyquaternion pyyaml rospkg pexpect
pip install mujoco==2.3.7 dm_control==1.0.14
pip install opencv-python matplotlib einops packaging h5py ipython
pip install transformers  # for CLIP text encoder + FAST tokenizer

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

Per-suite training and evaluation on LIBERO, following the protocol used by OpenVLA-OFT, pi0, Dream-VLA, and MM-ACT:

- **ACT / FAST-ACT**: Train one policy per task (10 separate models per suite via SLURM array)
- **LAV-ACT / LAV-ACT+FAST**: Train one policy across all 10 tasks (single model, language-conditioned)
- Evaluate on the same 10 tasks with different initial conditions (50 rollouts per task)
- Report per-task success rates and suite average

| Suite | # Tasks | Episode Length | Description |
|---|---|---|---|
| LIBERO-Spatial | 10 | 300 | Spatial relationship reasoning |
| LIBERO-Object | 10 | 300 | Object recognition/manipulation |
| LIBERO-Goal | 10 | 300 | Goal-conditioned manipulation |
| LIBERO-Long | 10 | 600 | Long-horizon multi-step tasks |

## Running Experiments

### Step 0: Train FAST tokenizer (once, only needed for FAST variants)

```bash
python tokenizer.py \
    --dataset_path /path/to/libero_90 \
    --save_path ./fast_tokenizer \
    --chunk_size 50 --action_dim 7
```

### ACT baseline (per-task, continuous actions)

```bash
sbatch job.sh libero_spatial
```

### LAV-ACT (multi-task, continuous actions + CLIP language)

```bash
sbatch job_LACT.sh libero_spatial
```

### FAST-ACT (per-task, discrete tokens)

```bash
sbatch job_actFAST.sh libero_spatial
```

To add language conditioning on top (LAV-ACT + FAST), uncomment `LANG_ARGS` in `job_actFAST.sh`, remove the `--array` SBATCH header, and remove `--task_id`.

All suites: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`

### Evaluation

Eval is included at the bottom of each job script as a commented-out block. Uncomment it to run eval after training completes.

Checkpoints and eval results are saved to:
- `checkpoints/{suite}_act/` -- ACT baseline (per-task subdirs)
- `checkpoints/{suite}_lact/` -- LAV-ACT (single model)
- `checkpoints/{suite}_act_fast/` -- FAST-ACT (per-task subdirs)
- `checkpoints/{suite}_act_fast/` -- LAV-ACT + FAST (when LANG_ARGS enabled)

## Swapping tokenizers

The tokenizer is pluggable via the `ActionTokenizer` base class in `tokenizer.py`. To add a new one, subclass it and decorate with `@register_tokenizer`. The rest of the codebase loads tokenizers via `load_tokenizer(path)` which auto-dispatches based on a saved type marker.

## Repo Structure
- `imitate_episodes.py` -- Train and evaluate ACT
- `policy.py` -- ACT policy wrapper
- `tokenizer.py` -- Action tokenizer interface + FAST+ implementation
- `detr/` -- Model definitions (DETRVAE), modified from DETR
- `constants.py` -- Task configs and constants
- `utils.py` -- Data loading (continuous + tokenized + language)
- `job.sh` -- ACT baseline (per-task)
- `job_LACT.sh` -- LAV-ACT (multi-task, continuous + language)
- `job_actFAST.sh` -- FAST-ACT (per-task, discrete tokens)
- `eval.sh` / `eval_fast.sh` -- Standalone eval scripts (legacy)
