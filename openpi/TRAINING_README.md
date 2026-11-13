# Train ForceVT

Use a Linux CUDA workstation with Python 3.12 and [uv](https://docs.astral.sh/uv/). From this directory:

```bash
uv sync --no-dev
DATA_ROOT_BASE=/path/to/516_close_bottle_multihealth_lerobot \
  uv run bash scripts/handcap_training/512_close_bottle/train_health_distill_tactile_wrist_force.sh
```

The launcher computes normalization statistics, trains the JAX π₀.₅ policy and saves checkpoints in `checkpoints/<config>/<experiment>/`. The model starts from the public OpenPI π₀.₅ base checkpoint. Set up Google Cloud Storage access if downloading the default checkpoint with `gsutil`. Set `PI05_BASE_PARAMS=/path/to/pi05_base/params` to use a local copy. Override `BATCH_SIZE`, `FSDP_DEVICES`, `CUDA_VISIBLE_DEVICES` and `NUM_WORKERS` to match your hardware. Set `RESUME=1` to continue a run that has a checkpoint.

## Data

The four paper task launchers expect a separate LeRobot v3 HandCap dataset for each of the 50, 75 and 100 tactile fidelity groups. Each dataset root must contain `data/`, `meta/` and `videos/`, with synchronized wrist, tactile, force, pose, gripper and instruction fields. Set `HEALTH50_DATA_ROOT`, `HEALTH75_DATA_ROOT` and `HEALTH100_DATA_ROOT` to point to your prepared datasets, or match the directory names in the launcher. The Bread Moving example uses groups 0, 50 and 100.

The [data guide](../data/README.md) links the Hugging Face archives to the paper tasks and explains what is needed to convert the recordings to the LeRobot layout.

## Launchers

- [Erase board](scripts/handcap_training/510_erase_board/train_health_distill_tactile_wrist_force.sh)
- [Screwdriving](scripts/handcap_training/513_screw/train_health_distill_tactile_wrist_force.sh)
- [Close bottle](scripts/handcap_training/512_close_bottle/train_health_distill_tactile_wrist_force.sh)
- [Stowing](scripts/handcap_training/516_stowing/train_health_distill_tactile_wrist_force.sh)
- [Bread Moving example](scripts/handcap_training/bread_moving/train_health_distill_tactile_wrist_force.sh)

The June training launchers use relational force–vision distillation and cross-modal alignment, with optional tactile random masking. Set `TACTILE_MASK_RATIO` to enable the retained spatial mask, and set `BATCH_MASK_RATIO` to control the fraction of observations eligible for masking (default `0.5`).
