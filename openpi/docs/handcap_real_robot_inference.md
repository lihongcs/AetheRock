# Serve ForceVT

The policy server restores a trained JAX checkpoint and the normalization statistics saved with it. From the OpenPI directory:

```bash
uv run python scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config=pi05_512_close_bottle_health_distill_tactile_wrist_force \
  --policy.dir=/path/to/checkpoints/config/experiment/step
```

Keep both `params/` and `assets/` in the checkpoint directory. Use the same config name that was used to train it. The server listens on localhost by default.

## Flexiv client

The robot client requires a working Flexiv RDK installation and the camera drivers for your setup. Make a copy of [the camera config](../src/perception/configs/camera/handcap_realsense_tactile.json) and fill in the MVS camera serial number, RealSense serial numbers and both tactile camera indices. Pass its path through `CAMERA_CONFIG_PATH`. Set `FLEXIV_ROBOT_IP`, `FLEXIV_ROBOT_SN`, `FLEXIV_INIT_POSE`, `POLICY_DIR` and `PROMPT` for your setup. Use a verified seven-value initial pose for the robot. Set `POLICY_CONFIG` to match the checkpoint. Then run:

```bash
uv run bash run_inference_pi05_health_distill_tactile_wrist_force.sh --dry-run
```

The dry run checks the policy and client without commanding robot motion.
