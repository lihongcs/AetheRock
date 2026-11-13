# Normalization statistics

Each ForceVT launcher runs `scripts/compute_norm_stats.py` on the selected tactile fidelity datasets. It writes `norm_stats.json` under `assets/<config>/<asset-id>/`. Training copies these statistics into the checkpoint. The policy server reads the copy from the checkpoint's `assets/` directory.
