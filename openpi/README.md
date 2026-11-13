# ForceVT training and inference

ForceVT is the AetheRock policy trained on wrist images, bilateral tactile observations and gripper-force measurements. This package is based on the June 2026 OpenPI snapshot and uses the JAX π₀.₅ backbone. See the [paper](https://arxiv.org/abs/2606.09777) and [project page](https://lihongcs.github.io/aetherock/).

- [Training setup, data and launchers](TRAINING_README.md)
- [Policy serving and Flexiv inference](docs/handcap_real_robot_inference.md)

OpenPI and the retained LeRobot dataset code are distributed under the [Apache 2.0 License](LICENSE).
