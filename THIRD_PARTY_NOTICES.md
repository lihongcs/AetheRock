# Third-party notices

AetheRock's original software contributions, including the HandCap collector and the ForceVT implementation, are available under the [Apache License, Version 2.0](LICENSE). Third-party code retains its original license and attribution.

- **OpenPI** by Physical Intelligence provides the training and policy framework in `openpi/`. The AetheRock fork adapts the model, training, data loading, and serving code for ForceVT. The Big Vision code used by the Gemma and SigLIP components retains its original file headers. [OpenPI source](https://github.com/Physical-Intelligence/openpi) · [Apache 2.0 license](openpi/LICENSE)
- **LeRobot** by Hugging Face supplies the vendored dataset code in `handcap/Postprocess/lerobot/`, `handcap/Postprocess/lerobot30_env/lerobot/`, and `openpi/src/openpi/lerobot/`. This distribution includes HandCap and ForceVT adaptations. [LeRobot source](https://github.com/huggingface/lerobot) · [License and upstream notices](licenses/LeRobot.txt)
- **Universal Manipulation Interface (UMI)** by the Columbia Artificial Intelligence and Robotics Lab supplies utilities under `handcap/umi/`, `handcap/Postprocess/umi_utils/`, and `handcap/time_calib/`, and the `precise_wait` helper in `handcap/Utils/`. [UMI source](https://github.com/real-stanford/universal_manipulation_interface) · [MIT license](licenses/UMI.txt)
- **Diffusion Policy** by the Columbia Artificial Intelligence and Robotics Lab supplies the replay-buffer implementation in `handcap/Postprocess/dp_utils/replay_buffer.py`. [Diffusion Policy source](https://github.com/real-stanford/diffusion_policy) · [MIT license](licenses/Diffusion-Policy.txt)
- **pyopenxr examples** by Christopher Bruns supplies Vive Tracker example code adapted by the HandCap tracker and visualization scripts. [pyopenxr examples source](https://github.com/cmbruns/pyopenxr_examples) · [Apache 2.0 license](licenses/pyopenxr-examples.txt)
- **imagecodecs** by Christoph Gohlke supplies Numcodecs adapters. The original BSD 3-Clause notice appears in the vendored file. [imagecodecs source](https://github.com/cgohlke/imagecodecs) · [Vendored adapter and license](handcap/Postprocess/dp_utils/imagecodecs_numcodecs.py)

See the [dataset guide](data/README.md) for the dataset preview attribution.
