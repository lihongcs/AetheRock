import dataclasses
import pathlib

import pytest

from openpi.training import config as training_config


@pytest.mark.parametrize("with_datasets", [False, True])
def test_forcevt_creates_training_and_serving_data_configs(tmp_path, monkeypatch, with_datasets):
    # Construct the actual ForceVT config without fetching the tokenizer weights.
    monkeypatch.setattr(training_config._tokenizer, "PaligemmaTokenizer", lambda _: object())
    config = training_config.get_config("pi05_512_close_bottle_health_distill_tactile_wrist_force")
    data_factory = config.data
    roots = ()
    labels = ()
    if with_datasets:
        labels = ("50", "75", "100")
        roots = tuple(str(tmp_path / f"health{label}") for label in labels)
        data_factory = dataclasses.replace(
            data_factory, health_data_roots=roots, health_labels=labels, health_sampling_weights=(1, 1, 5)
        )

    data = data_factory.create(tmp_path, config.model)

    assert data.asset_id == "512_close_bottle_health_distill"
    assert data.health_data_roots == roots
    assert data.health_labels == labels
    assert data.health_repo_ids == tuple(pathlib.Path(root).name for root in roots)
    assert data.action_sequence_keys == ("action", "observation.forces.left", "observation.forces.right")
