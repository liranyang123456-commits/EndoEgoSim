import json

from endosim.config import GenConfig
from scripts.finetune_vggt import SimWindowDataset


def _sequence(root, name, k1):
    directory = root / name
    directory.mkdir()
    (directory / "meta.json").write_text(
        json.dumps(
            {
                "n_frames": 16,
                "motion_type": "insertion",
                "reference_fraction": [0.8] * 15 + [None],
                "appearance": {"barrel_k1": k1},
            }
        ),
        encoding="utf-8",
    )
    return str(directory)


def test_barrel_distortion_is_disabled_by_default():
    assert GenConfig().appearance.barrel_prob == 0.0


def test_training_dataset_excludes_legacy_barrel_sequences(tmp_path):
    clean = _sequence(tmp_path, "seq_00000001", 0.0)
    barrel = _sequence(tmp_path, "seq_00000002", -0.08)

    dataset = SimWindowDataset(
        root=str(tmp_path),
        barrel_policy="exclude",
        hard_frac=0.0,
    )

    assert dataset.seq_dirs == [clean]
    assert dataset.barrel_dirs == [barrel]


def test_legacy_policy_retains_barrel_sequences(tmp_path):
    _sequence(tmp_path, "seq_00000001", 0.0)
    _sequence(tmp_path, "seq_00000002", -0.08)

    dataset = SimWindowDataset(
        root=str(tmp_path),
        barrel_policy="legacy",
        hard_frac=0.0,
    )

    assert len(dataset.seq_dirs) == 2
    assert len(dataset.barrel_dirs) == 1
