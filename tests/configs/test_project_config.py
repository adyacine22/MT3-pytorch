from __future__ import annotations

from configs import config_path, load_project_config


def test_project_config_loads_expected_sections():
    cfg = load_project_config()
    assert "paths" in cfg
    assert "audio" in cfg
    assert "symbolic" in cfg

    dataset_paths = cfg["paths"]["datasets"]
    for key in ("maestro_root", "slakh_root", "unified_index"):
        assert key in dataset_paths
        assert isinstance(dataset_paths[key], str)

    cache_paths = cfg["paths"]["cache"]
    assert "chunk_manifest" in cache_paths

    audio_io = cfg["audio"]["io"]
    assert isinstance(audio_io["sample_rate"], int)
    assert isinstance(audio_io["convert_to_mono"], bool)

    codec_cfg = cfg["symbolic"]["codec"]
    assert "steps_per_second" in codec_cfg


def test_project_config_path_points_to_file():
    cfg_path = config_path()
    assert cfg_path.exists()
    assert cfg_path.suffix == ".py"
    assert cfg_path.is_file()
