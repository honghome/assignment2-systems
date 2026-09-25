from pathlib import Path
import sys

import pytest

from cs336_systems import benchmarking


@pytest.mark.parametrize(
    ("sizes", "modes", "snapshot_file", "expected_names"),
    [
        (["small"], ["forward"], "snapshot.pickle", ["snapshot.pickle"]),
        (["small"], ["all"], "snapshot.pickle", ["snapshot_small_forward.pickle", "snapshot_small_forward-backward.pickle", "snapshot_small_train-step.pickle"]),
        (["small", "medium"], ["forward"], "snapshot.pickle", ["snapshot_small_forward.pickle", "snapshot_medium_forward.pickle"]),
        (["small", "small"], ["forward", "forward"], "snapshot.pickle", ["snapshot.pickle"]),
        (["small"], ["all"], None, [None, None, None]),
    ],
)
def test_snapshot_paths(monkeypatch, capsys, tmp_path, sizes, modes, snapshot_file, expected_names):
    arguments = ["benchmarking", "--device", "cpu", "--size", *sizes, "--mode", *modes]
    if snapshot_file is not None:
        arguments.extend(["--memory-snapshot-file", str(tmp_path / snapshot_file)])
    configs = []

    def record_benchmark(config):
        configs.append(config)
        return {}

    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setattr(benchmarking, "benchmark", record_benchmark)
    monkeypatch.setattr(benchmarking, "build_dataframe", lambda rows: rows)
    monkeypatch.setattr(benchmarking, "format_output", lambda rows, output_format: "")

    benchmarking.main()

    actual_paths = [Path(config.memory_snapshot_file) if config.memory_snapshot_file is not None else None for config in configs]
    expected_paths = [tmp_path / name if name is not None else None for name in expected_names]
    assert actual_paths == expected_paths