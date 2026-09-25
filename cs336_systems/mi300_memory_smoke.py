from __future__ import annotations

import argparse
from pathlib import Path

import torch


def format_bool(value: bool) -> str:
	return "yes" if value else "no"


def main() -> None:
	parser = argparse.ArgumentParser(description="Smoke-test PyTorch memory snapshot support on a GPU node.")
	parser.add_argument("--device", default="cuda:0")
	parser.add_argument("--output-file", default="outputs/mi300_memory_smoke.md")
	parser.add_argument("--snapshot-file", default="outputs/mi300_memory_smoke_snapshot.pickle")
	args = parser.parse_args()

	output_path = Path(args.output_file)
	snapshot_path = Path(args.snapshot_file)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	snapshot_path.parent.mkdir(parents=True, exist_ok=True)

	cuda_available = torch.cuda.is_available()
	has_record_memory_history = hasattr(torch.cuda.memory, "_record_memory_history")
	has_dump_snapshot = hasattr(torch.cuda.memory, "_dump_snapshot")
	snapshot_status = "not attempted"
	device_name = "unavailable"
	allocated_gib = 0.0
	reserved_gib = 0.0

	try:
		if cuda_available:
			device = torch.device(args.device)
			device_name = torch.cuda.get_device_name(device)
			if has_record_memory_history and has_dump_snapshot:
				torch.cuda.memory._record_memory_history(max_entries=100_000)
				try:
					# Allocate and use a small tensor so the snapshot contains at least one real allocation.
					x = torch.randn(1024, 1024, device=device)
					y = x @ x.T
					torch.cuda.synchronize(device)
					allocated_gib = torch.cuda.memory_allocated(device) / 2**30
					reserved_gib = torch.cuda.memory_reserved(device) / 2**30
					torch.cuda.memory._dump_snapshot(str(snapshot_path))
					snapshot_status = f"wrote `{snapshot_path}`"
					del x, y
				finally:
					torch.cuda.memory._record_memory_history(enabled=None)
			else:
				snapshot_status = "missing PyTorch memory history APIs"
		else:
			snapshot_status = "CUDA/ROCm device unavailable to PyTorch"
	except Exception as exc:
		snapshot_status = f"failed: `{type(exc).__name__}: {exc}`"

	lines = [
		"# MI300 PyTorch Memory Snapshot Smoke Test",
		"",
		f"PyTorch version: `{torch.__version__}`",
		f"torch.version.cuda: `{torch.version.cuda}`",
		f"torch.version.hip: `{torch.version.hip}`",
		f"torch.cuda.is_available(): `{cuda_available}`",
		f"Device requested: `{args.device}`",
		f"Device name: `{device_name}`",
		f"Has `_record_memory_history`: `{format_bool(has_record_memory_history)}`",
		f"Has `_dump_snapshot`: `{format_bool(has_dump_snapshot)}`",
		f"Allocated after smoke op: `{allocated_gib:.3f} GiB`",
		f"Reserved after smoke op: `{reserved_gib:.3f} GiB`",
		f"Snapshot status: {snapshot_status}",
	]
	output = "\n".join(lines) + "\n"
	print(output)
	output_path.write_text(output, encoding="utf-8")


if __name__ == "__main__":
	main()