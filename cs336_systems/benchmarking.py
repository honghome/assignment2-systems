from __future__ import annotations

import argparse
import contextlib
import math
import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer

import pandas as pd
import torch

import cs336_basics.model as basics_model
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy, softmax


@dataclass(frozen=True)
class ModelSize:
	name: str
	d_model: int
	d_ff: int
	num_layers: int
	num_heads: int


MODEL_SIZES: dict[str, ModelSize] = {
	"small": ModelSize("small", d_model=768, d_ff=3072, num_layers=12, num_heads=12),
	"medium": ModelSize("medium", d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
	"large": ModelSize("large", d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
	"xl": ModelSize("xl", d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
	"10B": ModelSize("10B", d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}


@dataclass(frozen=True)
class BenchmarkConfig:
	model_size: ModelSize
	vocab_size: int = 10_000
	batch_size: int = 4
	context_length: int = 512
	warmup_steps: int = 5
	measurement_steps: int = 10
	mode: str = "train-step"
	device: str = "auto"
	dtype: str = "float32"
	lr: float = 1e-3
	weight_decay: float = 0.01
	seed: int = 0
	enable_nvtx: bool = False
	memory_snapshot_file: str | None = None
	memory_history_max_entries: int = 1_000_000


def resolve_device(device: str) -> torch.device:
	if device == "auto":
		return torch.device("cuda" if torch.cuda.is_available() else "cpu")
	resolved_device = torch.device(device)
	if resolved_device.type == "cuda" and not torch.cuda.is_available():
		raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
	return resolved_device


def resolve_dtype(dtype: str) -> torch.dtype:
	dtypes = {
		"float32": torch.float32,
		"float16": torch.float16,
		"bfloat16": torch.bfloat16,
	}
	return dtypes[dtype]


def synchronize(device: torch.device) -> None:
	if device.type == "cuda":
		torch.cuda.synchronize(device)


def nvtx_range(device: torch.device, name: str, enabled: bool):
	if enabled and device.type == "cuda":
		return torch.cuda.nvtx.range(name)
	return contextlib.nullcontext()


def maybe_autocast(device: torch.device, dtype: torch.dtype):
	if device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}:
		return torch.autocast(device_type="cuda", dtype=dtype)
	return contextlib.nullcontext()


def annotated_scaled_dot_product_attention(
	Q: torch.Tensor,
	K: torch.Tensor,
	V: torch.Tensor,
	mask: torch.Tensor | None = None,
) -> torch.Tensor:
	device = Q.device
	d_k = K.shape[-1]
	with nvtx_range(device, "attention_scores_matmul", True):
		attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

	if mask is not None:
		with nvtx_range(device, "attention_mask", True):
			attention_scores = torch.where(mask, attention_scores, float("-inf"))

	with nvtx_range(device, "attention_softmax", True):
		attention_weights = softmax(attention_scores, dim=-1)

	with nvtx_range(device, "attention_value_matmul", True):
		return torch.matmul(attention_weights, V)


def make_model(config: BenchmarkConfig, device: torch.device, dtype: torch.dtype) -> BasicsTransformerLM:
	if config.enable_nvtx:
		basics_model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

	size = config.model_size
	model = BasicsTransformerLM(
		vocab_size=config.vocab_size,
		context_length=config.context_length,
		d_model=size.d_model,
		num_layers=size.num_layers,
		num_heads=size.num_heads,
		d_ff=size.d_ff,
	)
	model = model.to(device=device)
	return model.train()


def make_batch(config: BenchmarkConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
	input_ids = torch.randint(
		low=0,
		high=config.vocab_size,
		size=(config.batch_size, config.context_length),
		device=device,
		dtype=torch.long,
	)
	target_ids = torch.randint(
		low=0,
		high=config.vocab_size,
		size=(config.batch_size, config.context_length),
		device=device,
		dtype=torch.long,
	)
	return input_ids, target_ids


def run_forward(
	model: BasicsTransformerLM,
	input_ids: torch.Tensor,
	target_ids: torch.Tensor,
	device: torch.device,
	dtype: torch.dtype,
	enable_nvtx: bool,
) -> torch.Tensor:
	del target_ids
	with torch.no_grad(), nvtx_range(device, "forward", enable_nvtx), maybe_autocast(device, dtype):
		return model(input_ids)


def run_forward_backward(
	model: BasicsTransformerLM,
	input_ids: torch.Tensor,
	target_ids: torch.Tensor,
	device: torch.device,
	dtype: torch.dtype,
	enable_nvtx: bool,
) -> torch.Tensor:
	with nvtx_range(device, "zero_grad", enable_nvtx):
		model.zero_grad(set_to_none=True)
	with nvtx_range(device, "forward", enable_nvtx), maybe_autocast(device, dtype):
		logits = model(input_ids)
	with nvtx_range(device, "loss", enable_nvtx), maybe_autocast(device, dtype):
		loss = cross_entropy(logits, target_ids)
	with nvtx_range(device, "backward", enable_nvtx):
		loss.backward()
	return loss.detach()


def run_train_step(
	model: BasicsTransformerLM,
	input_ids: torch.Tensor,
	target_ids: torch.Tensor,
	optimizer: torch.optim.Optimizer,
	device: torch.device,
	dtype: torch.dtype,
	enable_nvtx: bool,
) -> torch.Tensor:
	with nvtx_range(device, "zero_grad", enable_nvtx):
		optimizer.zero_grad(set_to_none=True)
	with nvtx_range(device, "forward", enable_nvtx), maybe_autocast(device, dtype):
		logits = model(input_ids)
	with nvtx_range(device, "loss", enable_nvtx), maybe_autocast(device, dtype):
		loss = cross_entropy(logits, target_ids)
	with nvtx_range(device, "backward", enable_nvtx):
		loss.backward()
	with nvtx_range(device, "optimizer_step", enable_nvtx):
		optimizer.step()
	return loss.detach()


def run_one_step(
	mode: str,
	model: BasicsTransformerLM,
	input_ids: torch.Tensor,
	target_ids: torch.Tensor,
	optimizer: torch.optim.Optimizer,
	device: torch.device,
	dtype: torch.dtype,
	enable_nvtx: bool,
) -> torch.Tensor:
	if mode == "forward":
		return run_forward(model, input_ids, target_ids, device, dtype, enable_nvtx)
	if mode == "forward-backward":
		return run_forward_backward(model, input_ids, target_ids, device, dtype, enable_nvtx)
	if mode == "train-step":
		return run_train_step(model, input_ids, target_ids, optimizer, device, dtype, enable_nvtx)
	raise ValueError(f"Unknown benchmark mode: {mode}")


def benchmark(config: BenchmarkConfig) -> dict[str, float | int | str]:
	torch.manual_seed(config.seed)
	device = resolve_device(config.device)
	dtype = resolve_dtype(config.dtype)
	if config.memory_snapshot_file is not None and device.type != "cuda":
		raise RuntimeError("Memory snapshots require a CUDA device.")
	model = make_model(config, device, dtype)
	optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
	input_ids, target_ids = make_batch(config, device)

	if device.type == "cuda":
		torch.cuda.reset_peak_memory_stats(device)

	with nvtx_range(device, "warmup", config.enable_nvtx):
		for _ in range(config.warmup_steps):
			run_one_step(config.mode, model, input_ids, target_ids, optimizer, device, dtype, config.enable_nvtx)
			synchronize(device)

	timings_s: list[float] = []
	try:
		if config.memory_snapshot_file is not None:
			torch.cuda.memory._record_memory_history(max_entries=config.memory_history_max_entries)

		with nvtx_range(device, "measurement", config.enable_nvtx):
			for _ in range(config.measurement_steps):
				start = timer()
				run_one_step(config.mode, model, input_ids, target_ids, optimizer, device, dtype, config.enable_nvtx)
				synchronize(device)
				timings_s.append(timer() - start)

		if config.memory_snapshot_file is not None:
			memory_snapshot_path = Path(config.memory_snapshot_file)
			memory_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
			torch.cuda.memory._dump_snapshot(str(memory_snapshot_path))
	finally:
		if config.memory_snapshot_file is not None:
			torch.cuda.memory._record_memory_history(enabled=None)

	peak_memory_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
	mean_s = statistics.mean(timings_s)
	std_s = statistics.stdev(timings_s) if len(timings_s) > 1 else 0.0

	return {
		"size": config.model_size.name,
		"mode": config.mode,
		"device": str(device),
		"dtype": config.dtype,
		"vocab_size": config.vocab_size,
		"batch_size": config.batch_size,
		"context_length": config.context_length,
		"warmup_steps": config.warmup_steps,
		"measurement_steps": config.measurement_steps,
		"d_model": config.model_size.d_model,
		"d_ff": config.model_size.d_ff,
		"num_layers": config.model_size.num_layers,
		"num_heads": config.model_size.num_heads,
		"parameters": sum(parameter.numel() for parameter in model.parameters()),
		"mean_s": mean_s,
		"std_s": std_s,
		"mean_ms": mean_s * 1_000,
		"std_ms": std_s * 1_000,
		"peak_memory_gib": peak_memory_bytes / 2**30,
	}


def expand_model_sizes(sizes: Iterable[str]) -> list[ModelSize]:
	selected: list[ModelSize] = []
	for size in sizes:
		if size == "all":
			selected.extend(MODEL_SIZES.values())
		else:
			selected.append(MODEL_SIZES[size])
	seen: set[str] = set()
	deduplicated = []
	for size in selected:
		if size.name not in seen:
			deduplicated.append(size)
			seen.add(size.name)
	return deduplicated


def build_dataframe(rows: list[dict[str, float | int | str]]) -> pd.DataFrame:
	columns = [
		"size",
		"mode",
		"device",
		"dtype",
		"batch_size",
		"context_length",
		"warmup_steps",
		"measurement_steps",
		"d_model",
		"d_ff",
		"num_layers",
		"num_heads",
		"parameters",
		"mean_ms",
		"std_ms",
		"peak_memory_gib",
	]
	dataframe = pd.DataFrame(rows)
	return dataframe[columns]


def format_cell(value: object) -> str:
	if isinstance(value, float):
		return f"{value:.3f}"
	return str(value)


def dataframe_to_markdown(dataframe: pd.DataFrame) -> str:
	headers = [str(column) for column in dataframe.columns]
	rows = [[format_cell(value) for value in row] for row in dataframe.itertuples(index=False, name=None)]
	widths = [len(header) for header in headers]
	for row in rows:
		widths = [max(width, len(value)) for width, value in zip(widths, row, strict=True)]

	header_line = "| " + " | ".join(header.ljust(width) for header, width in zip(headers, widths, strict=True)) + " |"
	divider_line = "| " + " | ".join("-" * width for width in widths) + " |"
	row_lines = ["| " + " | ".join(value.ljust(width) for value, width in zip(row, widths, strict=True)) + " |" for row in rows]
	return "\n".join([header_line, divider_line, *row_lines])


def dataframe_to_typst(dataframe: pd.DataFrame) -> str:
	rows = [[format_cell(value) for value in row] for row in dataframe.itertuples(index=False, name=None)]
	table_rows = [[*map(str, dataframe.columns)], *rows]
	formatted_rows = ["  " + ", ".join(f"[{value}]" for value in row) + "," for row in table_rows]
	return "#table(\n  columns: " + str(len(dataframe.columns)) + ",\n" + "\n".join(formatted_rows) + "\n)"


def format_output(dataframe: pd.DataFrame, output_format: str) -> str:
	if output_format == "csv":
		return dataframe.to_csv(index=False)
	if output_format == "latex":
		return dataframe.to_latex(index=False, float_format=lambda value: f"{value:.3f}")
	if output_format == "typst":
		return dataframe_to_typst(dataframe)
	return dataframe_to_markdown(dataframe)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Benchmark BasicsTransformerLM forward, backward, and optimizer-step runtimes.")
	parser.add_argument("--size", nargs="+", choices=[*MODEL_SIZES.keys(), "all"], default=["small"], help="Model size(s) to benchmark.")
	parser.add_argument("--mode", nargs="+", choices=["forward", "forward-backward", "train-step", "all"], default=["all"], help="Benchmark mode(s).")
	parser.add_argument("--batch-size", type=int, default=4)
	parser.add_argument("--context-length", type=int, default=512)
	parser.add_argument("--vocab-size", type=int, default=10_000)
	parser.add_argument("--warmup-steps", type=int, default=5)
	parser.add_argument("--measurement-steps", type=int, default=10)
	parser.add_argument("--device", default="auto", help="Device to use, e.g. auto, cuda, cuda:0, or cpu.")
	parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--weight-decay", type=float, default=0.01)
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--enable-nvtx", action="store_true", help="Emit NVTX ranges for profiler timelines.")
	parser.add_argument("--memory-snapshot-file", type=str, default=None, help="Optional GPU memory snapshot pickle; multi-case runs append model size and mode.")
	parser.add_argument("--memory-history-max-entries", type=int, default=1_000_000, help="Maximum entries recorded by PyTorch CUDA memory history.")
	parser.add_argument("--output-format", choices=["markdown", "csv", "latex", "typst"], default="markdown")
	parser.add_argument("--output-file", type=str, default=None, help="Optional file path to write the formatted table.")
	return parser.parse_args()


def expand_modes(modes: Iterable[str]) -> list[str]:
	if "all" in modes:
		return ["forward", "forward-backward", "train-step"]
	return list(dict.fromkeys(modes))


def main() -> None:
	args = parse_args()
	if args.measurement_steps < 1:
		raise ValueError("--measurement-steps must be at least 1.")
	if args.warmup_steps < 0:
		raise ValueError("--warmup-steps must be non-negative.")
	if args.memory_history_max_entries < 1:
		raise ValueError("--memory-history-max-entries must be at least 1.")

	model_sizes = expand_model_sizes(args.size)
	modes = expand_modes(args.mode)
	rows = []
	for model_size in model_sizes:
		for mode in modes:
			memory_snapshot_file = args.memory_snapshot_file
			if memory_snapshot_file is not None and len(model_sizes) * len(modes) > 1:
				snapshot_path = Path(memory_snapshot_file)
				memory_snapshot_file = str(snapshot_path.with_name(f"{snapshot_path.stem}_{model_size.name}_{mode}{snapshot_path.suffix}"))
			config = BenchmarkConfig(
				model_size=model_size,
				vocab_size=args.vocab_size,
				batch_size=args.batch_size,
				context_length=args.context_length,
				warmup_steps=args.warmup_steps,
				measurement_steps=args.measurement_steps,
				mode=mode,
				device=args.device,
				dtype=args.dtype,
				lr=args.lr,
				weight_decay=args.weight_decay,
				seed=args.seed,
				enable_nvtx=args.enable_nvtx,
				memory_snapshot_file=memory_snapshot_file,
				memory_history_max_entries=args.memory_history_max_entries,
			)
			print(f"Running {model_size.name} / {mode} on {resolve_device(args.device)}...", flush=True)
			rows.append(benchmark(config))

	dataframe = build_dataframe(rows)
	formatted_output = format_output(dataframe, args.output_format)
	print(formatted_output)
	if args.output_file is not None:
		with open(args.output_file, "w", encoding="utf-8") as output_file:
			output_file.write(formatted_output)
			if not formatted_output.endswith("\n"):
				output_file.write("\n")


if __name__ == "__main__":
	main()
