from __future__ import annotations

import argparse
import contextlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyModel(nn.Module):
	def __init__(self, in_features: int, out_features: int):
		super().__init__()
		# First fully connected layer: maps each input vector from 16 features to 10 features.
		self.fc1 = nn.Linear(in_features, 10, bias=False)
		# LayerNorm normalizes the 10-dimensional hidden activation for each example.
		self.ln = nn.LayerNorm(10)
		# Second fully connected layer: maps the 10 hidden features to the output size.
		self.fc2 = nn.Linear(10, out_features, bias=False)
		# ReLU keeps positive values and clamps negative values to zero.
		self.relu = nn.ReLU()

	def forward(self, x):
		# Apply fc1 first, then ReLU nonlinearity.
		x = self.relu(self.fc1(x))
		# Apply LayerNorm after the first linear layer and ReLU.
		x = self.ln(x)
		# Apply fc2 to produce the final model output, also called logits here.
		x = self.fc2(x)
		return x


def resolve_dtype(dtype: str) -> torch.dtype:
	# Convert the command-line string into the PyTorch dtype object used by autocast.
	return {
		"float16": torch.float16,
		"bfloat16": torch.bfloat16,
		"float32": torch.float32,
	}[dtype]


def maybe_autocast(device: torch.device, dtype: torch.dtype):
	# Autocast only matters on CUDA when we request a lower-precision compute dtype.
	if device.type == "cuda" and dtype != torch.float32:
		return torch.autocast(device_type="cuda", dtype=dtype)
	# On CPU, or for float32, use a do-nothing context manager so the code path is identical.
	return contextlib.nullcontext()


def main() -> None:
	# Define command-line flags so the same script can run locally or on a GPU job.
	parser = argparse.ArgumentParser(description="Inspect ToyModel dtypes under autocast")
	parser.add_argument("--device", default="cuda:0")
	parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
	parser.add_argument("--output-file", default="outputs/mixed_precision_toy_dtypes.md")
	args = parser.parse_args()

	# Build the torch.device and torch.dtype from the parsed command-line arguments.
	device = torch.device(args.device)
	dtype = resolve_dtype(args.dtype)
	# Fix the random seed so repeated runs use the same random inputs and targets.
	torch.manual_seed(0)

	# Move the model and synthetic data to the requested device.
	model = ToyModel(in_features=16, out_features=4).to(device)
	# A batch of 8 fake examples, each with 16 input features.
	x = torch.randn(8, 16, device=device)
	# A fake regression target with the same shape as the model output.
	target = torch.randn(8, 4, device=device)
	# This dictionary will store the dtype produced by each layer we inspect.
	activations: dict[str, torch.dtype] = {}

	def save_dtype(name: str):
		# A forward hook runs automatically when its layer executes during model(x).
		def hook(_module, _inputs, output):
			# Save the dtype of this layer's output tensor.
			activations[name] = output.dtype

		return hook

	# Register hooks on the layers requested by the assignment.
	model.fc1.register_forward_hook(save_dtype("fc1_output"))
	model.ln.register_forward_hook(save_dtype("layer_norm_output"))
	model.fc2.register_forward_hook(save_dtype("fc2_output"))

	# Run the forward pass and loss computation inside autocast when enabled.
	with maybe_autocast(device, dtype):
		logits = model(x)
		loss = F.mse_loss(logits, target)

	# Backpropagate once so we can also inspect gradient dtypes.
	loss.backward()

	# Collect all parameter dtypes. Usually these stay float32 under autocast.
	parameter_dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
	# Collect all gradient dtypes after backward. These usually match the parameters.
	gradient_dtypes = sorted({str(parameter.grad.dtype) for parameter in model.parameters() if parameter.grad is not None})

	# Format the results as Markdown so the output can be pasted directly into the report.
	lines = [
		"# ToyModel Mixed Precision Dtypes",
		"",
		f"Device: `{device}`",
		f"Autocast dtype: `{dtype}`",
		"",
		"| Component | dtype |",
		"| --- | --- |",
		f"| Model parameters | `{', '.join(parameter_dtypes)}` |",
		f"| `ToyModel.fc1` output | `{activations['fc1_output']}` |",
		f"| `ToyModel.ln` output | `{activations['layer_norm_output']}` |",
		f"| `ToyModel.fc2` output | `{activations['fc2_output']}` |",
		f"| Loss | `{loss.dtype}` |",
		f"| Model gradients | `{', '.join(gradient_dtypes)}` |",
	]
	output = "\n".join(lines) + "\n"
	# Print the Markdown table to the terminal log.
	print(output)

	# Also save the Markdown table to a file artifact for remote jobs.
	output_path = Path(args.output_file)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(output, encoding="utf-8")


if __name__ == "__main__":
	main()