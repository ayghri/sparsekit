# Copyright (c) 2026 - Ayoub Ghriss & Contributors
# Licensed under CC BY-NC 4.0 (see LICENSE or https://creativecommons.org/licenses/by-nc/4.0/)
# Non-commercial use only; contact us for commercial licensing.
"""
Licensed under CC BY-NC 4.0 (see LICENSE or https://creativecommons.org/licenses/by-nc/4.0/)
Non-commercial use only; contact us for commercial licensing.
"""

from typing import Generator, Optional
from contextlib import contextmanager
import torch
from torch import nn
import time
import math

Tensor = torch.Tensor


def transfer_to_device(obj, device: Optional[torch.device]):
    """
    Recursively transfer tensors in nested structures to target device.

    Args:
        obj: Input object (tensor, list, dict, tuple, or other)
        device: Target device

    Returns:
        Object with same structure, tensors moved to device
    """
    if device is None:
        return obj
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    elif isinstance(obj, dict):
        return {k: transfer_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [transfer_to_device(v, device) for v in obj]
    elif isinstance(obj, tuple):
        # Preserve namedtuple types
        if hasattr(obj, "_fields"):
            return type(obj)(*[transfer_to_device(v, device) for v in obj])
        return tuple(transfer_to_device(v, device) for v in obj)
    else:
        # Non-tensor, non-container types (int, float, None, etc.)
        return obj


class TimeContainer:
    """
    A simple container to hold a single float value for time measurement.
    This is used to allow the context manager to yield a mutable object.
    """

    def __init__(self, name: str):
        self.name = name
        self.time = 0.0

    def set_time(self, value: float):
        self.time = value

    def get_time(self) -> float:
        return self.time

    def __repr__(self) -> str:
        if self.time < 1e-3:
            return f"{self.name} took: {self.time * 1e6:.2f}us"
        elif self.time < 1.0:
            return f"{self.name} took: {self.time * 1e3:.2f}ms"
        return f"{self.name} took: {self.time:.2f}s"


@contextmanager
def measure(
    name: str, verbose: bool = True
) -> Generator[TimeContainer, None, None]:
    """
    A context manager to measure execution time.

    It returns the time taken and can optionally print it to the console.

    Args:
        name (str): The description of the code group being measured.
        verbose (bool): If True (the default), prints the execution time.

    Yields:
        TimeContainer: A container object that will be updated with the
        measured time in seconds after the group completes.
    """
    start_time = time.perf_counter()
    time_container = TimeContainer(name)
    try:
        yield time_container
    finally:
        end_time = time.perf_counter()
        duration = end_time - start_time
        time_container.set_time(duration)
        if verbose:
            print(time_container)


@torch.no_grad()
def calculate_sparsity_per_layer(model):
    sparsity_dict = {}
    total_sparsity_num = 0
    total_sparsity_den = 0
    param_idx = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            numel = param.numel()
            # zero_params = (param.abs() < threshold).sum().item()
            zero_params = numel - (param.abs() > 0).sum().item()
            layer_sparsity = (zero_params / numel) * 100.0 if numel > 0 else 0.0
            key_name = f"L{param_idx}_{name}"
            status_tags = []
            if status_tags:
                key_name += f" [{', '.join(status_tags)}]"
            sparsity_dict[key_name] = layer_sparsity
            total_sparsity_num += zero_params
            total_sparsity_den += numel
            param_idx += 1
    overall_sparsity = (
        (total_sparsity_num / total_sparsity_den) * 100.0
        if total_sparsity_den > 0
        else 0.0
    )
    return overall_sparsity, sparsity_dict


@torch.no_grad()  # Ensure no gradients are computed within this function
def print_model_sparsity(model: nn.Module, threshold: float = 1e-8):
    """
    Calculates and prints the sparsity of a PyTorch model, including total parameters.

    Args:
        model (nn.Module): The PyTorch model to analyze.
        threshold (float): Absolute value threshold below which a parameter
                           is considered zero. Defaults to 1e-8.
    """
    print("-" * 90)  # Adjusted width for new column
    print(f"Sparsity Analysis (Threshold: {threshold:.1e})")
    print("-" * 90)
    print(
        f"{'Parameter Name':<45} {'Shape':<20} {'NNZ':<12} {'Total':<12} {'Sparsity (%)':<15}"
    )
    print("=" * 90)

    total_params_overall = 0  # Includes non-trainable if you iterate over all
    total_params_trainable = 0
    total_nnz_trainable = 0

    for name, param in model.named_parameters():
        # Track all parameters for total count
        numel = param.numel()
        total_params_overall += numel  # Count all params encountered

        # Only consider trainable parameters for sparsity stats
        if param.requires_grad:
            if numel == 0:  # Skip empty parameters if any
                print(
                    f"{name:<45} {str(tuple(param.shape)):<20} {'N/A':<12} {numel:<12,d} {'N/A':<15}"
                )
                continue

            # Calculate non-zero elements based on the threshold
            if hasattr(param, "dystil_k"):
                nnz = (param.abs() > threshold).sum().item()
            else:
                nnz = param.numel()

            sparsity = (1.0 - nnz / numel) * 100.0

            total_params_trainable += numel
            total_nnz_trainable += nnz

            print(
                f"{name:<45} {str(tuple(param.shape)):<20} {nnz:<12,d} {numel:<12,d} {sparsity:<15.2f}"
            )
        else:
            # Print info for non-trainable parameters
            print(
                f"{name:<45} {str(tuple(param.shape)):<20} {'N/A (NT)':<12} {numel:<12,d} {'N/A (NT)':<15}"
            )

    print("-" * 90)
    if total_params_trainable > 0:
        overall_sparsity = (
            1.0 - total_nnz_trainable / total_params_trainable
        ) * 100.0
        print(
            f"{'Overall Trainable':<45} {' ':<20} {total_nnz_trainable:<12,d} {total_params_trainable:<12,d} {overall_sparsity:<15.2f}"
        )
        # print(f"(Total Trainable Parameters: {total_params_trainable:,d})") # Included in summary line
    else:
        print("Model has no trainable parameters.")
    # Optionally print total overall parameters if different from trainable
    # if total_params_overall != total_params_trainable:
    #      print(f"(Total Overall Parameters: {total_params_overall:,d})")
    print("-" * 90)


# Helper function for soft-thresholding
@torch.no_grad()
def soft_threshold(x, thresh):
    thresh = torch.tensor(max(0.0, thresh), dtype=x.dtype, device=x.device)
    return torch.sign(x) * torch.relu(torch.abs(x) - thresh)


def ensure_true_ratio(mask: torch.Tensor, target_ratio: float) -> torch.Tensor:
    """
    Modifies a boolean mask by adding random True entries to ensure it has at least
    the target_ratio of True values. Existing True values are preserved.

    Args:
        mask (torch.Tensor): The input boolean or 0/1 tensor.
        target_ratio (float): The desired minimum ratio of True values (0.0 to 1.0).

    Returns:
        torch.Tensor: A new boolean tensor with the adjusted True values.
    """
    if not (0.0 <= target_ratio <= 1.0):
        raise ValueError("Target ratio s must be between 0.0 and 1.0.")

    # Work with a boolean copy to avoid modifying the original and for consistent ops
    output_mask = mask.clone().bool()
    num_elements = output_mask.numel()

    if num_elements == 0:
        if target_ratio == 0:
            return output_mask  # Empty mask correctly has 0% True
        else:
            # Cannot add True to an empty tensor to meet a non-zero ratio
            # Depending on strictness, could raise error or just return.
            # For now, returning as is, as it's the best we can do.
            print(
                f"Warning: Cannot achieve target_ratio {target_ratio} for an empty mask."
            )
            return output_mask

    # Calculate current and target number of True values
    current_true_count = (
        output_mask.sum().item()
    )  # .sum() on bool tensor counts True

    # We want to ensure *at least* s ratio, so use math.ceil
    target_true_count = math.ceil(target_ratio * num_elements)
    # Ensure target_true_count is int for calculations
    target_true_count = int(target_true_count)

    num_trues_to_add = target_true_count - current_true_count

    if num_trues_to_add <= 0:
        # Already meets or exceeds the target ratio
        return output_mask

    # Find indices of False values
    # (~output_mask) inverts the boolean mask (True -> False, False -> True)
    # .nonzero(as_tuple=True) returns a tuple of tensors, one for each dimension,
    # containing the coordinates of the True values in (~output_mask), i.e., False in output_mask.
    false_indices_tuple = (~output_mask).nonzero(as_tuple=True)

    # Number of available slots to turn from False to True
    # All tensors in false_indices_tuple have the same length.
    num_available_false_slots = 0
    if false_indices_tuple and false_indices_tuple[0].numel() > 0:
        num_available_false_slots = false_indices_tuple[0].numel()

    if num_available_false_slots == 0:
        # No False values to flip, cannot add more Trues
        # This case implies current_true_count < target_true_count but mask is all True,
        # which should only happen if target_ratio > 1 (already checked) or due to float precision.
        # Or if num_elements > 0 but the mask somehow has no False values yet doesn't meet the ratio.
        # Most likely, it means the mask is already all True.
        print(
            f"Warning: Mask is all True but target ratio {target_ratio} "
            f"(target count {target_true_count}) not met by current count {current_true_count}. "
            f"This might indicate an issue or an impossible target."
        )
        return output_mask

    # Determine how many False values to actually flip
    num_to_flip = min(num_trues_to_add, num_available_false_slots)

    if num_to_flip > 0:
        # Generate random permutation of indices for the available False slots
        # These are indices INTO the list of false_indices_tuple elements
        perm = torch.randperm(
            num_available_false_slots, device=output_mask.device
        )
        indices_to_select_from_false_list = perm[:num_to_flip]

        # Get the actual coordinates to flip by indexing into the false_indices_tuple
        # For example, if false_indices_tuple = (rows, cols), and we selected some indices `k`
        # from `perm`, then `rows[k]` and `cols[k]` give the coordinates to flip.
        coords_to_flip_list = []
        for dim_coords in false_indices_tuple:
            coords_to_flip_list.append(
                dim_coords[indices_to_select_from_false_list]
            )

        coords_to_flip_tuple = tuple(coords_to_flip_list)

        # Set these randomly selected False positions to True
        output_mask[coords_to_flip_tuple] = True

    # Verify (optional, for debugging)
    # final_true_ratio = output_mask.sum().item() / num_elements if num_elements > 0 else 0
    # print(f"Target ratio: {target_ratio:.4f}, Achieved ratio: {final_true_ratio:.4f}")
    # print(f"Target count: {target_true_count}, Achieved count: {output_mask.sum().item()}")

    return output_mask


def mask_post_accumulate_hook(param: Tensor, mask: Tensor) -> None:
    param.grad = param.grad * mask  # type: ignore


def mask_grad_hook(grad: Tensor, mask: Tensor) -> Tensor:
    return grad * mask


def get_input_hook(name: str, activations: dict, procees_fn=None):
    """Creates a forward hook to capture the input activations of a layer."""

    def hook(model, input, output):
        if name not in activations:
            activations[name] = []
        if procees_fn is not None:
            input = procees_fn(input)
        activations[name].append(input)

    return hook


def get_output_hook(name: str, activations: dict, device=None):
    """Creates a forward hook to capture the output activations of a layer."""

    def hook(model, input, output):
        if name not in activations:
            activations[name] = []
        if device is not None:
            output = output.to(device)
        activations[name].append(output)

    return hook


def get_forward_hook(
    name: str,
    activations: dict,
    input_process_fn=None,
    output_process_fn=None,
):
    """Creates a forward hook to capture the input and output activations of a layer."""

    def hook(model, input, output):
        # print(f"Hook triggered for layer: {model}")
        if name not in activations:
            activations[name] = {"input": [], "output": []}
        if input_process_fn is not None:
            input = input_process_fn(input)
        if output_process_fn is not None:
            output = output_process_fn(output)
        activations[name]["input"].append(input)
        activations[name]["output"].append(output)

    return hook


class InputCatcher(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.inputs = None

    def forward(self, **kwargs):
        self.inputs = kwargs
        return self.layer(**kwargs)


class LLMIOCatcher:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.hooks = {}
        self.inputs = {}
        self.outputs = {}

    @staticmethod
    def _layer_name(layer_idx):
        return f"layer_{layer_idx}"

    def attach_layer(self, layer_idx):
        name = f"layer_{layer_idx}"
        layer: nn.Module = self.model.model.layers[layer_idx]

        def input_hook(module, args, kwargs):
            self.inputs[name] = {"args": args, "kwargs": kwargs}

        def output_hook(module, args, output):
            if self.device is not None and isinstance(output, Tensor):
                self.outputs[name] = output.to(self.device)
            else:
                self.outputs[name] = output

        self.hooks[f"{name}_pre"] = layer.register_forward_pre_hook(
            input_hook, with_kwargs=True
        )
        self.hooks[f"{name}_post"] = layer.register_forward_hook(output_hook)

        # hook = get_output_hook(name, self.activations, device=self.device)
        # self.hooks[name] = layer.register_forward_hook(hook)
        # self.model.model.layers[layer_idx] = InputCatcher(layer)

    def detach_layer(self, layer_idx):
        name = self._layer_name(layer_idx)

        self.inputs.pop(name, None)
        self.outputs.pop(name, None)
        for suffix in ("_pre", "_post"):
            hook_name = f"{name}{suffix}"
            if hook_name in self.hooks:
                self.hooks[hook_name].remove()
                del self.hooks[hook_name]


class ModuleInputCatcher:
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device
        self.hooks = {}
        self.inputs = {}

    def attach(self, module: nn.Module, name: str, raise_error=False):
        assert name not in self.inputs
        assert name not in self.hooks
        self.inputs[name] = []

        def input_hook(module, args, kwargs):
            self.inputs[name].append(
                {
                    "args": transfer_to_device(args, self.device),
                    "kwargs": transfer_to_device(kwargs, self.device),
                }
            )
            if raise_error:
                raise RuntimeError("Reached target layer inputs")

        self.hooks[name] = module.register_forward_pre_hook(
            input_hook, with_kwargs=True
        )

    def detach(self, name):
        assert name in self.inputs
        assert name in self.hooks
        self.inputs.pop(name, None)
        self.hooks.pop(name).remove()


class ModuleOutputCatcher:
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device
        self.hooks = {}
        self.outputs = {}

    def attach(self, module: nn.Module, name: str, raise_error=False):
        assert name not in self.outputs
        assert name not in self.hooks
        self.outputs[name] = []

        def output_hook(module, args, output):
            self.outputs[name].append(transfer_to_device(output, self.device))
            if raise_error:
                raise RuntimeError("Reached target layer output")

        self.hooks[name] = module.register_forward_hook(output_hook)

    def detach(self, name):
        assert name in self.outputs
        assert name in self.hooks
        self.outputs.pop(name, None)
        self.hooks.pop(name).remove()
