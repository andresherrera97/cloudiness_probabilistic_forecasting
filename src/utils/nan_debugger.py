import torch
import logging
from typing import Optional, Dict
from collections import defaultdict


class NaNDebugger:
    """
    A utility class to debug NaN values during neural network training.
    Tracks gradients, activations, and parameter statistics to help identify
    the source of NaN values.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        log_freq: int = 100,
        gradient_clip_threshold: float = 1000.0,
    ):
        self.model = model
        self.log_freq = log_freq
        self.gradient_clip_threshold = gradient_clip_threshold
        self.step = 0
        self.nan_locations = defaultdict(list)

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("NaNDebugger")

        # Register hooks
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on all layers"""
        for name, module in self.model.named_modules():
            # Forward hook to check activations
            self.handles.append(
                module.register_forward_hook(
                    lambda mod, inp, out, name=name: self._activation_hook(
                        mod, inp, out, name
                    )
                )
            )
            # Backward hook to check gradients
            self.handles.append(
                module.register_full_backward_hook(
                    lambda mod, grad_in, grad_out, name=name: self._gradient_hook(
                        mod, grad_in, grad_out, name
                    )
                )
            )

    def _activation_hook(self, module, inp, out, name):
        """Check for NaN values in forward pass"""
        if self.step % self.log_freq == 0:
            if isinstance(out, tuple):
                out = out[0]

            if torch.isnan(out).any():
                self.nan_locations[name].append(
                    {
                        "step": self.step,
                        "type": "activation",
                        "stats": self._compute_tensor_stats(out),
                    }
                )
                self.logger.warning(f"NaN detected in activation of layer {name}")

    def _gradient_hook(self, module, grad_input, grad_output, name):
        """Check for NaN values and large gradients in backward pass"""
        if self.step % self.log_freq == 0:
            # Check gradients
            for idx, grad in enumerate(grad_input):
                if grad is not None:
                    if torch.isnan(grad).any():
                        self.nan_locations[name].append(
                            {
                                "step": self.step,
                                "type": "gradient_input",
                                "gradient_idx": idx,
                                "stats": self._compute_tensor_stats(grad),
                            }
                        )
                        self.logger.warning(
                            f"NaN detected in gradient input of layer {name}"
                        )

                    # Check for exploding gradients
                    max_grad = torch.abs(grad).max().item()
                    if max_grad > self.gradient_clip_threshold:
                        self.logger.warning(
                            f"Large gradient detected in layer {name}: {max_grad:.2f}"
                        )

    @staticmethod
    def _compute_tensor_stats(tensor: torch.Tensor) -> Dict:
        """Compute statistics for a tensor"""
        with torch.no_grad():
            non_nan_mask = ~torch.isnan(tensor)
            non_nan_tensor = tensor[non_nan_mask]

            if len(non_nan_tensor) == 0:
                return {"all_nan": True, "nan_percentage": 100.0}

            return {
                "all_nan": False,
                "nan_percentage": (torch.isnan(tensor).float().mean() * 100).item(),
                "min": non_nan_tensor.min().item(),
                "max": non_nan_tensor.max().item(),
                "mean": non_nan_tensor.mean().item(),
                "std": non_nan_tensor.std().item() if len(non_nan_tensor) > 1 else 0.0,
            }

    def check_parameters(self) -> Dict[str, Dict]:
        """Check model parameters for NaN values and compute statistics"""
        param_stats = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_stats[name] = {
                    "param_stats": self._compute_tensor_stats(param.data),
                    "grad_stats": self._compute_tensor_stats(param.grad),
                }
        return param_stats

    def step_callback(self, loss: Optional[torch.Tensor] = None):
        """Call this after each training step"""
        self.step += 1

        if self.step % self.log_freq == 0:
            # Check loss
            if loss is not None and torch.isnan(loss).any():
                self.logger.warning(f"NaN detected in loss at step {self.step}")

            # Check parameters periodically
            param_stats = self.check_parameters()
            for name, stats in param_stats.items():
                if stats["param_stats"].get("all_nan", False) or stats[
                    "grad_stats"
                ].get("all_nan", False):
                    self.logger.warning(
                        f"NaN detected in parameters or gradients of {name}"
                    )

    def get_nan_report(self) -> Dict:
        """Generate a complete report of all NaN occurrences"""
        return {
            "nan_locations": dict(self.nan_locations),
            "total_steps": self.step,
            "parameter_stats": self.check_parameters(),
        }

    def close(self):
        """Remove all hooks"""
        for handle in self.handles:
            handle.remove()
