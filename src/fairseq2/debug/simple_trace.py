"""
Simple tracing utilities for debugging non-determinism.

This module provides minimal, high-level tracing functions that can be easily
inserted into existing training code with just a few lines of changes.
"""

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor


class SimpleTracer:
    """Simple tracer that captures tensors at specific points."""
    
    def __init__(self, run_name: str):
        
        self.output_dir = Path(os.environ.get("TRACE_DIR"))
        self.run_name = Path(run_name)
        self.run_dir = self.output_dir / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.run_dir / "step_data").mkdir(exist_ok=True)
        (self.run_dir / "weights").mkdir(exist_ok=True)
        (self.run_dir / "activations").mkdir(exist_ok=True)
        (self.run_dir / "gradients").mkdir(exist_ok=True)
        (self.run_dir / "metadata").mkdir(exist_ok=True)
        
        self.current_step = 0
        self.trace_log = []
        
    def set_step(self, step: int):
        """Set the current training step."""
        self.current_step = step
        
    def trace_tensor(self, tensor: Tensor, name: str, category: str = "activations"):
        """Trace a single tensor."""
        if not isinstance(tensor, Tensor):
            return
        
        # Skip empty or meaningless tensors
        if not self._is_tensor_meaningful(tensor):
            return
            
        # Create filename
        filename = f"step_{self.current_step:06d}_{name}.pt"
        filepath = self.run_dir / category / filename
        
        # Save tensor
        torch.save(tensor.detach().cpu(), filepath)
        
        # Log metadata
        metadata = {
            "step": self.current_step,
            "name": name,
            "category": category,
            "filename": filename,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "top-10": [float(x.item()) for x in tensor.flatten()[:10].detach().cpu()],
            "checksum": hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest(),
        }
        
        self.trace_log.append(metadata)
        
        metadata_file = self.run_dir / "metadata" / f"step_{self.current_step:06d}_{name}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _is_tensor_meaningful(self, tensor: Tensor) -> bool:
        """Check if tensor has meaningful data worth saving."""
        # Skip tensors with no elements
        if tensor.numel() == 0:
            return False
        
        # Skip tensors with any zero dimension
        if any(dim == 0 for dim in tensor.shape):
            return False
        
        # Skip tensors that are all zeros (likely uninitialized FSDP shards)
        # Use a small threshold to account for numerical precision
        if tensor.abs().max().item() < 1e-12:
            return False
        
        # Skip tensors that are all NaN or Inf
        if not torch.isfinite(tensor).all():
            return False
            
        return True
            
    def _normalize_param_name(self, name: str) -> str:
        """Normalize parameter names for proper file ordering."""
        import re
        # Find layer numbers and pad them with zeros
        # This fixes ordering: layer.9 -> layer.009, layer.99 -> layer.099
        name = re.sub(r'\.layers\.(\d+)\.', lambda m: f'.layers.{int(m.group(1)):03d}.', name)
        return name.replace(".", "_")
    
    def trace_model_weights(self, model: nn.Module, prefix: str = ""):
        for name, param in model.named_parameters():
            if param.requires_grad:
                normalized_name = self._normalize_param_name(name)
                weight_name = f"{prefix}_{normalized_name}" if prefix else normalized_name
                self.trace_tensor(param.data, weight_name, "weights")
                
    def trace_model_gradients(self, model: nn.Module, prefix: str = ""):
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                normalized_name = self._normalize_param_name(name)
                grad_name = f"{prefix}_{normalized_name}_grad" if prefix else f"{normalized_name}_grad"
                self.trace_tensor(param.grad, grad_name, "gradients")
                
    def save_log(self):
        log_file = self.run_dir / "trace_log.json"
        
        with open(log_file, 'w') as f:
            json.dump(self.trace_log, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        
        print(f"Trace log successfully saved to {log_file}")
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save_log()


def compare_runs(run1_dir: str, run2_dir: str, tolerance: float = 1e-6) -> Dict[str, Any]:
    """Compare two runs and return detailed results."""
    run1_path = Path(run1_dir)
    run2_path = Path(run2_dir)
    
    # Scan artifact directories instead of using trace_log.json
    categories = ["weights", "gradients", "activations", "step_data"]
    
    # Collect all artifacts from both runs
    artifacts1 = {}
    artifacts2 = {}
    
    for category in categories:
        cat_dir1 = run1_path / category
        cat_dir2 = run2_path / category
        
        if cat_dir1.exists():
            for file_path in cat_dir1.glob("*.pt"):
                key = f"{category}_{file_path.name}"
                artifacts1[key] = {
                    "category": category,
                    "filename": file_path.name,
                    "filepath": file_path
                }
                
        if cat_dir2.exists():
            for file_path in cat_dir2.glob("*.pt"):
                key = f"{category}_{file_path.name}"
                artifacts2[key] = {
                    "category": category,
                    "filename": file_path.name,
                    "filepath": file_path
                }
    
    # Find common and missing artifacts
    common_keys = set(artifacts1.keys()) & set(artifacts2.keys())
    missing_keys = set(artifacts1.keys()) ^ set(artifacts2.keys())
    
    results = {
        "total_traces": len(common_keys),
        "missing_traces": len(missing_keys),
        "missing_in_run1": [k for k in missing_keys if k not in artifacts1],
        "missing_in_run2": [k for k in missing_keys if k not in artifacts2],
        "comparisons": [],
        "summary": {},
    }
    
    # Compare each common artifact
    for key in sorted(common_keys):
        artifact1 = artifacts1[key]
        artifact2 = artifacts2[key]
        
        # Load tensors
        tensor1 = torch.load(artifact1['filepath'])
        tensor2 = torch.load(artifact2['filepath'])
        
        # Extract step and name from filename (assuming format: step_XXXXXX_name.pt)
        filename = artifact1['filename']
        name_part = filename.replace('.pt', '')
        step_str = "unknown"
        name = name_part
        if name_part.startswith('step_'):
            parts = name_part.split('_', 2)
            if len(parts) >= 2:
                step_str = parts[1]
                name = '_'.join(parts[2:]) if len(parts) > 2 else parts[1]
        
        # Compare
        comparison = {
            "key": key,
            "step": step_str,
            "name": name,
            "category": artifact1['category'],
            "shape1": tensor1.shape,
            "shape2": tensor2.shape,
            "shape_match": list(tensor1.shape) == list(tensor2.shape),
            "dtype1": tensor1.dtype,
            "dtype2": tensor2.dtype,
            "dtype_match": str(tensor1.dtype) == str(tensor2.dtype),
        }
        
        if comparison["shape_match"] and comparison["dtype_match"]:
            diff = torch.abs(tensor1 - tensor2)
            max_diff = float(torch.max(diff).item()) if diff.numel() > 0 else 0.0
            
            # Find the index of maximum difference
            max_diff_index = None
            max_diff_value1 = None
            max_diff_value2 = None
            if diff.numel() > 0 and max_diff > 0:
                flat_diff = diff.flatten()
                max_diff_flat_idx = torch.argmax(flat_diff).item()
                max_diff_index = torch.unravel_index(torch.tensor(max_diff_flat_idx), tensor1.shape)
                max_diff_value1 = float(tensor1.flatten()[max_diff_flat_idx].item())
                max_diff_value2 = float(tensor2.flatten()[max_diff_flat_idx].item())
            
            # Check for zeros
            tensor1_max = float(torch.max(torch.abs(tensor1)).item()) if tensor1.numel() > 0 else 0.0
            tensor2_max = float(torch.max(torch.abs(tensor2)).item()) if tensor2.numel() > 0 else 0.0
            tensor1_is_zero = tensor1_max < 1e-12
            tensor2_is_zero = tensor2_max < 1e-12
            
            # Store first 10 values for inspection
            flat1 = tensor1.flatten()[:10]
            flat2 = tensor2.flatten()[:10]
            tensor1_values = [float(v.item()) for v in flat1]
            tensor2_values = [float(v.item()) for v in flat2]
            
            comparison.update({
                "max_diff": max_diff,
                "max_diff_index": max_diff_index,
                "max_diff_value1": max_diff_value1,
                "max_diff_value2": max_diff_value2,
                "mean_diff": float(torch.mean(diff).item()) if diff.numel() > 0 else 0.0,
                "std_diff": float(torch.std(diff).item()) if diff.numel() > 0 else 0.0,
                "identical": max_diff < tolerance,
                "checksum_match": torch.equal(tensor1, tensor2),
                "tensor1_is_zero": tensor1_is_zero,
                "tensor2_is_zero": tensor2_is_zero,
                "tensor1_max_abs": tensor1_max,
                "tensor2_max_abs": tensor2_max,
                "tensor1_values": tensor1_values,
                "tensor2_values": tensor2_values,
            })
        else:
            comparison.update({
                "max_diff": float('inf'),
                "max_diff_index": None,
                "max_diff_value1": None,
                "max_diff_value2": None,
                "mean_diff": float('inf'),
                "std_diff": float('inf'),
                "identical": False,
                "checksum_match": False,
                "tensor1_is_zero": True,
                "tensor2_is_zero": True,
                "tensor1_max_abs": 0.0,
                "tensor2_max_abs": 0.0,
                "tensor1_values": [],
                "tensor2_values": [],
            })
            
        results["comparisons"].append(comparison)
        
    # Generate summary
    total_identical = sum(1 for c in results["comparisons"] if c["identical"])
    total_different = len(results["comparisons"]) - total_identical
    
    results["summary"] = {
        "total_compared": len(results["comparisons"]),
        "identical": total_identical,
        "different": total_different,
        "identity_rate": total_identical / len(results["comparisons"]) if results["comparisons"] else 0.0,
        "max_difference": max(c["max_diff"] for c in results["comparisons"]) if results["comparisons"] else 0.0,
    }
    
    # Group by category
    by_category = {}
    for comp in results["comparisons"]:
        cat = comp["category"]
        if cat not in by_category:
            by_category[cat] = {"total": 0, "identical": 0, "different": 0}
        by_category[cat]["total"] += 1
        if comp["identical"]:
            by_category[cat]["identical"] += 1
        else:
            by_category[cat]["different"] += 1
            
    results["by_category"] = by_category
    
    return results


def print_comparison_summary(results: Dict[str, Any]):
    """Print a human-readable summary of comparison results."""
    print("=" * 60)
    print("TRACE COMPARISON SUMMARY")
    print("=" * 60)
    
    summary = results["summary"]
    print(f"Total traces compared: {summary['total_compared']}")
    print(f"Identical: {summary['identical']}")
    print(f"Different: {summary['different']}")
    print(f"Identity rate: {summary['identity_rate']:.2%}")
    print(f"Max difference: {summary['max_difference']:.2e}")
    
    if results["missing_traces"] > 0:
        print(f"\n⚠️  Missing traces: {results['missing_traces']}")
        if results["missing_in_run1"]:
            print(f"   Missing in run 1: {len(results['missing_in_run1'])}")
        if results["missing_in_run2"]:
            print(f"   Missing in run 2: {len(results['missing_in_run2'])}")
    
    print("\nBy Category:")
    for category, stats in results["by_category"].items():
        print(f"  {category.upper()}:")
        print(f"    Total: {stats['total']}")
        print(f"    Identical: {stats['identical']}")
        print(f"    Different: {stats['different']}")
        print(f"    Identity rate: {stats['identical']/stats['total']:.2%}")
    
    # Organize by prefix (pre_step, post_step, etc.) and component
    print(f"\nDETAILED BREAKDOWN BY PREFIX AND COMPONENT:")
    print("=" * 80)
    
    # Group comparisons by prefix and component
    prefix_groups = {}
    for comp in results["comparisons"]:
        name = comp["name"]
        
        # Extract prefix (pre_step, post_step, etc.)
        prefix = "unknown"
        if name.startswith("pre_step"):
            prefix = "pre_step"
        elif name.startswith("post_step"):
            prefix = "post_step"
        elif name.startswith("post_bwd"):
            prefix = "post_bwd"
        elif name.startswith("post_grad_norm"):
            prefix = "post_grad_norm"
        else:
            # Try to find other prefixes
            parts = name.split("_", 2)
            if len(parts) >= 2:
                prefix = f"{parts[0]}_{parts[1]}"
        
        # Extract component type
        component = "other"
        if "encoder_layers_" in name:
            # Extract layer number
            import re
            layer_match = re.search(r'encoder_layers_(\d+)_', name)
            if layer_match:
                layer_num = int(layer_match.group(1))
                component = f"layer_{layer_num:03d}"
        elif "encoder_frontend" in name or "frontend" in name:
            component = "frontend"
        elif "final_proj" in name or "classifier" in name or "head" in name:
            component = "final_proj"
        elif "loss" in name:
            component = "loss"
        
        if prefix not in prefix_groups:
            prefix_groups[prefix] = {}
        if component not in prefix_groups[prefix]:
            prefix_groups[prefix][component] = []
        prefix_groups[prefix][component].append(comp)
    
    # Print organized results
    for prefix in sorted(prefix_groups.keys()):
        print(f"\n{'='*60}")
        print(f"PREFIX: {prefix.upper()}")
        print(f"{'='*60}")
        
        for component in sorted(prefix_groups[prefix].keys()):
            comps = prefix_groups[prefix][component]
            identical_count = sum(1 for c in comps if c.get("identical", False))
            total_count = len(comps)
            
            # Find component with max difference and get context
            max_diff = 0.0
            max_diff_context = ""
            max_diff_tensor_name = ""
            for c in comps:
                c_diff = c.get("max_diff", 0)
                if c_diff > max_diff:
                    max_diff = c_diff
                    max_diff_tensor_name = c["name"]
                    # Get the actual location of max difference
                    if c.get("max_diff_index") is not None and c.get("max_diff_value1") is not None:
                        index = c["max_diff_index"]
                        val1 = c["max_diff_value1"]
                        val2 = c["max_diff_value2"]
                        if isinstance(index, tuple):
                            index_str = str(index)
                        else:
                            index_str = str(index)
                        max_diff_context = f" (tensor:{max_diff_tensor_name} index:{index_str} {val1:.6e} vs {val2:.6e})"
            
            print(f"\n  {component.upper()}: {identical_count}/{total_count} identical, max_diff: {max_diff:.6e}{max_diff_context}")
            
            # Show worst differences for this component
            different_comps = [c for c in comps if not c.get("identical", False)]
            if different_comps:
                different_comps.sort(key=lambda x: x.get("max_diff", 0), reverse=True)
                for i, comp in enumerate(different_comps[:3]):  # Top 3 worst
                    max_diff = comp.get('max_diff', 0)
                    print(f"    {i+1}. {comp['name']} | max_diff: {max_diff:.6e}")
                    
                    # Show actual max difference location
                    if comp.get("max_diff_index") is not None and comp.get("max_diff_value1") is not None:
                        index = comp["max_diff_index"]
                        val1 = comp["max_diff_value1"]
                        val2 = comp["max_diff_value2"]
                        shape = comp.get("shape1", "unknown")
                        if isinstance(index, tuple):
                            index_str = str(index)
                        else:
                            index_str = str(index)
                        print(f"       Max diff at index {index_str} in shape {shape}:")
                        print(f"       Run1: {val1:.6e}")
                        print(f"       Run2: {val2:.6e}")
                        print(f"       |Diff|: {abs(val1 - val2):.6e}")
                    
                    # Show first few values for additional context
                    if comp.get("tensor1_values") and comp.get("tensor2_values"):
                        vals1 = comp["tensor1_values"][:10]  # First 3 for space
                        vals2 = comp["tensor2_values"][:10]
                        vals1_str = ", ".join([f"{v:8.3e}" for v in vals1])
                        vals2_str = ", ".join([f"{v:8.3e}" for v in vals2])
                        diffs = [abs(v1 - v2) for v1, v2 in zip(vals1, vals2)]
                        diff_str = ", ".join([f"{d:8.3e}" for d in diffs])
                        
                        print(f"       First 3 values:")
                        print(f"         Run1: [{vals1_str}]")
                        print(f"         Run2: [{vals2_str}]")
                        print(f"         Diff: [{diff_str}]")
            
            # Show some identical ones for verification
            identical_comps = [c for c in comps if c.get("identical", False)]
            if identical_comps and len(identical_comps) <= 2:  # Only if few identical ones
                for comp in identical_comps[:1]:  # Just show one
                    print(f"    ✅ {comp['name']} | IDENTICAL")
                    if comp.get("tensor1_values"):
                        vals = comp["tensor1_values"][:5]
                        vals_str = ", ".join([f"{v:8.3e}" for v in vals])
                        print(f"       Values: [{vals_str}]")
    
    print("=" * 60)
