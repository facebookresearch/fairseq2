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

def analyze_layer_differences(results: Dict[str, Any], threshold: float = 1e-6) -> Dict[str, Any]:
    """
    Analyze differences per layer/model part with structured output.
    
    Args:
        results: Output from compare_runs()
        threshold: Threshold for considering a difference significant
        
    Returns:
        Dictionary with structured analysis per layer/component
    """
    layer_analysis = {}
    
    for comp in results["comparisons"]:
        name = comp["name"]
        category = comp["category"]
        
        # Extract layer/component identifier
        layer_id = _extract_layer_id(name)
        
        # Create unique key for this layer/component combination
        key = f"{category}_{layer_id}"
        
        if key not in layer_analysis:
            layer_analysis[key] = {
                "layer_id": layer_id,
                "category": category,
                "tensors": [],
                "total_differences": 0,
                "difference_indices": [],
                "top_differences": [],
                "total_magnitude": 0.0,
                "tensor_count": 0
            }
        
        # Use the already computed comparison data
        max_diff = comp.get("max_diff", 0.0)
        mean_diff = comp.get("mean_diff", 0.0)
        is_identical = comp.get("identical", True)
        
        # Check if this tensor has significant differences
        has_differences = not is_identical and max_diff > threshold
        
        # For tensors with differences, we need to reload them to get detailed analysis
        if has_differences:
            # Try to reconstruct filepaths from the artifacts
            artifacts1, artifacts2 = _get_artifact_paths_from_results(results)
            key_lookup = f"{category}_{comp.get('key', '').split('_', 1)[-1]}"
            
            if key_lookup in artifacts1 and key_lookup in artifacts2:
                tensor1 = torch.load(artifacts1[key_lookup])
                tensor2 = torch.load(artifacts2[key_lookup])
                tensor_analysis = _analyze_tensor_differences(tensor1, tensor2, threshold)
            else:
                # Fallback: create analysis from existing comparison data
                tensor_analysis = _create_analysis_from_comparison(comp, threshold)
        else:
            # No significant differences
            tensor_analysis = {
                "num_differences": 0,
                "difference_indices": [],
                "top_differences": [],
                "total_magnitude": 0.0,
                "shape": comp.get("shape1", [])
            }
        
        layer_analysis[key]["tensors"].append({
            "name": name,
            "tensor_analysis": tensor_analysis,
            "comparison": comp  # Keep original comparison for reference
        })
        
        # Aggregate statistics
        layer_analysis[key]["total_differences"] += tensor_analysis["num_differences"]
        layer_analysis[key]["difference_indices"].extend(tensor_analysis["difference_indices"])
        layer_analysis[key]["top_differences"].extend(tensor_analysis["top_differences"])
        layer_analysis[key]["total_magnitude"] += tensor_analysis["total_magnitude"]
        layer_analysis[key]["tensor_count"] += 1
    
    # Post-process each layer's analysis
    for key in layer_analysis:
        layer = layer_analysis[key]
        
        # Sort and limit difference indices (compact representation)
        layer["difference_indices"] = _compact_indices(layer["difference_indices"][:20])
        
        # Sort and limit top differences
        layer["top_differences"].sort(key=lambda x: x["magnitude"], reverse=True)
        layer["top_differences"] = layer["top_differences"][:3]
        
        # Calculate average magnitude
        layer["avg_magnitude"] = layer["total_magnitude"] / layer["tensor_count"] if layer["tensor_count"] > 0 else 0.0
    
    return layer_analysis


def _extract_layer_id(tensor_name: str) -> str:
    """Extract layer/component identifier from tensor name."""
    import re
    
    # Handle encoder layers
    layer_match = re.search(r'encoder_layers_(\d+)', tensor_name)
    if layer_match:
        return f"layer_{int(layer_match.group(1)):03d}"
    
    # Handle other components
    if "encoder_frontend" in tensor_name or "frontend" in tensor_name:
        return "frontend"
    elif "final_proj" in tensor_name or "classifier" in tensor_name or "head" in tensor_name:
        return "final_proj"
    elif "loss" in tensor_name:
        return "loss"
    elif "embedding" in tensor_name:
        return "embedding"
    else:
        # Try to extract a meaningful component name
        parts = tensor_name.split("_")
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
        return "other"


def _get_artifact_paths_from_results(results: Dict[str, Any]) -> tuple:
    """Extract artifact paths from results (this is a workaround since compare_runs doesn't store paths)."""
    # This is a limitation - we need the original run directories to reload tensors
    # For now, return empty dicts and rely on fallback analysis
    return {}, {}


def analyze_layer_differences_detailed(run1_dir: str, run2_dir: str, threshold: float = 1e-6) -> Dict[str, Any]:
    """
    Analyze differences per layer/model part with detailed index information.
    This version reloads tensors to get exact differing indices.
    
    Args:
        run1_dir: Path to first run directory
        run2_dir: Path to second run directory  
        threshold: Threshold for considering a difference significant
        
    Returns:
        Dictionary with detailed analysis per layer/component
    """
    from pathlib import Path
    import torch
    
    run1_path = Path(run1_dir)
    run2_path = Path(run2_dir)
    
    # Collect all artifacts from both runs
    categories = ["weights", "gradients", "activations", "step_data"]
    artifacts1 = {}
    artifacts2 = {}
    
    for category in categories:
        cat_dir1 = run1_path / category
        cat_dir2 = run2_path / category
        
        if cat_dir1.exists():
            for file_path in cat_dir1.glob("*.pt"):
                key = f"{category}_{file_path.name}"
                artifacts1[key] = file_path
                
        if cat_dir2.exists():
            for file_path in cat_dir2.glob("*.pt"):
                key = f"{category}_{file_path.name}"
                artifacts2[key] = file_path
    
    # Find common artifacts
    common_keys = set(artifacts1.keys()) & set(artifacts2.keys())
    
    layer_analysis = {}
    
    for key in sorted(common_keys):
        # Load tensors
        tensor1 = torch.load(artifacts1[key])
        tensor2 = torch.load(artifacts2[key])
        
        # Extract info from filename
        filename = artifacts1[key].name
        category = key.split('_')[0]
        name_part = filename.replace('.pt', '')
        
        # Extract step and name from filename
        step_str = "unknown"
        name = name_part
        if name_part.startswith('step_'):
            parts = name_part.split('_', 2)
            if len(parts) >= 2:
                step_str = parts[1]
                name = '_'.join(parts[2:]) if len(parts) > 2 else parts[1]
        
        # Extract layer/component identifier
        layer_id = _extract_layer_id(name)
        
        # Create unique key for this layer/component combination
        analysis_key = f"{category}_{layer_id}"
        
        if analysis_key not in layer_analysis:
            layer_analysis[analysis_key] = {
                "layer_id": layer_id,
                "category": category,
                "tensors": [],
                "total_differences": 0,
                "all_difference_indices": [],
                "top_differences": [],
                "total_magnitude": 0.0,
                "tensor_count": 0
            }
        
        # Analyze this tensor pair
        tensor_analysis = _analyze_tensor_differences_detailed(tensor1, tensor2, threshold)
        
        layer_analysis[analysis_key]["tensors"].append({
            "name": name,
            "step": step_str,
            "filename": filename,
            "tensor_analysis": tensor_analysis
        })
        
        # Aggregate statistics
        layer_analysis[analysis_key]["total_differences"] += tensor_analysis["num_differences"]
        layer_analysis[analysis_key]["all_difference_indices"].extend(tensor_analysis["difference_indices"])
        layer_analysis[analysis_key]["top_differences"].extend(tensor_analysis["top_differences"])
        layer_analysis[analysis_key]["total_magnitude"] += tensor_analysis["total_magnitude"]
        layer_analysis[analysis_key]["tensor_count"] += 1
    
    # Post-process each layer's analysis
    for analysis_key in layer_analysis:
        layer = layer_analysis[analysis_key]
        
        # Sort and limit top differences
        layer["top_differences"].sort(key=lambda x: x["magnitude"], reverse=True)
        layer["top_differences"] = layer["top_differences"][:3]
        
        # Calculate average magnitude
        layer["avg_magnitude"] = layer["total_magnitude"] / layer["tensor_count"] if layer["tensor_count"] > 0 else 0.0
    
    return layer_analysis


def _analyze_tensor_differences_detailed(tensor1: Tensor, tensor2: Tensor, threshold: float) -> Dict[str, Any]:
    """Analyze differences between two tensors with detailed index information."""
    diff = torch.abs(tensor1 - tensor2)
    
    # Find indices where difference exceeds threshold
    significant_diff_mask = diff > threshold
    significant_indices = torch.nonzero(significant_diff_mask, as_tuple=False).cpu()
    
    # Convert to list of tuples/ints
    if significant_indices.numel() == 0:
        difference_indices = []
    elif significant_indices.shape[1] == 1:
        # 1D tensor - convert to list of ints
        difference_indices = [int(idx.item()) for idx in significant_indices.flatten()]
    else:
        # Multi-dimensional - convert to list of tuples
        difference_indices = [tuple(idx.tolist()) for idx in significant_indices]
    
    # Get top differences
    flat_diff = diff.flatten()
    top_k = min(3, len(flat_diff))
    
    top_differences = []
    if top_k > 0:
        top_values, top_indices = torch.topk(flat_diff, top_k)
        
        for i in range(top_k):
            flat_idx = top_indices[i].item()
            unravel_idx = torch.unravel_index(torch.tensor(flat_idx), tensor1.shape)
            
            if len(unravel_idx) == 1:
                index = int(unravel_idx[0].item())
            else:
                index = tuple(idx.item() for idx in unravel_idx)
                
            top_differences.append({
                "magnitude": float(top_values[i].item()),
                "index": index,
                "value1": float(tensor1.flatten()[flat_idx].item()),
                "value2": float(tensor2.flatten()[flat_idx].item())
            })
    
    return {
        "num_differences": len(difference_indices),
        "difference_indices": difference_indices,
        "top_differences": top_differences,
        "total_magnitude": float(torch.sum(diff).item()),
        "shape": list(tensor1.shape)
    }


def print_structured_differences_detailed(run1_dir: str, run2_dir: str, threshold: float = 1e-6):
    """Print detailed structured difference analysis with exact differing indices."""
    layer_analysis = analyze_layer_differences_detailed(run1_dir, run2_dir, threshold)
    
    if not layer_analysis:
        print("No significant differences found!")
        return
    
    print("=" * 80)
    print("DETAILED STRUCTURED DIFFERENCE ANALYSIS")
    print("=" * 80)
    print(f"Threshold: {threshold:.2e}")
    print()
    
    # Group by category
    categories = {}
    for key, analysis in layer_analysis.items():
        cat = analysis["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((key, analysis))
    
    for category in sorted(categories.keys()):
        print(f"\n{'='*60}")
        print(f"CATEGORY: {category.upper()}")
        print(f"{'='*60}")
        
        # Sort layers within category
        layers = sorted(categories[category], key=lambda x: x[1]["layer_id"])
        
        for key, analysis in layers:
            layer_id = analysis["layer_id"]
            total_diffs = analysis["total_differences"]
            tensor_count = analysis["tensor_count"]
            
            print(f"\n{layer_id.upper()}:")
            print(f"  Tensors analyzed: {tensor_count}")
            print(f"  Total differing elements > {threshold:.2e}: {total_diffs}")
            
            if total_diffs > 0:
                # Show compact representation of differing indices (top 20)
                all_indices = analysis["all_difference_indices"]
                indices_str = _compact_indices(all_indices[:20])
                more_count = max(0, len(all_indices) - 20)
                if more_count > 0:
                    print(f"  Differing indices (first 20 of {len(all_indices)}): {indices_str} ... +{more_count} more")
                else:
                    print(f"  Differing indices: {indices_str}")
                
                # Show top 3 maximum differences
                print(f"  Top 3 maximum differences:")
                for i, diff in enumerate(analysis["top_differences"], 1):
                    magnitude = diff["magnitude"]
                    index = diff["index"]
                    val1 = diff["value1"]
                    val2 = diff["value2"]
                    print(f"    {i}. Index {index}: |{val1:.6e} - {val2:.6e}| = {magnitude:.6e}")
                
                # Show total magnitude
                total_mag = analysis["total_magnitude"]
                avg_mag = analysis["avg_magnitude"]
                print(f"  Total magnitude: {total_mag:.6e}")
                print(f"  Average magnitude per tensor: {avg_mag:.6e}")
                
                # Show detailed tensor breakdown
                print(f"  Tensor breakdown:")
                for tensor_info in analysis["tensors"]:
                    t_name = tensor_info["name"]
                    t_step = tensor_info["step"]
                    t_analysis = tensor_info["tensor_analysis"]
                    t_diffs = t_analysis["num_differences"]
                    t_mag = t_analysis["total_magnitude"]
                    t_shape = t_analysis["shape"]
                    
                    if t_diffs > 0:
                        print(f"    ❌ {t_name} (step {t_step}, shape {t_shape}):")
                        print(f"       {t_diffs} differing elements, magnitude {t_mag:.6e}")
                        
                        # Show indices for this tensor (compact, first 10)
                        t_indices = t_analysis["difference_indices"]
                        if len(t_indices) <= 10:
                            t_indices_str = _compact_indices(t_indices)
                            print(f"       Indices: {t_indices_str}")
                        else:
                            t_indices_str = _compact_indices(t_indices[:10])
                            print(f"       Indices (first 10 of {len(t_indices)}): {t_indices_str} ... +{len(t_indices)-10} more")
                        
                        # Show top difference for this tensor
                        if t_analysis["top_differences"]:
                            top_diff = t_analysis["top_differences"][0]
                            print(f"       Max diff: {top_diff['magnitude']:.6e} at {top_diff['index']}")
                    else:
                        print(f"    ✅ {t_name} (step {t_step}, shape {t_shape}): identical")
            else:
                print(f"  ✅ All tensors identical within threshold")


def analyze_layer_differences_simple(results: Dict[str, Any], threshold: float = 1e-6) -> Dict[str, Any]:
    """
    Simplified analysis using only the comparison data from compare_runs().
    This version doesn't reload tensors, so it's less detailed but more reliable.
    """
    layer_analysis = {}
    
    for comp in results["comparisons"]:
        name = comp["name"]
        category = comp["category"]
        max_diff = comp.get("max_diff", 0.0)
        
        # Extract layer/component identifier
        layer_id = _extract_layer_id(name)
        
        # Create unique key for this layer/component combination
        key = f"{category}_{layer_id}"
        
        if key not in layer_analysis:
            layer_analysis[key] = {
                "layer_id": layer_id,
                "category": category,
                "tensors": [],
                "total_differences": 0,
                "max_differences": [],
                "total_magnitude": 0.0,
                "tensor_count": 0
            }
        
        # Check if this tensor has significant differences
        has_differences = max_diff > threshold
        
        tensor_info = {
            "name": name,
            "max_diff": max_diff,
            "mean_diff": comp.get("mean_diff", 0.0),
            "has_differences": has_differences,
            "shape": comp.get("shape1", []),
            "max_diff_index": comp.get("max_diff_index"),
            "max_diff_value1": comp.get("max_diff_value1"),
            "max_diff_value2": comp.get("max_diff_value2")
        }
        
        layer_analysis[key]["tensors"].append(tensor_info)
        
        # Aggregate statistics
        if has_differences:
            layer_analysis[key]["total_differences"] += 1
            layer_analysis[key]["max_differences"].append({
                "name": name,
                "magnitude": max_diff,
                "index": comp.get("max_diff_index"),
                "value1": comp.get("max_diff_value1"),
                "value2": comp.get("max_diff_value2")
            })
        
        # Estimate total magnitude using mean_diff * tensor size
        tensor_size = 1
        for dim in comp.get("shape1", []):
            tensor_size *= dim
        estimated_magnitude = comp.get("mean_diff", 0.0) * tensor_size
        layer_analysis[key]["total_magnitude"] += estimated_magnitude
        layer_analysis[key]["tensor_count"] += 1
    
    # Post-process each layer's analysis
    for key in layer_analysis:
        layer = layer_analysis[key]
        
        # Sort max differences by magnitude
        layer["max_differences"].sort(key=lambda x: x["magnitude"], reverse=True)
        layer["max_differences"] = layer["max_differences"][:3]  # Keep top 3
        
        # Calculate average magnitude
        layer["avg_magnitude"] = layer["total_magnitude"] / layer["tensor_count"] if layer["tensor_count"] > 0 else 0.0
    
    return layer_analysis
    """Analyze differences between two tensors."""
    diff = torch.abs(tensor1 - tensor2)
    
    # Find indices where difference exceeds threshold
    significant_diff_mask = diff > threshold
    significant_indices = torch.nonzero(significant_diff_mask).cpu().numpy()
    
    # Get top differences
    flat_diff = diff.flatten()
    top_k = min(3, len(flat_diff))
    top_values, top_indices = torch.topk(flat_diff, top_k)
    
    top_differences = []
    for i in range(top_k):
        flat_idx = top_indices[i].item()
        unravel_idx = torch.unravel_index(torch.tensor(flat_idx), tensor1.shape)
        top_differences.append({
            "magnitude": float(top_values[i].item()),
            "index": tuple(unravel_idx.tolist()) if len(unravel_idx) > 1 else int(unravel_idx.item()),
            "value1": float(tensor1.flatten()[flat_idx].item()),
            "value2": float(tensor2.flatten()[flat_idx].item())
        })
    
    return {
        "num_differences": len(significant_indices),
        "difference_indices": [tuple(idx) if len(idx) > 1 else int(idx[0]) for idx in significant_indices],
        "top_differences": top_differences,
        "total_magnitude": float(torch.sum(diff).item()),
        "shape": list(tensor1.shape)
    }


def _compact_indices(indices: list) -> str:
    """Convert list of indices to compact string representation."""
    if not indices:
        return "[]"
    
    # For scalar indices, try to find ranges
    if all(isinstance(idx, int) for idx in indices):
        indices.sort()
        ranges = []
        start = indices[0]
        end = indices[0]
        
        for i in range(1, len(indices)):
            if indices[i] == end + 1:
                end = indices[i]
            else:
                if start == end:
                    ranges.append(str(start))
                else:
                    ranges.append(f"{start}-{end}")
                start = end = indices[i]
        
        # Add the last range
        if start == end:
            ranges.append(str(start))
        else:
            ranges.append(f"{start}-{end}")
        
        return "[" + ", ".join(ranges) + "]"
    
    # For tuple indices, just show first few
    if len(indices) <= 5:
        return str(indices)
    else:
        return str(indices[:5]) + f"... ({len(indices)} total)"


def print_structured_differences_simple(results: Dict[str, Any], threshold: float = 1e-6):
    """Print structured difference analysis using only comparison data (no tensor reloading)."""
    layer_analysis = analyze_layer_differences_simple(results, threshold)
    
    if not layer_analysis:
        print("No significant differences found!")
        return
    
    print("=" * 80)
    print("STRUCTURED DIFFERENCE ANALYSIS")
    print("=" * 80)
    print(f"Threshold: {threshold:.2e}")
    print()
    
    # Group by category
    categories = {}
    for key, analysis in layer_analysis.items():
        cat = analysis["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((key, analysis))
    
    for category in sorted(categories.keys()):
        print(f"\n{'='*60}")
        print(f"CATEGORY: {category.upper()}")
        print(f"{'='*60}")
        
        # Sort layers within category
        layers = sorted(categories[category], key=lambda x: x[1]["layer_id"])
        
        for key, analysis in layers:
            layer_id = analysis["layer_id"]
            total_diffs = analysis["total_differences"]
            tensor_count = analysis["tensor_count"]
            
            print(f"\n{layer_id.upper()}:")
            print(f"  Tensors analyzed: {tensor_count}")
            print(f"  Tensors with differences > {threshold:.2e}: {total_diffs}")
            
            if total_diffs > 0:
                # Show top 3 maximum differences
                print(f"  Top 3 maximum differences:")
                for i, diff in enumerate(analysis["max_differences"], 1):
                    magnitude = diff["magnitude"]
                    index = diff["index"]
                    val1 = diff["value1"]
                    val2 = diff["value2"]
                    tensor_name = diff["name"]
                    print(f"    {i}. {tensor_name}")
                    print(f"       Max diff: {magnitude:.6e} at index {index}")
                    if val1 is not None and val2 is not None:
                        print(f"       Values: {val1:.6e} vs {val2:.6e}")
                
                # Show total magnitude estimate
                total_mag = analysis["total_magnitude"]
                avg_mag = analysis["avg_magnitude"]
                print(f"  Estimated total magnitude: {total_mag:.6e}")
                print(f"  Average magnitude per tensor: {avg_mag:.6e}")
                
                # Show tensor breakdown
                print(f"  Tensor breakdown:")
                for tensor_info in analysis["tensors"]:
                    t_name = tensor_info["name"]
                    t_shape = tensor_info["shape"]
                    has_diff = tensor_info["has_differences"]
                    max_diff = tensor_info["max_diff"]
                    if has_diff:
                        print(f"    ❌ {t_name} (shape {t_shape}): max_diff {max_diff:.6e}")
                    else:
                        print(f"    ✅ {t_name} (shape {t_shape}): identical")
            else:
                print(f"  ✅ All tensors identical within threshold")


def print_difference_summary_by_category_simple(results: Dict[str, Any], threshold: float = 1e-6):
    """Print a high-level summary using only comparison data."""
    layer_analysis = analyze_layer_differences_simple(results, threshold)
    
    print("=" * 60)
    print("DIFFERENCE SUMMARY BY CATEGORY")
    print("=" * 60)
    print(f"Threshold: {threshold:.2e}")
    print()
    
    # Aggregate by category
    category_stats = {}
    for key, analysis in layer_analysis.items():
        cat = analysis["category"]
        if cat not in category_stats:
            category_stats[cat] = {
                "layers": 0,
                "total_tensor_differences": 0,
                "total_magnitude": 0.0,
                "max_magnitude": 0.0,
                "tensors": 0
            }
        
        stats = category_stats[cat]
        stats["layers"] += 1
        stats["total_tensor_differences"] += analysis["total_differences"]
        stats["total_magnitude"] += analysis["total_magnitude"]
        stats["tensors"] += analysis["tensor_count"]
        
        # Update max magnitude
        for diff in analysis["max_differences"]:
            stats["max_magnitude"] = max(stats["max_magnitude"], diff["magnitude"])
    
    # Print summary
    for category in sorted(category_stats.keys()):
        stats = category_stats[category]
        print(f"{category.upper()}:")
        print(f"  Components: {stats['layers']}")
        print(f"  Total tensors: {stats['tensors']}")
        print(f"  Tensors with differences: {stats['total_tensor_differences']}")
        print(f"  Estimated total magnitude: {stats['total_magnitude']:.6e}")
        print(f"  Max difference: {stats['max_magnitude']:.6e}")
        if stats['layers'] > 0:
            print(f"  Avg different tensors per component: {stats['total_tensor_differences']/stats['layers']:.1f}")
        print()
