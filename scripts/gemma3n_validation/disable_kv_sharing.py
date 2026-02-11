"""Quick test: verify parity works with KV sharing disabled."""

from fairseq2.models.gemma3n.config import Gemma3nConfig

# Monkey-patch to disable KV sharing
original_get_kv_sharing_config = None

def disabled_kv_sharing_config(layer_idx, num_layers, num_kv_shared_layers):
    """Return config that disables KV sharing for all layers."""
    return False, None, False  # No sharing, no source, not a source

import fairseq2.models.gemma3n.config as config_module
import fairseq2.models.gemma3n.factory as factory_module

original_get_kv_sharing_config = config_module.get_kv_sharing_config
config_module.get_kv_sharing_config = disabled_kv_sharing_config
factory_module.get_kv_sharing_config = disabled_kv_sharing_config

print("KV sharing disabled via monkey-patch")
print("This test will show if KV sharing is causing the parity issue")
print()
print("Expected: If parity is good with KV sharing disabled, the issue is in KV sharing logic")
print("Expected: If parity is still bad, the issue is something else (maybe the callback fix)")
print()
print("To test, modify test_parity.py to import this file at the top")
