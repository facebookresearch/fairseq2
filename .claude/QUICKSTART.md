# Quick Start Guide

**You're working in the fairseq2 repository with Claude Code configured.**

## ✅ Everything Is Automatic

When you opened Claude Code in this repository:

1. ✅ The `SKILL` skill loaded automatically
2. ✅ You saw: "📚 fairseq2 repository loaded. fairseq2 skill is active."
3. ✅ All fairseq2 rules are now in effect

**You don't need to do anything to use it.**

## 🎯 What This Means

### Source Code Is Truth

Before Claude writes ANY code:
- ✅ Reads `src/fairseq2/[module].py` to verify APIs
- ✅ Checks git history for recent changes
- ✅ Verifies function signatures exist

### Forbidden Directories

Claude will NOT consult:
- ❌ `recipes/` directory (outdated examples)
- ❌ `doc/` directory (unless explicitly modifying docs)
- ❌ Internet sources or training data (outdated)

### Preferred Patterns

Claude will prefer:
- ✅ `fairseq2.gang` over `torch.distributed`
- ✅ `fairseq2.trainer` over raw training loops
- ✅ `fairseq2.data` over raw PyTorch DataLoader
- ✅ Import from installed library: `from fairseq2.X import Y`

## 🔄 After `/compact`

When you run `/compact` to compress conversation history:

1. ✅ Skill automatically re-loads
2. ✅ You see: "♻️ Re-loading fairseq2 skill..."
3. ✅ All rules restored to context

**You don't need to manually restore anything.**

## 🛠️ Your User Skills Still Work

If you have personal skills (like `10x-engineer` suite):

```bash
/test-driven-development    # TDD workflow
/systematic-debugging        # Debugging workflow
/requesting-code-review      # Code review process
```

**What happens:**
- ✅ Your user skill provides the workflow
- ✅ Local skill enforces fairseq2 rules
- ✅ They work together seamlessly

**Priority**: Local skill rules cannot be overridden (by design for correctness).

## 📚 Documentation

For more details, see:

- **`.claude/README.md`** - Complete setup documentation
- **`.claude/SKILL_INTEGRATION.md`** - How local + user skills work together
- **`.claude/SETUP_SUMMARY.md`** - Comprehensive summary of the setup
- **`.claude/skills/SKILL.md`** - The actual skill (auto-loaded)

## 🔧 Customization

### User-Specific Settings

To add your own permissions or settings:

1. Copy the template:
   ```bash
   cp .claude/settings.local.json.template .claude/settings.local.json
   ```

2. Edit `.claude/settings.local.json` to add your custom settings

3. This file is gitignored - won't conflict with team settings

### Common Customizations

```json
{
  "permissions": {
    "allow": [
      "Bash(source activate my_custom_env)",
      "Bash(my_tool:*)"
    ]
  }
}
```

## ⚡ Quick Examples

### Writing Code

**Ask**: "Help me create a script that uses fairseq2's gang for distributed training"

**What happens:**
1. Claude reads `src/fairseq2/gang.py` to verify current APIs
2. Checks for recent changes: `git log --oneline -- src/fairseq2/gang.py`
3. Writes code using verified APIs
4. Imports from installed library: `from fairseq2.gang import Gang`
5. Includes CPU/GPU/multi-GPU support

### Understanding Components

**Ask**: "How does fairseq2's data pipeline work?"

**What happens:**
1. Claude reads `src/fairseq2/data/` source code
2. Checks tests in `tests/` for usage examples
3. Explains based on actual implementation
4. Does NOT consult docs or recipes

### Fixing Bugs

**Ask**: "This code using setup_default_gang() is failing"

**What happens:**
1. Claude checks `src/fairseq2/gang.py`
2. Sees `setup_default_gang()` was removed in Feb 2025
3. Finds replacement: `get_default_gangs()`
4. Fixes code with correct API
5. Explains the change

## 🎓 Learning fairseq2

When learning fairseq2 with Claude:

**Recommended approach:**
1. Ask about components: "How does the trainer work?"
2. Ask for examples: "Show me a minimal training script"
3. Ask for verification: "Is this code using current APIs?"

**Claude will:**
- ✅ Read source code to answer accurately
- ✅ Provide examples with verified APIs
- ✅ Include multi-device support (CPU/GPU/multi-GPU)
- ✅ Use fairseq2 constructs (not raw PyTorch)

## 🚨 Common Pitfalls (Avoided)

### ❌ Outdated Documentation
**Without this setup:** Claude might reference outdated docs
**With this setup:** Claude reads `src/fairseq2/` for truth

### ❌ Removed APIs
**Without this setup:** Code uses `setup_default_gang()` (removed Feb 2025)
**With this setup:** Claude verifies and uses `get_default_gangs()`

### ❌ Wrong Import Paths
**Without this setup:** `from src.fairseq2.gang import Gang`
**With this setup:** `from fairseq2.gang import Gang`

### ❌ Raw PyTorch Patterns
**Without this setup:** Uses `torch.distributed.init_process_group()`
**With this setup:** Uses `fairseq2.gang.setup_gang()`

## 📊 Skill Status

Check if the skill is active:

```bash
# Skill loads automatically on session start
# Skill re-loads automatically after /compact
# No manual action needed
```

If you want to verify:
- Look for the startup message: "📚 fairseq2 repository loaded..."
- After `/compact`, look for: "♻️ Re-loading fairseq2 skill..."

## 🎯 Summary

**Three things to remember:**

1. **Everything is automatic** - skill loads at start, re-loads after `/compact`
2. **Source code is truth** - Claude verifies against `src/fairseq2/` before coding
3. **Your skills still work** - user skills complement the local skill

**You can focus on your work** - Claude Code handles correctness automatically.

---

**Questions or issues?** See `.claude/README.md` for complete documentation or `.claude/SETUP_SUMMARY.md` for detailed explanations.
