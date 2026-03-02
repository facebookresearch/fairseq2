# Skill Integration Guide

This guide explains how the fairseq2 repository's local `SKILL` skill integrates with your personal user skills.

## Automatic Integration

When you open Claude Code in this repository, both local and user skills are available:

### Local Skill (Highest Priority)
- **`SKILL`** - Automatically loaded from `.claude/skills/SKILL.md`
- Enforces fairseq2-specific rules
- Source of truth for this repository

### User Skills (Complementary)
- Available from your `~/.claude/skills/` or other configured locations
- Can be invoked as usual
- Automatically respect local skill constraints

## How Skills Work Together

### Example: Using TDD with fairseq2

When you use a user skill like `10x-engineer:test-driven-development`:

```bash
/test-driven-development
```

**What happens:**
1. ✅ TDD skill loads and guides you through test-first development
2. ✅ Local `SKILL` skill enforces fairseq2 rules:
   - Must verify APIs against `src/fairseq2/` before writing code
   - Cannot consult `recipes/` or `doc/` directories
   - Must use fairseq2 constructs (gang, trainer, etc.)
3. ✅ Result: TDD workflow that respects fairseq2 architecture

### Example: Debugging with fairseq2

When you use `10x-engineer:systematic-debugging`:

```bash
/systematic-debugging
```

**What happens:**
1. ✅ Debugging skill provides systematic debugging workflow
2. ✅ Local skill ensures:
   - Debug against actual source in `src/fairseq2/`
   - Check git history for recent API changes
   - Don't trust outdated examples
3. ✅ Result: Proper debugging that accounts for rapid API changes

### Example: Code Review

When you use `10x-engineer:requesting-code-review`:

```bash
/requesting-code-review
```

**What happens:**
1. ✅ Code review skill structures the review request
2. ✅ Local skill verifies:
   - Code uses correct fairseq2 APIs (from source)
   - Imports are from installed library, not local paths
   - Prefers fairseq2 constructs over raw PyTorch
3. ✅ Result: Review request that follows fairseq2 conventions

## Skill Priority Hierarchy

```
┌─────────────────────────────────────────┐
│  Local: doc-code-author                 │  ← HIGHEST PRIORITY
│  - Source code verification             │  ← Cannot be overridden
│  - Forbidden directories                │  ← Absolute rules
│  - Import patterns                      │  ← Must follow
│  - fairseq2 constructs preference       │
└─────────────────────────────────────────┘
              ↓ (enforces constraints)
┌─────────────────────────────────────────┐
│  User Skills (10x-engineer suite, etc.) │
│  - Workflow guidance                    │
│  - Methodology                          │  ← Adapts to local rules
│  - Best practices                       │
│  - Tooling                              │
└─────────────────────────────────────────┘
```

## Conflict Resolution

If a user skill suggests something that conflicts with the local skill:

**Local skill wins, always.**

Example conflicts and resolutions:

| User Skill Says | Local Skill Says | Result |
|----------------|------------------|---------|
| "Look at the documentation" | "Don't consult doc/, read source" | Read `src/fairseq2/` |
| "Check the examples" | "Forbidden: recipes/" | Skip recipes, check `tests/` |
| "Import from local path" | "Import from installed library" | Use `from fairseq2.X import Y` |
| "Use PyTorch DDP" | "Prefer fairseq2 gang" | Use `from fairseq2.gang import Gang` |

## Recommended User Skills for fairseq2

These user skills complement `SKILL` well:

### Development Workflow
- ✅ `10x-engineer:test-driven-development` - Write tests first
- ✅ `10x-engineer:systematic-debugging` - Debug methodically
- ✅ `10x-engineer:brainstorming` - Explore requirements before coding

### Quality & Review
- ✅ `10x-engineer:requesting-code-review` - Structure review requests
- ✅ `10x-engineer:receiving-code-review` - Handle review feedback
- ✅ `10x-engineer:verification-before-completion` - Verify before claiming done

### Advanced Patterns
- ✅ `10x-engineer:defense-in-depth` - Multi-layer validation
- ✅ `10x-engineer:root-cause-tracing` - Trace bugs to source
- ✅ `10x-engineer:condition-based-waiting` - Fix flaky tests

### Planning & Execution
- ✅ `10x-engineer:writing-plans` - Plan before implementing
- ✅ `10x-engineer:executing-plans` - Execute with review checkpoints
- ✅ `10x-engineer:subagent-driven-development` - Parallel task execution

## Skills to Avoid or Use Carefully

Some skills may not fit well with fairseq2 development:

### ⚠️ Generic Documentation Skills
- **Problem**: Might suggest reading docs instead of source
- **Solution**: Local skill enforces source-first approach automatically

### ⚠️ Generic Example-Based Skills
- **Problem**: Might reference examples that are outdated
- **Solution**: Local skill forbids recipes/ and enforces source verification

### ⚠️ Library-Agnostic ML Skills
- **Problem**: Might suggest PyTorch patterns instead of fairseq2 abstractions
- **Solution**: Local skill enforces fairseq2 constructs preference

**Bottom line**: User skills still work, they just automatically adapt to fairseq2's constraints.

## Creating Custom Skills for fairseq2

If you want to create your own skills that complement `SKILL`:

### Good Custom Skill Ideas
- Performance profiling workflows for fairseq2 models
- Dataset preparation patterns for fairseq2 data pipeline
- Multi-GPU debugging for gang-based distributed training
- Model architecture design patterns

### How to Make Them Compatible

In your custom skill, reference the local skill:

```markdown
## Integration with Repository Rules

This skill works alongside the repository's `SKILL` skill:
- All API verification rules from `SKILL` apply
- Source code in `src/fairseq2/` is the source of truth
- Forbidden directories (recipes/, doc/) remain forbidden
- Prefer fairseq2 constructs as specified in `SKILL`
```

## Testing Skill Integration

To verify your user skills work well with the local skill:

1. **Invoke both skills** in a test scenario:
   ```bash
   # Example: TDD with fairseq2
   /test-driven-development
   # Then write a feature that uses fairseq2 APIs
   ```

2. **Check that:**
   - ✅ Source code verification happens before coding
   - ✅ Forbidden directories are avoided
   - ✅ fairseq2 constructs are preferred
   - ✅ User skill workflow still provides value

3. **If conflicts arise:**
   - Local skill constraints should be respected
   - User skill should adapt gracefully
   - Workflow should still be helpful

## Summary

**The integration is automatic and seamless:**

1. ✅ Local `SKILL` skill loads automatically
2. ✅ Your user skills remain available and useful
3. ✅ Local skill enforces fairseq2-specific rules
4. ✅ User skills provide workflow and methodology
5. ✅ Conflicts resolve in favor of local skill (repository truth)

**You don't need to do anything special** - just use your user skills as normal, and they'll automatically respect fairseq2's requirements.

**Result:** Best of both worlds - fairseq2 correctness + your preferred workflows.
