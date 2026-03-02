# Template Component Installation Guide

This guide explains how to make the fairseq2 skill installable from an organization template repository.

## Changes Made for Template Compatibility

### 1. Renamed Skill File

**Before**: `.claude/skills/doc-code-author.md`
**After**: `.claude/skills/SKILL.md`

**Reason**: Claude Code's template system requires skills to be named `SKILL.md` for component installation.

### 2. Created Component Metadata

**File**: `.claude/skills/component.yaml`

This metadata file defines:
- Component name: `fairseq2`
- Component type: `skill`
- Auto-loading behavior
- Compatibility information
- Documentation links

### 3. Updated All References

All documentation and configuration files have been updated to reference `SKILL.md` instead of `doc-code-author.md`:

- ✅ `.claude/settings.json` - Changed `autoLoad` array
- ✅ `.claude/hooks.json` - Changed `loadSkills` arrays
- ✅ `CLAUDE.md` - Updated skill file references
- ✅ All documentation files - Updated links and mentions

## How to Install from Template Repository

### Option 1: Local Repository Usage (Current Setup)

**Works now out of the box:**

1. Clone the fairseq2 repository
2. Open Claude Code
3. Skill automatically loads (via hooks)

**No installation command needed** - the skill is part of the repository.

### Option 2: Organization Template Installation

**For colleagues who want to install this as a component:**

If you publish this to an organization template repository (e.g., `your-org/claude-templates`), users can install it:

```bash
# Install the fairseq2 skill from organization template
claude code install your-org/fairseq2

# Or if it's in a subdirectory
claude code install your-org/claude-templates/fairseq2
```

This will:
1. Download `SKILL.md` and `component.yaml`
2. Install to their local Claude Code skills directory
3. Make it available for use in any project

### Option 3: Hybrid Approach (Recommended)

**Best of both worlds:**

1. **Local repository**: Skill auto-loads via hooks (current setup)
2. **Other projects**: Users can install from template to use elsewhere

**Setup for organization template repository:**

```
your-org/claude-templates/
├── fairseq2/
│   ├── .claude/
│   │   └── skills/
│   │       ├── SKILL.md
│   │       └── component.yaml
│   └── README.md
```

## Publishing to Organization Template Repository

### Step 1: Create Template Repository Structure

```bash
# In your organization's template repository
mkdir -p templates/fairseq2/.claude/skills

# Copy the skill and metadata
cp /path/to/fairseq2/.claude/skills/SKILL.md templates/fairseq2/.claude/skills/
cp /path/to/fairseq2/.claude/skills/component.yaml templates/fairseq2/.claude/skills/

# Optional: Copy documentation
cp /path/to/fairseq2/.claude/README.md templates/fairseq2/
cp /path/to/fairseq2/.claude/QUICKSTART.md templates/fairseq2/
```

### Step 2: Commit and Push

```bash
cd your-org/claude-templates
git add templates/fairseq2/
git commit -m "Add fairseq2 development skill"
git push
```

### Step 3: Configure Template Registry (if needed)

If your organization uses a custom template registry, configure it:

```json
// In organization's template registry config
{
  "templates": {
    "fairseq2": {
      "path": "templates/fairseq2",
      "type": "skill",
      "description": "fairseq2 development skill with source verification"
    }
  }
}
```

## Verification

### Test Local Installation

After renaming to `SKILL.md`, verify the skill still works:

```bash
cd /path/to/fairseq2
claude code

# You should see:
# 📚 fairseq2 repository loaded. fairseq2 development skill is active.
```

### Test Template Installation (from organization repo)

Once published to your template repository:

```bash
# In a different project
cd /path/to/other-project

# Install the skill
claude code install your-org/fairseq2

# Verify it's installed
claude code skills list
# Should show "fairseq2" in the list
```

## Troubleshooting Template Installation

### Error: "skill directory must contain SKILL.md file"

**Cause**: The skill file is not named `SKILL.md`

**Solution**: ✅ Already fixed - file is now named `SKILL.md`

### Error: "Component was excluded from the build"

**Cause**: Missing or invalid `component.yaml` metadata

**Solution**: ✅ Already fixed - `component.yaml` created with proper metadata

### Skill Not Auto-Loading in Repository

**Check**:
1. Verify `.claude/hooks.json` exists
2. Verify `loadSkills: ["SKILL"]` in hooks
3. Restart Claude Code

### Skill Not Found After Template Installation

**Check**:
1. Verify installation path: `~/.claude/skills/fairseq2/`
2. Verify `SKILL.md` exists in that directory
3. Run: `claude code skills list` to see installed skills

## Directory Structure After Changes

```
.claude/
├── skills/
│   ├── SKILL.md              ← Renamed from doc-code-author.md
│   └── component.yaml        ← New metadata file
├── settings.json             ← Updated to reference SKILL
├── hooks.json                ← Updated to load SKILL
├── README.md                 ← Updated references
├── QUICKSTART.md             ← Updated references
├── SETUP_SUMMARY.md          ← Updated references
├── SKILL_INTEGRATION.md      ← Updated references
└── INDEX.md                  ← Updated references
```

## Benefits of This Approach

### ✅ Dual Usage

**In fairseq2 repository:**
- Skill auto-loads via hooks
- No manual installation needed
- Always up-to-date with repository

**In other projects:**
- Install from organization template
- Use fairseq2 patterns elsewhere
- Can be updated independently

### ✅ Standard Naming

- `SKILL.md` is the Claude Code standard
- Compatible with template system
- Easy to recognize and maintain

### ✅ Organization-Wide Sharing

- Publish once to template repository
- All team members can install
- Consistent development practices

## Next Steps

### For Repository Maintainers

1. ✅ Changes complete - skill renamed and metadata created
2. Test locally to ensure auto-loading still works
3. Commit changes to repository
4. Optionally publish to organization template repository

### For Users

**If working in fairseq2 repository:**
- No action needed - skill auto-loads

**If working in other projects:**
- Install from organization template (once published)
- Skill will be available but won't auto-load (use manually)

### For Organization Admins

1. Create organization template repository structure
2. Copy `SKILL.md` and `component.yaml` to templates directory
3. Configure template registry if needed
4. Announce availability to team

## Summary

**What changed:**
- ✅ Skill file renamed: `doc-code-author.md` → `SKILL.md`
- ✅ Component metadata created: `component.yaml`
- ✅ All references updated in config and docs

**What works:**
- ✅ Local repository auto-loading (via hooks)
- ✅ Template installation (via claude code install)
- ✅ Organization-wide sharing (via template repo)

**What to do:**
- For fairseq2 repo: Nothing - already working
- For template publishing: Copy to org template repo
- For users elsewhere: Install from template once published

The skill is now **fully compatible** with Claude Code's template component system while maintaining all existing functionality.
