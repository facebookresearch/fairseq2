# Claude Code Setup Summary

This document summarizes the complete Claude Code setup for the fairseq2 repository.

## What Was Created

### Directory Structure

```
.claude/
├── README.md                          # Complete setup documentation
├── SKILL_INTEGRATION.md              # How local + user skills work together
├── settings.json                      # Repository-wide settings (committed)
├── settings.local.json                # User-specific settings (gitignored)
├── settings.local.json.template       # Template for local settings
├── hooks.json                         # Auto-loading hooks (committed)
└── skills/
    └── SKILL.md            # Primary fairseq2 skill (auto-loaded)
```

### Files Modified

- **`CLAUDE.md`** - Updated to reference `.claude/` directory and automatic loading
- **`.gitignore`** - Added `.claude/*.local.*` to ignore user-specific settings

## How It Works

### 1. Automatic Skill Loading

When you open Claude Code in this repository:

```
Session Start
     ↓
.claude/hooks.json triggers
     ↓
fairseq2 skill loads
     ↓
You see: "📚 fairseq2 repository loaded. fairseq2 skill is active."
     ↓
All rules are immediately in effect
```

### 2. Persistent Context After `/compact`

When you run `/compact`:

```
/compact executed
     ↓
Conversation history compressed
     ↓
.claude/hooks.json triggers
     ↓
fairseq2 skill re-loads
     ↓
You see: "♻️ Re-loading fairseq2 skill after /compact..."
     ↓
All rules restored to context
```

### 3. Skill Priority System

```
Priority Level 1 (HIGHEST): Local Repository Skill
├── SKILL.md
├── Source: .claude/skills/
├── Auto-loaded: Yes
├── Priority: Cannot be overridden
└── Rules:
    ├── Source code in src/fairseq2/ is ONLY truth
    ├── FORBIDDEN: recipes/ and doc/ directories
    ├── Must verify APIs against source before coding
    ├── Prefer fairseq2 constructs over PyTorch
    └── Import from installed library, not local paths

Priority Level 2 (LOWER): User Skills
├── Examples: 10x-engineer:*, custom skills
├── Source: ~/.claude/skills/ or other locations
├── Auto-loaded: No (invoke manually)
├── Priority: Adapts to local skill rules
└── Behavior:
    ├── Still available and functional
    ├── Respect local skill constraints
    ├── Complement with workflow guidance
    └── Conflicts resolve in favor of local skill
```

## Key Features

### ✅ Zero Manual Configuration

Users don't need to do anything:
- Skill loads automatically on session start
- Re-loads automatically after `/compact`
- No need to remember to invoke it
- No need to manually re-read after compression

### ✅ User Skill Compatibility

User skills from `10x-engineer` suite and others:
- Still available via `/skill-name` or Skill tool
- Automatically respect local skill constraints
- Complement local skill with workflow guidance
- Seamless integration

### ✅ Team Consistency

All team members get the same behavior:
- Same skills auto-load
- Same rules enforced
- Same forbidden directories
- User-specific settings separate (`.local.json`)

### ✅ Context Persistence

Critical context survives compression:
- Skill auto-reloads after `/compact`
- Rules remain in effect
- No need to manually restore guidelines

## Configuration Files Explained

### `settings.json` (Committed)

**Purpose**: Repository-wide Claude Code settings

**Key sections**:
```json
{
  "skills": {
    "loadOrder": ["local", "user"],
    "local": {
      "path": ".claude/skills",
      "priority": "highest",
      "autoLoad": ["SKILL"]
    }
  },
  "permissions": {
    "allow": ["Bash(git:*)", "Bash(python:*)", ...]
  },
  "context": {
    "always_include": ["CLAUDE.md", ".claude/skills/SKILL.md"]
  }
}
```

**Who modifies**: Team (committed to git)

### `settings.local.json` (Gitignored)

**Purpose**: User-specific overrides

**Key sections**:
```json
{
  "permissions": {
    "allow": [
      "Bash(source activate my_custom_env)",
      "Bash(my_custom_tool:*)"
    ]
  }
}
```

**Who modifies**: Individual users (not committed)

### `hooks.json` (Committed)

**Purpose**: Define when skills load and reminders show

**Key hooks**:
```json
{
  "hooks": {
    "session:start": {
      "description": "Load fairseq2 skill at session start",
      "loadSkills": ["SKILL"]
    },
    "command:compact": {
      "description": "Re-load skill after /compact",
      "loadSkills": ["SKILL"]
    },
    "before:code": {
      "description": "Reminder to verify against source",
      "command": "echo '⚠️ Reminder: Verify APIs...'"
    }
  }
}
```

**Who modifies**: Team (committed to git)

## Documentation Files

### `.claude/README.md`

**Audience**: All users of the repository

**Contents**:
- Complete directory structure explanation
- Skill priority system
- Configuration file purposes
- How to use the setup
- Troubleshooting guide

### `.claude/SKILL_INTEGRATION.md`

**Audience**: Users with personal skills (10x-engineer suite, etc.)

**Contents**:
- How local and user skills work together
- Conflict resolution rules
- Recommended user skills for fairseq2
- Testing skill integration
- Creating custom compatible skills

### `.claude/skills/SKILL.md`

**Audience**: Claude Code (the AI)

**Contents**:
- Complete fairseq2 development rules
- Source code verification workflow
- Forbidden directories
- Preferred constructs
- Example patterns
- Checklists for code and documentation

## Usage Examples

### Starting a New Session

```bash
cd /path/to/fairseq2
claude code
```

**What happens**:
1. ✅ `.claude/hooks.json` triggers `session:start`
2. ✅ `SKILL` skill loads automatically
3. ✅ Message: "📚 fairseq2 repository loaded..."
4. ✅ All rules immediately in effect

### Compacting Conversation History

```bash
/compact
```

**What happens**:
1. ✅ History compressed
2. ✅ `.claude/hooks.json` triggers `command:compact`
3. ✅ `SKILL` skill re-loads
4. ✅ Message: "♻️ Re-loading fairseq2 skill..."
5. ✅ Rules restored to context

### Using User Skills

```bash
# Use TDD workflow
/test-driven-development

# Use debugging workflow
/systematic-debugging

# Request code review
/requesting-code-review
```

**What happens**:
1. ✅ User skill loads and provides workflow
2. ✅ Local `SKILL` skill constraints apply
3. ✅ User skill adapts to local rules
4. ✅ Result: Workflow + fairseq2 correctness

### Writing Code

When Claude writes code in this repo:

```
Before writing code:
├── Check: Did I read src/fairseq2/[module].py?
├── Check: Did I verify function signatures?
├── Check: Did I check git history?
└── If NO to any → STOP and read source first

While writing code:
├── Import from installed library: ✅ from fairseq2.X import Y
├── NOT from local paths: ❌ from src.fairseq2.X import Y
├── Prefer fairseq2 constructs: ✅ fairseq2.gang
├── NOT raw PyTorch: ❌ torch.distributed (unless necessary)
├── Consult tests/: ✅ For usage examples
└── NEVER consult recipes/ or doc/: ❌ Forbidden
```

## Maintenance

### Adding New Skills

To add another local skill:

1. Create `.claude/skills/new-skill.md`
2. Update `.claude/settings.json`:
   ```json
   "autoLoad": ["SKILL", "new-skill"]
   ```
3. Optionally add hook in `.claude/hooks.json`
4. Commit to git

### Updating Existing Skill

1. Edit `.claude/skills/SKILL.md`
2. Commit changes
3. Next session: auto-loads updated version
4. After `/compact`: auto-loads updated version

### User-Specific Customization

1. Copy `.claude/settings.local.json.template` to `.claude/settings.local.json`
2. Add your custom permissions, environment variables, etc.
3. This file is gitignored - won't conflict with team settings

## Benefits Summary

### For Individual Users

✅ **No manual setup** - everything automatic
✅ **Persistent context** - survives `/compact`
✅ **User skills still work** - alongside local skill
✅ **Clear guidance** - always know the rules

### For Teams

✅ **Consistent behavior** - all users get same rules
✅ **Version controlled** - skills and settings in git
✅ **No conflicts** - local settings isolated
✅ **Self-documenting** - README explains everything

### For fairseq2 Development

✅ **Correct APIs** - always verified against source
✅ **No outdated info** - forbidden directories enforced
✅ **Preferred patterns** - fairseq2 constructs over PyTorch
✅ **Rapid changes handled** - source code is truth

## Next Steps

### For Users

1. **Start using** - just open Claude Code, skill auto-loads
2. **Read documentation** - `.claude/README.md` for details
3. **Customize locally** - `.claude/settings.local.json` for personal settings
4. **Use your skills** - 10x-engineer suite and others still work

### For Team

1. **Test the setup** - verify auto-loading works
2. **Add more skills** - if needed for specific workflows
3. **Share knowledge** - point team members to `.claude/README.md`
4. **Iterate** - update skills as fairseq2 evolves

## Troubleshooting

### Skill Not Auto-Loading

**Check**:
1. `.claude/hooks.json` exists and is valid JSON
2. `.claude/settings.json` has correct `autoLoad` array
3. Try manual load: `/doc-code-author`

### Permissions Blocked

**Solution**:
1. Add to `.claude/settings.local.json`:
   ```json
   {"permissions": {"allow": ["Bash(your_command:*)"]}}
   ```

### Conflicts with User Skills

**Remember**:
- Local skill ALWAYS wins
- User skills adapt automatically
- This is by design for fairseq2 correctness

### After Git Pull

**If hooks/settings updated**:
1. Restart Claude Code
2. Skill will auto-load with new version

## Summary

This setup provides:

🎯 **Automatic skill loading** - no manual steps
🎯 **Persistent context** - survives compression
🎯 **Clear priorities** - local skill is source of truth
🎯 **User compatibility** - personal skills still work
🎯 **Team consistency** - everyone gets same behavior
🎯 **Self-documenting** - comprehensive README files

**Result**: Working with fairseq2 is automatic and correct. You shouldn't need to think about guidelines - Claude Code handles it automatically through the skill system.
