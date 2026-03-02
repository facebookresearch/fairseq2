# `.claude/` Directory Structure

This directory contains Claude Code configuration and skills for the fairseq2 repository.

## Directory Structure

```
.claude/
├── README.md                    # This file
├── settings.json                # Repository-wide Claude Code settings
├── settings.local.json          # User-specific local settings (gitignored)
├── hooks.json                   # Hook definitions for automatic skill loading
└── skills/
    └── SKILL.md                 # Primary skill for fairseq2 development
```

## Skills Priority System

### Local Repository Skill (Highest Priority)

The **`SKILL.md`** file in `.claude/skills/` is the **ABSOLUTE SOURCE OF TRUTH** for working with this repository:

- ✅ **Automatically loaded** when Claude Code starts
- ✅ **Automatically re-loaded** after `/compact` command
- ✅ **Highest priority** - overrides any conflicting guidance from user skills
- ✅ **Repository-specific** - contains fairseq2-specific rules and workflows

**Core principles enforced by this skill:**
- Source code in `src/fairseq2/` is the only reliable truth
- Never consult `recipes/` or `doc/` directories (except when explicitly modifying docs)
- Always verify APIs against source before writing code
- Prefer fairseq2 constructs over raw PyTorch

### User Skills (Lower Priority)

Users can have their own skills in `~/.claude/skills/` or other locations. These skills:

- ✅ **Still available** - can be invoked using `/skill-name` or the Skill tool
- ⚠️ **Lower priority** - if conflicts arise, the local `SKILL` skill wins
- ✅ **Complementary** - can add additional capabilities (e.g., debugging workflows, testing patterns)

**Example user skills that complement `SKILL`:**
- `10x-engineer:systematic-debugging` - debugging workflows
- `10x-engineer:test-driven-development` - TDD methodology
- Custom skills for code review, performance analysis, etc.

### Skill Integration Rules

When both local and user skills apply:

1. **Local skill rules are ABSOLUTE** - cannot be overridden
   - Source code verification requirement
   - Forbidden directories (recipes/, doc/)
   - Import patterns

2. **User skills add value** - workflow and methodology
   - How to approach debugging
   - Testing strategies
   - Code review processes

3. **User skills adapt** - to local skill constraints
   - User debugging skill should still verify against `src/fairseq2/`
   - User TDD skill should still use fairseq2 constructs
   - User skills respect the forbidden directories

## Configuration Files

### `settings.json` (Repository Settings)

Defines repository-wide Claude Code behavior:
- Skill loading order (local first, then user)
- Auto-load the `SKILL` skill
- Baseline permissions for common commands
- Always include CLAUDE.md and the skill in context

**Committed to git** - shared across all users of the repository.

### `settings.local.json` (User-Specific Settings)

Your personal overrides and extensions:
- Additional permissions specific to your environment
- Custom environment variables
- Personal preferences

**Not committed to git** - each user maintains their own.

**Example:**
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

### `hooks.json` (Automatic Behaviors)

Defines when skills are loaded and reminders are shown:

- **`session:start`**: Load `SKILL` when Claude starts
- **`command:compact`**: Re-load skill after `/compact` to restore context
- **`before:code`**: Reminder to verify against source code

## How to Use This Setup

### Starting a Session

When you open Claude Code in this repository:

1. ✅ The fairseq2 skill automatically loads
2. ✅ You'll see: "📚 fairseq2 repository loaded. fairseq2 development skill is active."
3. ✅ All rules from the skill are immediately in effect

### After `/compact`

When you run `/compact` to compress conversation history:

1. ✅ The skill automatically re-loads
2. ✅ You'll see: "♻️ Re-loading fairseq2 skill after /compact..."
3. ✅ All rules are restored to context

### Using User Skills Alongside Local Skill

You can still use your personal skills (e.g., from `10x-engineer` suite):

```bash
# Use a user skill for debugging
/systematic-debugging

# Use a user skill for TDD
/test-driven-development
```

**Important:** These user skills will automatically adapt to respect the local skill's constraints:
- They'll still verify against `src/fairseq2/` source code
- They'll still avoid forbidden directories
- They'll still use fairseq2 constructs

### Invoking the Local Skill Manually

If you ever need to manually re-load the skill:

```bash
# Invoke the skill (though auto-loading usually makes this unnecessary)
# The skill name when invoking is the directory name, not the filename
```

However, this is rarely needed since:
- Auto-loaded at session start
- Auto-loaded after `/compact`
- Always included in context via `settings.json`

## Extending This Setup

### Adding More Local Skills

To add additional repository-specific skills:

1. Create a new skill file: `.claude/skills/my-skill.md`
2. Add it to auto-load in `settings.json`:
   ```json
   "skills": {
     "local": {
       "autoLoad": ["SKILL", "my-skill"]
     }
   }
   ```
3. Optionally add a hook in `hooks.json` if it should load at specific events

### Customizing Hooks

Edit `.claude/hooks.json` to add custom behaviors:

```json
{
  "hooks": {
    "before:test": {
      "description": "Reminder before running tests",
      "command": "echo 'Running tests against src/fairseq2/...'"
    }
  }
}
```

### Sharing Settings Across Team

Settings in `.claude/settings.json` and `.claude/hooks.json` are committed to git.

**Best practice:**
- Commit: Repository-wide skills, hooks, baseline permissions
- Don't commit: User-specific settings (use `.local.json` files)

## Troubleshooting

### Skill Not Loading

If the skill doesn't auto-load:

1. Check `.claude/hooks.json` exists and is valid JSON
2. Check `.claude/settings.json` has correct skill path
3. Restart Claude Code
4. Manually invoke: `/doc-code-author`

### Conflicts Between User and Local Skills

If user skills conflict with local skill rules:

1. **Local skill always wins** - this is by design
2. User skills should adapt to respect local constraints
3. If conflicts persist, consider updating user skill to check for local skills

### Permission Errors

If commands are blocked:

1. Check `.claude/settings.json` permissions
2. Add needed permissions to `.claude/settings.local.json`
3. Common additions:
   ```json
   {
     "permissions": {
       "allow": [
         "Bash(your_command:*)"
       ]
     }
   }
   ```

## Summary

This setup ensures:

✅ **Automatic skill loading** - no manual steps needed
✅ **Persistent context** - skill survives `/compact`
✅ **Clear priority** - local skill is the source of truth
✅ **User compatibility** - user skills still work, just adapt to local rules
✅ **Team consistency** - all users get the same baseline behavior

The goal is to make working with fairseq2 automatic and correct - you shouldn't need to think about whether you're following the right guidelines, Claude Code handles it automatically.
