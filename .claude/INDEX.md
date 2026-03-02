# .claude Directory - Index

Welcome to the fairseq2 repository's Claude Code configuration!

## 📖 Documentation Index

Choose the right document based on what you need:

### 🚀 Getting Started

**[QUICKSTART.md](QUICKSTART.md)** - Start here!
- What happens automatically when you open Claude Code
- How the skill works behind the scenes
- Quick examples of what to expect
- Common pitfalls that are automatically avoided

👉 **Read this first if you're new to this repository.**

### 📚 Complete Setup Details

**[README.md](README.md)** - Full documentation
- Complete directory structure explanation
- Settings files and their purposes
- Hooks and automatic loading
- Skill priority system
- Customization guide
- Troubleshooting

👉 **Read this to understand the complete setup.**

### 🤝 Using User Skills

**[SKILL_INTEGRATION.md](SKILL_INTEGRATION.md)** - Skill integration guide
- How local and user skills work together
- Priority hierarchy
- Recommended user skills for fairseq2
- Conflict resolution
- Creating compatible custom skills

👉 **Read this if you use the 10x-engineer suite or other personal skills.**

### 📋 Comprehensive Reference

**[SETUP_SUMMARY.md](SETUP_SUMMARY.md)** - Everything in one place
- What was created and why
- How automatic loading works
- Configuration files explained
- Usage examples
- Maintenance guide
- Benefits summary

👉 **Read this for a complete reference.**

### 📦 Template Installation

**[TEMPLATE_INSTALLATION.md](TEMPLATE_INSTALLATION.md)** - Publishing as a template component
- How the skill was made template-compatible
- Publishing to organization template repositories
- Installing from templates in other projects
- Dual usage: local auto-loading + template installation

👉 **Read this if publishing to an organization template repository.**

## 🎯 The Core Skill

**[skills/SKILL.md](skills/SKILL.md)** - The actual skill
- fairseq2-specific development rules
- Source code verification workflow
- Forbidden directories
- Preferred constructs
- Example patterns
- Checklists

👉 **This is what Claude reads** (automatically loaded, no need to read manually).

## ⚙️ Configuration Files

**[settings.json](settings.json)** - Repository-wide settings
- Skill loading configuration
- Auto-load specifications
- Baseline permissions
- Context always includes

👉 **Committed to git** - team-wide settings.

**settings.local.json** - Your personal settings
- User-specific overrides
- Custom permissions
- Personal environment

👉 **Not committed to git** - your personal customizations.

**[settings.local.json.template](settings.local.json.template)** - Template for local settings
- Example structure
- Common customizations

👉 **Copy and rename to `settings.local.json`** to customize.

**[hooks.json](hooks.json)** - Automatic behaviors
- Session start: Load skill
- After `/compact`: Reload skill
- Before code: Reminder to verify

👉 **Committed to git** - defines automatic loading.

## 🗂️ Directory Structure

```
.claude/
├── INDEX.md                          ← You are here
│
├── QUICKSTART.md                     ← Start here (new users)
├── README.md                         ← Full documentation
├── SKILL_INTEGRATION.md              ← User skill compatibility
├── SETUP_SUMMARY.md                  ← Complete reference
├── TEMPLATE_INSTALLATION.md          ← Publishing to template repos
│
├── settings.json                     ← Repo settings (committed)
├── settings.local.json               ← User settings (gitignored)
├── settings.local.json.template      ← Template to copy
├── hooks.json                        ← Auto-loading (committed)
│
└── skills/
    ├── SKILL.md                      ← The core skill (auto-loaded)
    └── component.yaml                ← Template component metadata
```

## 🎓 Learning Path

### If you're new to this repository:

1. **Read**: [QUICKSTART.md](QUICKSTART.md)
2. **Understand**: The skill loads automatically, you don't need to do anything
3. **Use**: Just start working - Claude knows the rules

### If you want to understand the setup:

1. **Read**: [README.md](README.md)
2. **Explore**: Configuration files ([settings.json](settings.json), [hooks.json](hooks.json))
3. **Customize**: Copy [settings.local.json.template](settings.local.json.template) if needed

### If you use personal skills:

1. **Read**: [SKILL_INTEGRATION.md](SKILL_INTEGRATION.md)
2. **Understand**: Priority hierarchy (local wins, user adapts)
3. **Test**: Use your skills normally - they'll respect local rules

### If you want complete details:

1. **Read**: [SETUP_SUMMARY.md](SETUP_SUMMARY.md)
2. **Reference**: Configuration files and skill
3. **Maintain**: Update as needed

### If you want to publish to template repository:

1. **Read**: [TEMPLATE_INSTALLATION.md](TEMPLATE_INSTALLATION.md)
2. **Understand**: SKILL.md naming and component.yaml metadata
3. **Publish**: Copy to organization template repository

## 🔍 Quick Answers

### "Does the skill load automatically?"
✅ **Yes** - at session start and after `/compact`. See [QUICKSTART.md](QUICKSTART.md).

### "Can I use my 10x-engineer skills?"
✅ **Yes** - they work alongside the local skill. See [SKILL_INTEGRATION.md](SKILL_INTEGRATION.md).

### "Where do I add my custom permissions?"
👉 Copy [settings.local.json.template](settings.local.json.template) to `settings.local.json`. See [README.md](README.md).

### "What happens when I run /compact?"
✅ Skill automatically re-loads. See [hooks.json](hooks.json) or [QUICKSTART.md](QUICKSTART.md).

### "What are the fairseq2 rules?"
👉 Read [skills/SKILL.md](skills/SKILL.md) (but Claude already knows them).

### "How do I troubleshoot?"
👉 See troubleshooting section in [README.md](README.md).

### "How do I publish this to my organization's template repository?"
👉 See [TEMPLATE_INSTALLATION.md](TEMPLATE_INSTALLATION.md) for step-by-step instructions.

## 📞 Quick Reference Card

| Need | Read | Time |
|------|------|------|
| Quick overview | [QUICKSTART.md](QUICKSTART.md) | 5 min |
| Full setup guide | [README.md](README.md) | 15 min |
| User skill integration | [SKILL_INTEGRATION.md](SKILL_INTEGRATION.md) | 10 min |
| Complete reference | [SETUP_SUMMARY.md](SETUP_SUMMARY.md) | 20 min |
| Template installation | [TEMPLATE_INSTALLATION.md](TEMPLATE_INSTALLATION.md) | 10 min |
| The actual skill | [skills/SKILL.md](skills/SKILL.md) | 15 min |

## 🎯 Key Principle

**Everything is automatic. You don't need to do anything.**

The skill:
- ✅ Loads when you open Claude Code
- ✅ Re-loads after `/compact`
- ✅ Ensures source code verification
- ✅ Prevents outdated API usage
- ✅ Works with your user skills

You can focus on your work - Claude Code handles correctness automatically.

---

**Start with [QUICKSTART.md](QUICKSTART.md) if you're new here!**
