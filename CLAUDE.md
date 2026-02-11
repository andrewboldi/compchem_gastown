# Refinery Context (compchem_gastown)

> **Recovery**: Run `gt prime` after compaction, clear, or new session

## Rig Identity

- **Name**: compchem_gastown
- **Prefix**: cg
- **Type**: refinery
- **Repository**: https://github.com/andrewboldi/compchem_gastown
- **Default Branch**: main

## Quick Reference

- Check MQ: `gt mq list`
- Process next: `gt mq process`
- Find work: `bd ready`
- Show issue: `bd show <id>`

## Refinery Protocol

1. **Startup**: Run `gt prime` to inject full context
2. **Patrol**: Check hook with `gt mol status`, execute if work present
3. **Mail**: Check `gt mail inbox` for assignments
4. **Work**: Use `bd ready` to find available issues
5. **Complete**: Stage, commit, push, then `gt done`

## Directory Structure

```
refinery/rig/
├── .beads/          # Beads database redirect
├── .opencode/       # Opencode configuration
├── witness/         # Witness outputs
├── .gitignore       # Worktree exclusions
├── CLAUDE.md        # This file
└── LICENSE          # Project license
```

## Important Notes

- This is a **git worktree** - commits go to the main repo
- The `.beads/` directory is managed by the parent rig
- Use standard git workflow: add → commit → push → gt done
