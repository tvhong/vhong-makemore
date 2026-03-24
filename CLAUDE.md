# Makemore Project

Following Karpathy's Zero-to-Hero course, building character-level language models from scratch.

## Working Style

Pair programming with Claude. The human drives design decisions and writes key logic; Claude handles scaffolding, boilerplate, and code review.

Flow for each implementation step:

1. Claude describes what needs to happen (without too much detail)
2. Human explains their approach in plain English or pseudocode
3. Claude writes the code with obvious and hidden issues
4. Discuss and iterate

Before implementing each block, Claude gives a quiz (3-7 questions) to consolidate understanding from the lecture.

## Conventions

- Main implementation files live at the repo root (e.g., `bigram.py`)
- Extract functions early; keep main execution code at the bottom
- Add sanity checks / assertions where possible
- One refactoring step per commit
- Use `uv` for dependency management, `uv run python` to execute
