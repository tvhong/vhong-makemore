# Marimo notebook assistant

When creating or editing marimo notebooks, follow these rules.

## File format

Marimo notebooks are pure Python files. Only edit contents inside `@app.cell` decorators. Marimo handles function parameters and return statements automatically.

```python
@app.cell
def _():
    <your code here>
    return
```

## Fundamentals

- Cells execute automatically when their dependencies change (reactive)
- Variables cannot be redeclared across cells
- The notebook forms a directed acyclic graph (DAG)
- The last expression in a cell is automatically displayed
- UI elements are reactive and update downstream cells automatically
- Variables prefixed with underscore (e.g. `_my_var`) are local to the cell

## Code rules

1. All code must be complete and runnable
2. Import all modules in the first cell, always including `import marimo as mo`
3. Never redeclare variables across cells
4. Ensure no cycles in the dependency graph
5. Never use `global`
6. No comments in markdown or SQL cells

## Reactivity

- When a variable changes, all cells using it re-execute
- UI element values accessed via `.value` attribute
- Cannot access a UI element's value in the same cell where it's defined

## Visualization

- For matplotlib: use `plt.gca()` as the last expression instead of `plt.show()`
- For plotly: return the figure object directly
- For altair: return the chart object directly

## Layout and display

- `mo.md(text)` — display markdown
- `mo.hstack(elements)` — stack horizontally
- `mo.vstack(elements)` — stack vertically
- `mo.tabs(elements)` — tabbed interface
- `mo.stop(predicate, output)` — stop execution conditionally

## Common UI elements

- `mo.ui.slider(start, stop, value, label)`
- `mo.ui.dropdown(options, value, label)`
- `mo.ui.checkbox(label, value)`
- `mo.ui.text(value, label)`
- `mo.ui.button(value, kind)`
- `mo.ui.table(data)`
- `mo.ui.altair_chart(chart)`
- `mo.ui.plotly(figure)`

## Linting

After generating or editing a notebook, run `marimo check --fix` to catch and resolve common issues.

## Watch mode

This project uses `marimo edit --watch` so the user edits in vim while viewing outputs in the browser. The `pyproject.toml` has `watcher_on_save = "autorun"` so cells re-run on file save.
