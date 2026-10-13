"""The package's shape, which no other test sees.

lab/ is layered, each layer a folder built on the ones under it and never on one above, and the package's root, what
the command line and the page drive, on top of them all:

  foundation/  paths and the firm's target (config), the log, which code made a saved result (provenance)
  data/        the bar store and how it is read: bars, calendars, instruments, minutes, spreads, rates, the market's
               index, when an instrument stopped trading, the research lists and their point-in-time members
  signals/     indicators, regimes, the models' features and the work a grid's configurations share
  engine/      the backtest, its costs, buy & hold, the trade ledger, how a strategy is written, the trade model
  evaluation/  the scorecard, walk-forward, significance, Monte Carlo, robustness, an evaluation and the database it
               is saved to, each instrument alone, the ML task's picks, the research board
  upkeep/      bringing the store up to date from the vendors, and checking it
  the root     the re-runs, the batches, the catalog, forward testing, portfolios, the command line

A module's layer is the folder it is written in. Beside the package, backend/ (the page's server), scripts/,
research/ and strategies/ build on it; it builds on none of them.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
PACKAGE = "lab"
# The layers, bottom up; the package's root is above them all.
LAYERS = ("foundation", "data", "signals", "engine", "evaluation", "upkeep")
TOP = "the root"
RANK = {layer: number for number, layer in enumerate(LAYERS + (TOP,))}
# Imported where they are used: a model's library, which only the rules with a model need, and Databento's client,
# which only its own refreshes do.
IMPORTED_WHERE_USED = {"lightgbm", "databento"}
# What the package builds on nothing of: the folders beside it build on it.
BESIDE = ("backend", "scripts", "research", "strategies", "tests")


def _modules() -> dict[str, tuple[str, ast.Module]]:
    """Every module of the package by its dotted name in it (data.bars, runs): its layer and its parsed source."""
    found = {}
    for path in sorted((ROOT_DIR / PACKAGE).rglob("*.py")):
        if "__pycache__" in path.parts or path.name == "__init__.py":
            continue
        parts = path.relative_to(ROOT_DIR / PACKAGE).with_suffix("").parts
        layer = TOP if len(parts) == 1 else parts[0]
        found[".".join(parts)] = (layer if len(parts) <= 2 else f"{parts[0]}/{parts[1]}", ast.parse(path.read_text()))
    return found


def _imports(tree: ast.Module) -> list[tuple[ast.stmt, list[str]]]:
    """Each import statement and the dotted names it brings in (`from lab.data import bars` brings in lab.data.bars
    and lab.data: the module, or the name of one of its objects)."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.append((node, [alias.name for alias in node.names]))
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.append((node, [f"{node.module}.{alias.name}" for alias in node.names] + [node.module]))
    return out


def _ours(dotted: str, modules: dict) -> str | None:
    """The package's module a dotted name is, or is a name in; None for anything else."""
    parts = dotted.split(".")
    if parts[0] != PACKAGE:
        return None
    for k in range(len(parts) - 1, 0, -1):
        name = ".".join(parts[1:k + 1])
        if name in modules:
            return name
    return None


def _graph(modules: dict) -> dict[str, set[str]]:
    return {name: {target for _, names in _imports(tree) for dotted in names
                   if (target := _ours(dotted, modules)) and target != name}
            for name, (_, tree) in modules.items()}


def test_every_module_is_in_a_layer():
    astray = sorted(f"{name} ({layer})" for name, (layer, _) in _modules().items() if layer not in RANK)
    assert not astray, f"every module is in a layer's folder ({', '.join(LAYERS)}) or the package's root: {astray}"


def test_no_module_imports_a_layer_above_its_own():
    modules = _modules()
    layer = {name: found for name, (found, _) in modules.items()}
    upward = sorted(f"{name} ({layer[name]}) imports {target} ({layer[target]})"
                    for name, targets in _graph(modules).items() for target in targets
                    if RANK[layer[target]] > RANK[layer[name]])
    assert not upward, f"a module imports only its own layer and those under it: {upward}"


def test_no_modules_import_each_other_in_a_circle():
    graph = _graph(_modules())
    index, low, stack, on_stack, circles = {}, {}, [], set(), []

    def visit(node):
        index[node] = low[node] = len(index)
        stack.append(node)
        on_stack.add(node)
        for target in sorted(graph[node]):
            if target not in index:
                visit(target)
                low[node] = min(low[node], low[target])
            elif target in on_stack:
                low[node] = min(low[node], index[target])
        if low[node] == index[node]:
            circle = []
            while True:
                member = stack.pop()
                on_stack.discard(member)
                circle.append(member)
                if member == node:
                    break
            if len(circle) > 1:
                circles.append(sorted(circle))

    for node in sorted(graph):
        if node not in index:
            visit(node)
    assert not circles, f"no modules import each other in a circle, directly or through others: {circles}"


def test_a_module_s_imports_are_at_its_top():
    inside = []
    for name, (_, tree) in _modules().items():
        top = {id(node) for node in tree.body}
        for node, names in _imports(tree):
            if id(node) in top:
                continue
            roots = {dotted.split(".")[0] for dotted in names}
            if not roots <= IMPORTED_WHERE_USED:
                inside.append(f"{name}:{node.lineno}")
    assert not inside, f"every import is at its module's top, but {sorted(IMPORTED_WHERE_USED)}: {inside}"


def test_the_package_builds_on_nothing_beside_it():
    leaning = sorted(f"{name} imports {dotted}" for name, (_, tree) in _modules().items()
                     for _, names in _imports(tree) for dotted in names if dotted.split(".")[0] in BESIDE)
    assert not leaning, f"lab imports nothing of {', '.join(BESIDE)}: {leaning}"
