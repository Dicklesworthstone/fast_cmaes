"""Rich-powered TUI demo for fastcma.

Run with uv using Python 3.13:

    uv venv --python 3.13
    uv pip install .[demo]
    uv run python examples/rich_tui_demo.py

The demo minimizes the 2D Rosenbrock function and streams live metrics.
"""

from __future__ import annotations

import math
import time
from typing import List

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

import fastcma


def rosenbrock(x: List[float]) -> float:
    return 100.0 * (x[0] * x[0] - x[1]) ** 2 + (x[0] - 1.0) ** 2


def main():
    console = Console()

    es = fastcma.CMAES([ -1.2, 1.0 ], 0.5, ftarget=1e-10, maxfevals=20_000)

    progress = Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn("{task.description}"),
        TextColumn("evals: {task.completed}", justify="right"),
        TextColumn("sigma: {task.fields[sigma]:.3g}"),
        TextColumn("fbest: {task.fields[fbest]:.3g}"),
        TimeElapsedColumn(),
        refresh_per_second=10,
    )
    task_id = progress.add_task("Optimizing Rosenbrock", total=None, sigma=es.result[5][0], fbest=float("inf"))

    def stats_panel(iteration: int, fbest: float, xbest: List[float]):
        table = Table.grid(padding=1)
        table.add_column(justify="right", style="bold cyan")
        table.add_column(justify="left")
        table.add_row("Iteration", f"{iteration}")
        table.add_row("Best f", f"{fbest:.3e}")
        table.add_row("x_best", f"[{xbest[0]:.3f}, {xbest[1]:.3f}]")
        return Panel(table, title="Live Stats", box=box.ROUNDED, border_style="cyan")

    def legend_panel():
        txt = Text()
        txt.append("• Minimizing Rosenbrock f(x,y) = 100(x²−y)² + (x−1)²\n", style="bold")
        txt.append("• Full covariance CMA-ES, sigma adapts automatically\n")
        txt.append("• Stop when f <= 1e-10 or 20k evals\n")
        txt.append("• Compare with naive Python baseline in examples/python_benchmarks.py\n", style="dim")
        return Panel(txt, title="About", box=box.SQUARE)

    layout = lambda iteration, fbest, xbest: Group(
        Align.center(legend_panel()),
        Align.center(stats_panel(iteration, fbest, xbest)),
        progress,
    )

    with Live(layout(0, float("inf"), [math.nan, math.nan]), console=console, refresh_per_second=10):
        iteration = 0
        while True:
            X = es.ask()
            fitvals = [rosenbrock(x) for x in X]
            es.tell(X, fitvals)
            iteration += 1

            xbest, fbest, _, counteval, _iters, xmean, stds = es.result
            progress.update(task_id, completed=counteval, sigma=stds[0], fbest=fbest)
            Live.current.set_renderable(layout(iteration, fbest, xbest))

            if es.stop():
                break
            time.sleep(0.02)  # slow down for readability

    console.print(Panel(f"Converged with f={fbest:.3e} at x={xbest}", border_style="green"))


if __name__ == "__main__":
    main()
