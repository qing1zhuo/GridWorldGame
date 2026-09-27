"""Visualizations for PPO training, learned policy, and exact baseline.

The filename intentionally follows the spelling requested by this project.
All plotting functions return their Matplotlib objects and optionally save a
static figure, so they work both in ``main.ipynb`` and in scripts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np

from evaluate import deterministic_actor_policy


# Okabe-Ito colors plus labels/line styles so color is never the only cue.
COLORS = {
    "actor_loss": "#0072B2",
    "critic_loss": "#D55E00",
    "mean_reward": "#009E73",
    "optimal_action_fraction": "#CC79A7",
}
FORBIDDEN_COLOR = "#E69F00"
TARGET_COLOR = "#56B4E9"
GRID_COLOR = "#333333"
MISMATCH_COLOR = "#B2182B"
ACTION_SYMBOLS = np.asarray(["↑", "→", "↓", "←", "○"])


def _save_figure(fig, save_path, dpi=180):
    if save_path is None:
        return
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor="white")


def _history_arrays(history):
    if isinstance(history, Mapping):
        arrays = {str(key): np.asarray(value, dtype=np.float64)
                  for key, value in history.items()}
    elif isinstance(history, Sequence) and history and isinstance(history[0], Mapping):
        common = set(history[0])
        for record in history[1:]:
            common.intersection_update(record)
        arrays = {
            str(key): np.asarray([record[key] for record in history], dtype=np.float64)
            for key in common
        }
    else:
        raise ValueError("history must be a metric mapping or a non-empty list of mappings")
    if not arrays:
        raise ValueError("history contains no common metrics")
    for key, values in arrays.items():
        if values.ndim != 1 or values.size == 0:
            raise ValueError(f"history metric {key!r} must be a non-empty 1D sequence")
        if not np.isfinite(values).all():
            raise ValueError(f"history metric {key!r} contains a non-finite value")
    return arrays


def _raw_and_mean(ax, x, values, label, color, smoothing_window):
    ax.plot(x, values, color=color, linewidth=0.8, alpha=0.28,
            label=f"{label} (raw)")
    window = min(max(1, int(smoothing_window)), values.size)
    if window > 1:
        mean = np.convolve(values, np.ones(window) / window, mode="valid")
        ax.plot(x[window - 1:], mean, color=color, linewidth=1.8,
                label=f"{label} ({window}-point mean)")


def plot_training_curve(history, save_path=None, show=True,
                        smoothing_window=20, dpi=180):
    """Plot raw PPO losses, sampled reward, and exact evaluation history.

    Different quantities use separate axes.  Smoothing is an explicitly
    labelled moving mean; the raw observations remain visible underneath.
    """
    data = _history_arrays(history)
    panels = []
    if "actor_loss" in data or "critic_loss" in data:
        panels.append(("Optimization losses", "Loss",
                       [k for k in ("actor_loss", "critic_loss") if k in data]))
    if "mean_reward" in data:
        panels.append(("Sampled transitions", "Mean immediate reward", ["mean_reward"]))
    if "optimal_action_fraction" in data:
        panels.append(("Deterministic policy evaluation", "Optimal actions (%)",
                       ["optimal_action_fraction"]))
    if not panels:
        raise ValueError(
            "history needs actor_loss, critic_loss, mean_reward, or "
            "optimal_action_fraction"
        )

    x = data.get("rollout", np.arange(1, len(data[panels[0][2][0]]) + 1))
    with plt.rc_context({
        "font.size": 10,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }):
        fig, axes = plt.subplots(
            len(panels), 1, figsize=(7.8, 2.8 * len(panels)), sharex=True,
            squeeze=False, layout="constrained"
        )
        axes = axes[:, 0]
        for ax, (title, ylabel, keys) in zip(axes, panels):
            for key in keys:
                values = data[key] * 100.0 if key == "optimal_action_fraction" else data[key]
                _raw_and_mean(
                    ax, x, values, key.replace("_", " ").title(),
                    COLORS.get(key, "#333333"), smoothing_window
                )
            ax.set(title=title, ylabel=ylabel)
            if keys == ["optimal_action_fraction"]:
                ax.set_ylim(-2, 102)
                ax.axhline(100, color="#444444", linewidth=1, linestyle=":",
                           label="100% baseline match")
            ax.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.75)
            ax.legend(frameon=False, ncols=2)
        axes[-1].set_xlabel("Rollout iteration")

    _save_figure(fig, save_path, dpi)
    if show:
        plt.show()
    return fig, axes


def plot_evaluation_curve(evaluations, save_path=None, show=True, dpi=180):
    """Plot every exact baseline comparison point without smoothing."""
    data = _history_arrays(evaluations)
    required = {"rollout", "optimal_action_fraction", "max_absolute_value_gap"}
    missing = required.difference(data)
    if missing:
        raise ValueError(f"evaluations is missing metrics: {sorted(missing)}")
    lengths = {data[key].size for key in required}
    if len(lengths) != 1:
        raise ValueError("evaluation metrics must have equal lengths")

    x = data["rollout"]
    with plt.rc_context({
        "font.size": 10,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }):
        fig, axes = plt.subplots(2, 1, figsize=(7.8, 5.8), sharex=True,
                                 layout="constrained")
        axes[0].plot(x, 100 * data["optimal_action_fraction"], marker="o",
                     markersize=3, color=COLORS["optimal_action_fraction"])
        axes[0].axhline(100, color="#444444", linestyle=":", linewidth=1)
        axes[0].set(title="Exact deterministic-policy evaluation",
                    ylabel="Optimal actions (%)", ylim=(-2, 102))
        axes[1].plot(x, data["max_absolute_value_gap"], marker="s",
                     markersize=3, color="#0072B2")
        axes[1].set_yscale("symlog", linthresh=1e-6)
        axes[1].set(xlabel="Rollout iteration",
                    ylabel="max |V* − Vπ|\n(symmetric log scale)")
        for ax in axes:
            ax.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.75)

    _save_figure(fig, save_path, dpi)
    if show:
        plt.show()
    return fig, axes


def compute_greedy_policy(runner, cfg):
    """Return Actor argmax policy, action probabilities, and Critic values."""
    return deterministic_actor_policy(runner, cfg)


def _cell_type(cfg, row, col):
    return cfg.idx2grid[int(np.asarray(cfg.grid)[row, col])]


def _draw_cells(ax, cfg):
    for row in range(cfg.row_num):
        for col in range(cfg.col_num):
            cell_name = _cell_type(cfg, row, col)
            color = {"forbidden": FORBIDDEN_COLOR, "target": TARGET_COLOR}.get(
                cell_name, "white"
            )
            ax.add_patch(Rectangle((col, row), 1, 1, facecolor=color,
                                   edgecolor=GRID_COLOR, linewidth=1.0))
            label = {"forbidden": "F", "target": "T"}.get(cell_name)
            if label:
                ax.text(col + 0.10, row + 0.18, label, ha="left", va="center",
                        fontsize=9, fontweight="bold", color="#111111")


def _format_grid(ax, cfg, title):
    ax.set_xlim(0, cfg.col_num)
    ax.set_ylim(cfg.row_num, 0)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(cfg.col_num) + 0.5, labels=np.arange(1, cfg.col_num + 1))
    ax.set_yticks(np.arange(cfg.row_num) + 0.5, labels=np.arange(1, cfg.row_num + 1))
    ax.tick_params(length=0)
    ax.set(title=title, xlabel="Column", ylabel="Row")


def _draw_action(ax, action, row, col, alpha=1.0):
    center_x, center_y = col + 0.5, row + 0.5
    directions = {0: (0.0, -0.28), 1: (0.28, 0.0),
                  2: (0.0, 0.28), 3: (-0.28, 0.0)}
    if int(action) in directions:
        dx, dy = directions[int(action)]
        ax.arrow(center_x, center_y, dx, dy, width=0.022,
                 head_width=0.14, head_length=0.11,
                 length_includes_head=True, color="#222222", alpha=alpha)
    elif int(action) == 4:
        ax.plot(center_x, center_y, marker="o", markersize=7,
                markerfacecolor="none", markeredgecolor="#222222",
                markeredgewidth=1.5, alpha=alpha)


def _value_norm(values, mask=None):
    selected = np.asarray(values, dtype=np.float64)
    if mask is not None:
        selected = selected[np.asarray(mask, dtype=bool)]
    finite = selected[np.isfinite(selected)]
    if finite.size == 0:
        raise ValueError("value array contains no finite values")
    low, high = float(finite.min()), float(finite.max())
    if low < 0 < high:
        magnitude = max(abs(low), abs(high))
        return mpl.colors.TwoSlopeNorm(vmin=-magnitude, vcenter=0, vmax=magnitude), "RdBu_r"
    if np.isclose(low, high):
        low, high = low - 0.5, high + 0.5
    return mpl.colors.Normalize(vmin=low, vmax=high), "cividis"


def plot_policy(runner, cfg, save_path=None, show=True, dpi=180):
    """Plot the learned deterministic Actor policy and Critic value surface."""
    policy, probabilities, critic_values = compute_greedy_policy(runner, cfg)
    terminal = np.asarray(cfg.grid) == next(
        key for key, name in cfg.idx2grid.items() if name == "target"
    )

    with plt.rc_context({"font.size": 10, "axes.titleweight": "bold"}):
        fig, (policy_ax, value_ax) = plt.subplots(
            1, 2, figsize=(11.2, 5.2), layout="constrained"
        )
        _draw_cells(policy_ax, cfg)
        for row in range(cfg.row_num):
            for col in range(cfg.col_num):
                if terminal[row, col]:
                    policy_ax.text(col + 0.5, row + 0.54, "Terminal",
                                   ha="center", va="center", fontsize=8)
                    continue
                action = int(policy[row, col])
                confidence = float(probabilities[row, col, action])
                _draw_action(policy_ax, action, row, col, alpha=0.4 + 0.6 * confidence)
                policy_ax.text(col + 0.5, row + 0.88, f"{confidence:.3f}",
                               ha="center", va="center", fontsize=7, color="#333333")
        _format_grid(policy_ax, cfg, "PPO deterministic policy")
        policy_ax.legend(handles=[
            Patch(facecolor=FORBIDDEN_COLOR, edgecolor=GRID_COLOR, label="Penalty cell (F)"),
            Patch(facecolor=TARGET_COLOR, edgecolor=GRID_COLOR, label="Target (T)"),
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none",
                   markeredgecolor="#222222", label="Stay"),
        ], loc="upper center", bbox_to_anchor=(0.5, -0.13), ncols=3, frameon=False)

        norm, cmap = _value_norm(critic_values, ~terminal)
        image = value_ax.imshow(critic_values, cmap=cmap, norm=norm,
                                interpolation="nearest",
                                extent=(0, cfg.col_num, cfg.row_num, 0))
        value_ax.set_xticks(np.arange(cfg.col_num + 1), minor=True)
        value_ax.set_yticks(np.arange(cfg.row_num + 1), minor=True)
        value_ax.grid(which="minor", color=GRID_COLOR, linewidth=1.0)
        value_ax.tick_params(which="minor", length=0)
        for row in range(cfg.row_num):
            for col in range(cfg.col_num):
                prefix = {"forbidden": "F\n", "target": "T\n"}.get(
                    _cell_type(cfg, row, col), ""
                )
                value_ax.text(col + 0.5, row + 0.5,
                              f"{prefix}{critic_values[row, col]:.2f}",
                              ha="center", va="center", fontsize=8,
                              bbox={"boxstyle": "round,pad=0.14", "facecolor": "white",
                                    "edgecolor": "none", "alpha": 0.65})
        _format_grid(value_ax, cfg, "Critic estimate")
        fig.colorbar(image, ax=value_ax, label="Estimated V(s)", shrink=0.82)

    _save_figure(fig, save_path, dpi)
    if show:
        plt.show()
    return fig, (policy_ax, value_ax), policy, {
        "action_probabilities": probabilities,
        "critic_values": critic_values,
    }


def plot_baseline_comparison(reference, comparison, cfg, save_path=None,
                             show=True, dpi=180):
    """Compare all optimal baseline actions with PPO's deterministic policy."""
    optimal_actions = np.asarray(reference["optimal_actions"], dtype=bool)
    learned_policy = np.asarray(comparison["policy"], dtype=np.int64)
    correct = np.asarray(comparison["optimal_action_mask"], dtype=bool)
    terminal = np.asarray(reference["terminal_mask"], dtype=bool)
    shape = (cfg.row_num, cfg.col_num)
    if optimal_actions.shape != (*shape, cfg.action_dim) or learned_policy.shape != shape:
        raise ValueError("reference/comparison arrays do not match cfg grid shape")

    with plt.rc_context({"font.size": 10, "axes.titleweight": "bold"}):
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.2), layout="constrained")
        for ax in axes:
            _draw_cells(ax, cfg)
        for row in range(cfg.row_num):
            for col in range(cfg.col_num):
                if terminal[row, col]:
                    for ax in axes:
                        ax.text(col + 0.5, row + 0.54, "Terminal",
                                ha="center", va="center", fontsize=8)
                    continue
                tied = np.flatnonzero(optimal_actions[row, col])
                axes[0].text(col + 0.5, row + 0.52,
                             "".join(ACTION_SYMBOLS[tied]),
                             ha="center", va="center", fontsize=16)
                _draw_action(axes[1], learned_policy[row, col], row, col)
                if not correct[row, col]:
                    axes[1].text(col + 0.82, row + 0.20, "!",
                                 color=MISMATCH_COLOR, fontsize=15,
                                 fontweight="bold", ha="center", va="center")

        _format_grid(axes[0], cfg, "Optimal baseline (all ties)")
        _format_grid(axes[1], cfg, "PPO deterministic policy")
        fraction = comparison["metrics"]["optimal_action_fraction"]
        gap = comparison["metrics"]["max_absolute_value_gap"]
        fig.suptitle(
            f"Nonterminal optimal-action match: {fraction:.1%}  |  "
            f"max |V* − Vπ|: {gap:.3g}"
        )
        fig.supxlabel("F: penalty cell   T: terminal target   ○: stay   !: non-optimal action")

    _save_figure(fig, save_path, dpi)
    if show:
        plt.show()
    return fig, axes
