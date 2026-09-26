from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import torch
from rollout import encode_state


# Okabe-Ito colors. Cell labels and line styles provide redundant encodings.
COLORS = {
    "loss": "#0072B2",
    "actor_loss": "#D55E00",
    "critic_loss": "#009E73",
    "entropy": "#CC79A7",
    "return": "#000000",
}
FORBIDDEN_COLOR = "#E69F00"
TARGET_COLOR = "#56B4E9"
GRID_COLOR = "#333333"


def _encode_state(state, cfg):
    return encode_state(state, cfg)


def _save_figure(fig, save_path, dpi=180):
    if save_path is None:
        return

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, facecolor="white")


def _prepare_history(history):
    """Convert a metric mapping, list of dictionaries, or loss list to arrays."""
    if isinstance(history, Mapping):
        metrics = {
            str(name): np.asarray(values, dtype=np.float64)
            for name, values in history.items()
        }
    elif (
        isinstance(history, Sequence)
        and len(history) > 0
        and isinstance(history[0], Mapping)
    ):
        common_keys = set(history[0])
        for record in history[1:]:
            common_keys.intersection_update(record)
        metrics = {
            key: np.asarray([record[key] for record in history], dtype=np.float64)
            for key in common_keys
        }
    else:
        metrics = {"loss": np.asarray(history, dtype=np.float64)}

    if not metrics:
        raise ValueError("history must contain at least one metric")

    for name, values in metrics.items():
        if values.ndim != 1 or values.size == 0:
            raise ValueError(f"{name!r} must be a non-empty one-dimensional sequence")
    return metrics


def _plot_raw_and_mean(ax, values, label, color, smoothing_window):
    iterations = np.arange(1, len(values) + 1)
    ax.plot(
        iterations,
        values,
        color=color,
        linewidth=0.8,
        alpha=0.25,
        label=f"{label} (raw)",
    )

    window = min(int(smoothing_window), len(values))
    if window > 1:
        kernel = np.ones(window, dtype=np.float64) / window
        moving_mean = np.convolve(values, kernel, mode="valid")
        mean_iterations = np.arange(window, len(values) + 1)
        ax.plot(
            mean_iterations,
            moving_mean,
            color=color,
            linewidth=1.8,
            label=f"{label} ({window}-step mean)",
        )
    else:
        ax.plot(
            iterations,
            values,
            color=color,
            linewidth=1.5,
            label=label,
        )


def plot_training_curve(
    history,
    save_path=None,
    show=True,
    smoothing_window=50,
    dpi=180,
):
    """Plot available Actor-Critic metrics without mixing different y-scales."""
    metrics = _prepare_history(history)

    loss_keys = [
        key for key in ("loss", "actor_loss", "critic_loss") if key in metrics
    ]
    return_key = next(
        (
            key
            for key in ("episode_return", "rollout_return", "return")
            if key in metrics
        ),
        None,
    )

    panels = []
    if loss_keys:
        panels.append(("loss", loss_keys))
    if "entropy" in metrics:
        panels.append(("entropy", ["entropy"]))
    if return_key is not None:
        panels.append(("return", [return_key]))

    if not panels:
        raise ValueError(
            "history must contain loss, actor_loss, critic_loss, entropy, "
            "episode_return, rollout_return, or return"
        )

    with plt.rc_context(
        {
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        fig, axes = plt.subplots(
            len(panels),
            1,
            figsize=(7.4, 3.0 * len(panels)),
            sharex=True,
            squeeze=False,
            layout="constrained",
        )
        axes = axes[:, 0]

        for ax, (panel_name, keys) in zip(axes, panels):
            for key in keys:
                label = key.replace("_", " ").title()
                _plot_raw_and_mean(
                    ax,
                    metrics[key],
                    label,
                    COLORS.get(key, COLORS.get(panel_name, "#333333")),
                    smoothing_window,
                )

            if panel_name == "loss":
                ax.set(title="Actor-Critic optimization", ylabel="Loss")
            elif panel_name == "entropy":
                ax.set(title="Policy exploration", ylabel="Policy entropy")
            else:
                ax.set(title="Policy performance", ylabel="Return")

            ax.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.75)
            ax.legend(frameon=False, ncols=min(2, len(ax.lines)))

        axes[-1].set_xlabel("Training iteration")

    _save_figure(fig, save_path, dpi=dpi)
    if show:
        plt.show()
    return fig, axes


def compute_greedy_policy(runner, cfg):
    """Evaluate Actor probabilities and Critic values over the entire grid."""
    states = [
        _encode_state((row, col), cfg)
        for row in range(cfg.env.row_num)
        for col in range(cfg.env.col_num)
    ]
    state_tensor = torch.as_tensor(
        np.asarray(states),
        dtype=torch.float32,
        device=runner.device,
    )

    if not hasattr(runner, "model"):
        raise AttributeError(
            "runner must expose the Actor-Critic network as runner.model"
        )

    was_training = runner.model.training
    runner.model.eval()
    with torch.no_grad():
        logits, state_values = runner.model(state_tensor)
        action_probabilities = torch.softmax(logits, dim=-1)
        actions = action_probabilities.argmax(dim=1)
    if was_training:
        runner.model.train()

    rows, cols = cfg.env.row_num, cfg.env.col_num
    action_dim = action_probabilities.shape[-1]
    policy = actions.cpu().numpy().reshape(rows, cols)
    action_probabilities = action_probabilities.cpu().numpy().reshape(
        rows, cols, action_dim
    )
    state_values = state_values.cpu().numpy().reshape(rows, cols)
    return policy, action_probabilities, state_values


def _draw_grid_cells(ax, cfg):
    rows, cols = cfg.env.row_num, cfg.env.col_num
    for row in range(rows):
        for col in range(cols):
            cell_type = int(cfg.env.grid[row, col])
            color = "white"
            label = None
            if cell_type == 1:
                color = FORBIDDEN_COLOR
                label = "F"
            elif cell_type == 2:
                color = TARGET_COLOR
                label = "T"

            ax.add_patch(
                Rectangle(
                    (col, row),
                    1,
                    1,
                    facecolor=color,
                    edgecolor=GRID_COLOR,
                    linewidth=1.0,
                )
            )
            if label is not None:
                ax.text(
                    col + 0.10,
                    row + 0.18,
                    label,
                    ha="left",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color="#111111",
                )


def _format_grid_axis(ax, rows, cols, title):
    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(cols) + 0.5, labels=np.arange(1, cols + 1))
    ax.set_yticks(np.arange(rows) + 0.5, labels=np.arange(1, rows + 1))
    ax.tick_params(length=0)
    ax.set(title=title, xlabel="Column", ylabel="Row")


def plot_policy(runner, cfg, save_path=None, show=True, dpi=180):
    """Plot the Actor's greedy policy beside the Critic's state values."""
    policy, action_probabilities, state_values = compute_greedy_policy(
        runner, cfg
    )
    rows, cols = cfg.env.row_num, cfg.env.col_num

    with plt.rc_context({"font.size": 10, "axes.titleweight": "bold"}):
        fig, (policy_ax, value_ax) = plt.subplots(
            1,
            2,
            figsize=(11.2, 5.3),
            layout="constrained",
        )

        _draw_grid_cells(policy_ax, cfg)
        directions = {
            0: (0.0, -0.28),
            1: (0.28, 0.0),
            2: (0.0, 0.28),
            3: (-0.28, 0.0),
        }

        for row in range(rows):
            for col in range(cols):
                if cfg.env.grid[row, col] == 2:
                    policy_ax.text(col + 0.5, row + 0.55, "Terminal",
                                   ha="center", va="center", fontsize=9)
                    continue
                action = int(policy[row, col])
                confidence = float(action_probabilities[row, col, action])
                center_x, center_y = col + 0.5, row + 0.5

                if action in directions:
                    dx, dy = directions[action]
                    policy_ax.arrow(
                        center_x,
                        center_y,
                        dx,
                        dy,
                        width=0.018 + 0.012 * confidence,
                        head_width=0.13,
                        head_length=0.10,
                        length_includes_head=True,
                        color="#222222",
                        alpha=0.45 + 0.55 * confidence,
                    )
                else:
                    policy_ax.plot(
                        center_x,
                        center_y,
                        marker="o",
                        markersize=5 + 3 * confidence,
                        markerfacecolor="none",
                        markeredgecolor="#222222",
                        markeredgewidth=1.5,
                    )

                policy_ax.text(
                    center_x,
                    row + 0.88,
                    f"{confidence:.4f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#333333",
                )

        _format_grid_axis(policy_ax, rows, cols, "Actor greedy policy")
        policy_ax.legend(
            handles=[
                Patch(
                    facecolor=FORBIDDEN_COLOR,
                    edgecolor=GRID_COLOR,
                    label="Penalty cell (F)",
                ),
                Patch(
                    facecolor=TARGET_COLOR,
                    edgecolor=GRID_COLOR,
                    label="Target (T)",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="none",
                    markerfacecolor="none",
                    markeredgecolor="#222222",
                    label="Stay",
                ),
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncols=3,
            frameon=False,
        )
        policy_ax.text(
            0.0,
            -0.08,
            "Selected action probabilities (rounded to 4 decimals).",
            transform=policy_ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="#444444",
        )

        value_min = float(np.nanmin(state_values))
        value_max = float(np.nanmax(state_values))
        if value_min < 0.0 < value_max:
            magnitude = max(abs(value_min), abs(value_max))
            norm = mpl.colors.TwoSlopeNorm(
                vmin=-magnitude,
                vcenter=0.0,
                vmax=magnitude,
            )
            cmap = "RdBu_r"
        else:
            if np.isclose(value_min, value_max):
                value_min -= 0.5
                value_max += 0.5
            norm = mpl.colors.Normalize(vmin=value_min, vmax=value_max)
            cmap = "cividis"

        image = value_ax.imshow(
            state_values,
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            extent=(0, cols, rows, 0),
        )
        value_ax.set_xticks(
            np.arange(cols + 1), minor=True
        )
        value_ax.set_yticks(
            np.arange(rows + 1), minor=True
        )
        value_ax.grid(
            which="minor",
            color=GRID_COLOR,
            linewidth=1.0,
        )
        value_ax.tick_params(which="minor", length=0)

        for row in range(rows):
            for col in range(cols):
                cell_type = int(cfg.env.grid[row, col])
                prefix = "F\n" if cell_type == 1 else "T\n" if cell_type == 2 else ""
                value_ax.text(
                    col + 0.5,
                    row + 0.5,
                    f"{prefix}{state_values[row, col]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#111111",
                    bbox={
                        "boxstyle": "round,pad=0.15",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.62,
                    },
                )

        _format_grid_axis(value_ax, rows, cols, "Critic state value")
        fig.colorbar(image, ax=value_ax, label="Estimated V(s)", shrink=0.82)

    _save_figure(fig, save_path, dpi=dpi)
    if show:
        plt.show()

    diagnostics = {
        "action_probabilities": action_probabilities,
        "state_values": state_values,
    }
    return fig, (policy_ax, value_ax), policy, diagnostics


def plot_convergence(evaluations, cfg, save_path=None, show=True, dpi=180):
    """Exact all-state diagnostics, without smoothing or omitted evaluations."""
    x = np.asarray([r['iteration'] for r in evaluations])
    def series(key):
        return np.asarray([r[key] for r in evaluations])
    with plt.rc_context({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False}):
        fig, axes = plt.subplots(4, 1, figsize=(8.2, 11), sharex=True, layout='constrained')
        axes[0].plot(x, 100*series('optimal_action_fraction'), color='#0072B2')
        axes[0].set(ylabel='Optimal actions (%)', ylim=(-2, 102),
                    title='Exact evaluation of every nonterminal state (no smoothing)')
        axes[1].plot(x, series('greedy_max_gap'), label='Greedy policy', color='#0072B2')
        axes[1].plot(x, series('stochastic_max_gap'), label='Sampled policy', color='#D55E00', linestyle='--')
        axes[1].axhline(cfg.rl.policy_tolerance, color='black', linestyle=':', label='Sampled-policy tolerance')
        axes[1].set_yscale('symlog', linthresh=0.001)
        axes[1].set_ylim(bottom=0)
        axes[1].set(ylabel='Max value gap\n(symmetric log scale)')
        axes[1].legend(frameon=False)
        axes[2].plot(x, series('critic_max_error'), color='#009E73', label='Max |Critic - exact V(pi)|')
        axes[2].axhline(cfg.rl.critic_tolerance, color='black', linestyle=':', label='Critic tolerance')
        axes[2].set_yscale('symlog', linthresh=0.001)
        axes[2].set_ylim(bottom=0)
        axes[2].set(ylabel='Critic error\n(symmetric log scale)')
        axes[2].legend(frameon=False)
        for key, label, color, style in [
            ('optimal_mean_value', 'Optimal', 'black', ':'),
            ('greedy_mean_value', 'Greedy policy', '#0072B2', '-'),
            ('stochastic_mean_value', 'Sampled policy', '#D55E00', '--')]:
            axes[3].plot(x, series(key), label=label, color=color, linestyle=style)
        samples = int(np.count_nonzero(cfg.env.grid != 2) * cfg.rl.batch_size)
        axes[3].set(xlabel=f'Batch update ({samples} sampled transitions per update)',
                    ylabel='Mean discounted return\n(uniform nonterminal starts)')
        axes[3].legend(frameon=False)
        for ax in axes:
            ax.grid(True, alpha=0.25)
    _save_figure(fig, save_path, dpi=dpi)
    if show:
        plt.show()
    return fig, axes
