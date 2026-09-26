from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import torch

from rollout import encode_state


# Okabe-Ito colors; F/T text makes the cell meaning independent of color.
FORBIDDEN_COLOR = "#E69F00"
TARGET_COLOR = "#56B4E9"
LOSS_COLOR = "#0072B2"


def _save_figure(fig, save_path):
    if save_path is None:
        return

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, facecolor="white")


def plot_training_curve(losses, save_path=None, show=True):
    """Plot the raw mini-batch MSE loss for every training iteration."""
    losses = np.asarray(losses, dtype=np.float64)
    if losses.ndim != 1 or losses.size == 0:
        raise ValueError("losses must be a non-empty one-dimensional sequence")

    with plt.rc_context({"font.size": 10, "axes.titleweight": "bold"}):
        fig, ax = plt.subplots(figsize=(7.0, 4.2), layout="constrained")
        iterations = np.arange(1, losses.size + 1)
        ax.plot(iterations, losses, color=LOSS_COLOR, linewidth=1.1)
        ax.set(
            title="DQN training loss",
            xlabel="Training iteration",
            ylabel="Mini-batch MSE loss",
        )
        ax.set_ylim(bottom=0)
        ax.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.8)

    _save_figure(fig, save_path)
    if show:
        plt.show()
    return fig, ax


def compute_greedy_policy(runner, cfg):
    """Return greedy actions and Q values for every cell in row-major order."""
    states = [
        encode_state((row, col), cfg)
        for row in range(cfg.env.row_num)
        for col in range(cfg.env.col_num)
    ]
    state_tensor = torch.as_tensor(
        np.asarray(states),
        dtype=torch.float32,
        device=runner.device,
    )

    was_training = runner.main_net.training
    runner.main_net.eval()
    with torch.no_grad():
        q_values = runner.main_net(state_tensor)
        actions = q_values.argmax(dim=1)
    if was_training:
        runner.main_net.train()

    policy = actions.cpu().numpy().reshape(
        cfg.env.row_num, cfg.env.col_num
    )
    q_values = q_values.cpu().numpy().reshape(
        cfg.env.row_num, cfg.env.col_num, cfg.net.output_dim
    )
    return policy, q_values


def plot_policy(runner, cfg, save_path=None, show=True):
    """Plot the learned greedy action in every grid-world cell."""
    policy, q_values = compute_greedy_policy(runner, cfg)
    rows, cols = cfg.env.row_num, cfg.env.col_num

    with plt.rc_context({"font.size": 10, "axes.titleweight": "bold"}):
        fig, ax = plt.subplots(figsize=(6.2, 5.5), layout="constrained")

        for row in range(rows):
            for col in range(cols):
                cell_type = int(cfg.env.grid[row, col])
                color = "white"
                cell_label = None
                if cell_type == 1:
                    color = FORBIDDEN_COLOR
                    cell_label = "F"
                elif cell_type == 2:
                    color = TARGET_COLOR
                    cell_label = "T"

                ax.add_patch(
                    Rectangle(
                        (col, row),
                        1,
                        1,
                        facecolor=color,
                        edgecolor="#333333",
                        linewidth=1.0,
                    )
                )

                if cell_label is not None:
                    ax.text(
                        col + 0.10,
                        row + 0.18,
                        cell_label,
                        ha="left",
                        va="center",
                        fontsize=9,
                        fontweight="bold",
                        color="#111111",
                    )

                action = int(policy[row, col])
                center_x, center_y = col + 0.5, row + 0.5
                directions = {
                    0: (0.0, -0.28),
                    1: (0.28, 0.0),
                    2: (0.0, 0.28),
                    3: (-0.28, 0.0),
                }
                if action in directions:
                    dx, dy = directions[action]
                    ax.arrow(
                        center_x,
                        center_y,
                        dx,
                        dy,
                        width=0.025,
                        head_width=0.14,
                        head_length=0.11,
                        length_includes_head=True,
                        color="#222222",
                    )
                else:
                    ax.plot(
                        center_x,
                        center_y,
                        marker="o",
                        markersize=7,
                        markerfacecolor="none",
                        markeredgecolor="#222222",
                        markeredgewidth=1.5,
                    )

        ax.set_xlim(0, cols)
        ax.set_ylim(rows, 0)
        ax.set_aspect("equal")
        ax.set_xticks(np.arange(cols) + 0.5, labels=np.arange(1, cols + 1))
        ax.set_yticks(np.arange(rows) + 0.5, labels=np.arange(1, rows + 1))
        ax.tick_params(length=0)
        ax.set(
            title="Learned greedy policy",
            xlabel="Column",
            ylabel="Row",
        )

        ax.legend(
            handles=[
                Patch(
                    facecolor=FORBIDDEN_COLOR,
                    edgecolor="#333333",
                    label="Forbidden (F)",
                ),
                Patch(
                    facecolor=TARGET_COLOR,
                    edgecolor="#333333",
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
            bbox_to_anchor=(0.5, -0.12),
            ncols=3,
            frameon=False,
        )

    _save_figure(fig, save_path)
    if show:
        plt.show()
    return fig, ax, policy, q_values
