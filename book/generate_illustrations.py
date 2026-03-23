"""Generate professional illustrations for the Structural Fuzzing book.

Each function produces a single PNG in book/images/ at 300 DPI.
Consistent visual style: navy/blue/gold palette, Cambria serif, 6.5" wide.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe
from pathlib import Path

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
DARK_NAVY   = "#1A1A2E"
MEDIUM_BLUE = "#2C3E6B"
ACCENT_GOLD = "#D4A843"
ACCENT_RED  = "#C44E52"
ACCENT_GREEN = "#4C956C"
LIGHT_GRAY  = "#E8E8E8"
VERY_LIGHT  = "#F5F5F5"
TEXT_DARK    = "#333333"
WHITE        = "#FFFFFF"

PALETTE = [MEDIUM_BLUE, ACCENT_GOLD, ACCENT_RED, ACCENT_GREEN, "#7B68AE", "#E07B53"]

IMAGES_DIR = Path(__file__).parent / "images"

def _setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Cambria", "Times New Roman"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#AAAAAA",
        "axes.grid": False,
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
    })

def _save(fig, name):
    path = IMAGES_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print(f"  saved {path}")


# ===================================================================
# Chapter 1: Geometric Toolchain Overview
# ===================================================================
def ch01_geometric_toolchain():
    _setup_style()
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1.5, 5.5)
    ax.axis("off")

    # Pipeline stages
    stages = [
        ("Subset\nEnumeration", "Part III", ACCENT_GREEN, 0),
        ("Pareto\nFrontier", "Part II", ACCENT_GOLD, 1),
        ("Sensitivity\nProfile", "Part II", ACCENT_GOLD, 2),
        ("Model\nRobustness\nIndex", "Part II", ACCENT_GOLD, 3),
        ("Adversarial\nSearch", "Part III", ACCENT_GREEN, 4),
        ("Compositional\nTesting", "Part III", ACCENT_GREEN, 5),
    ]

    box_w, box_h = 1.4, 1.6
    gap = 0.45
    y_center = 2.5
    start_x = 0.2

    boxes = []
    for label, part, color, i in stages:
        x = start_x + i * (box_w + gap)
        y = y_center - box_h / 2
        box = FancyBboxPatch(
            (x, y), box_w, box_h,
            boxstyle="round,pad=0.12",
            facecolor=color, edgecolor=DARK_NAVY,
            linewidth=1.2, alpha=0.85
        )
        ax.add_patch(box)
        ax.text(x + box_w / 2, y + box_h / 2 + 0.1, label,
                ha="center", va="center", fontsize=8, fontweight="bold",
                color=WHITE, linespacing=1.2)
        ax.text(x + box_w / 2, y + 0.2, part,
                ha="center", va="center", fontsize=6.5, color=WHITE, alpha=0.8)
        boxes.append((x, y))

    # Arrows between boxes
    for i in range(len(stages) - 1):
        x1 = boxes[i][0] + box_w
        x2 = boxes[i + 1][0]
        y_mid = y_center
        ax.annotate("", xy=(x2, y_mid), xytext=(x1, y_mid),
                     arrowprops=dict(arrowstyle="-|>", color=DARK_NAVY,
                                     lw=1.5, shrinkA=2, shrinkB=2))

    # Title
    ax.text(5.25, 5.0, "Structural Fuzzing Pipeline",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color=DARK_NAVY)

    # Foundation bar
    foundation_box = FancyBboxPatch(
        (0.2, -1.0), 10.0, 0.7,
        boxstyle="round,pad=0.1",
        facecolor=MEDIUM_BLUE, edgecolor=DARK_NAVY,
        linewidth=1.0, alpha=0.3
    )
    ax.add_patch(foundation_box)
    ax.text(5.2, -0.65, "Part I: Geometric Foundations  \u2014  Mahalanobis \u00b7 Hyperbolic \u00b7 SPD \u00b7 TDA",
            ha="center", va="center", fontsize=8, color=DARK_NAVY, style="italic")

    _save(fig, "ch01-geometric-toolchain.png")


# ===================================================================
# Chapter 2: Euclidean vs Mahalanobis Distance
# ===================================================================
def ch02_euclidean_vs_mahalanobis():
    _setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.2))

    # Sample points
    np.random.seed(42)
    pts = np.random.multivariate_normal([0, 0], [[2.0, 1.2], [1.2, 1.0]], 30)

    # Euclidean equidistant contours
    theta = np.linspace(0, 2 * np.pi, 100)
    for r, alpha in [(1, 0.3), (2, 0.2), (3, 0.12)]:
        ax1.plot(r * np.cos(theta), r * np.sin(theta),
                 color=MEDIUM_BLUE, alpha=alpha + 0.3, lw=1.2)
        ax1.fill(r * np.cos(theta), r * np.sin(theta),
                 color=MEDIUM_BLUE, alpha=alpha * 0.3)
    ax1.scatter(pts[:, 0], pts[:, 1], c=ACCENT_GOLD, s=25, zorder=5, edgecolors=DARK_NAVY, linewidths=0.5)
    ax1.set_title("Euclidean Distance", color=DARK_NAVY)
    ax1.set_xlabel("Dimension 1")
    ax1.set_ylabel("Dimension 2")
    ax1.set_xlim(-4.5, 4.5)
    ax1.set_ylim(-4.5, 4.5)
    ax1.set_aspect("equal")
    ax1.text(2.5, -3.8, "d = 1, 2, 3", fontsize=7, color=MEDIUM_BLUE, alpha=0.7)

    # Mahalanobis equidistant contours (ellipses)
    cov = np.array([[2.0, 1.2], [1.2, 1.0]])
    eigvals, eigvecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))

    for r, alpha in [(1, 0.3), (2, 0.2), (3, 0.12)]:
        w = 2 * r * np.sqrt(eigvals[1])
        h = 2 * r * np.sqrt(eigvals[0])
        ell = Ellipse((0, 0), w, h, angle=angle,
                       facecolor=ACCENT_GOLD, edgecolor=ACCENT_GOLD,
                       alpha=alpha * 0.3, lw=0)
        ax2.add_patch(ell)
        ell_line = Ellipse((0, 0), w, h, angle=angle,
                            facecolor="none", edgecolor=ACCENT_GOLD,
                            alpha=alpha + 0.3, lw=1.2)
        ax2.add_patch(ell_line)

    ax2.scatter(pts[:, 0], pts[:, 1], c=MEDIUM_BLUE, s=25, zorder=5, edgecolors=DARK_NAVY, linewidths=0.5)
    ax2.set_title("Mahalanobis Distance", color=DARK_NAVY)
    ax2.set_xlabel("Dimension 1")
    ax2.set_ylabel("Dimension 2")
    ax2.set_xlim(-4.5, 4.5)
    ax2.set_ylim(-4.5, 4.5)
    ax2.set_aspect("equal")
    ax2.text(2.0, -3.8, r"$d_M = 1, 2, 3$", fontsize=7, color=ACCENT_GOLD, alpha=0.7)

    fig.suptitle("Isodistance Contours: Euclidean vs Mahalanobis",
                 fontsize=11, fontweight="bold", color=DARK_NAVY, y=1.02)
    fig.tight_layout()
    _save(fig, "ch02-euclidean-vs-mahalanobis.png")


# ===================================================================
# Chapter 3: Poincare Ball with Tree Embedding
# ===================================================================
def ch03_poincare_ball():
    _setup_style()
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    # Draw boundary circle
    boundary = Circle((0, 0), 1.0, fill=False, edgecolor=DARK_NAVY, lw=2.0)
    ax.add_patch(boundary)

    # Light radial shading
    for r in np.linspace(0.1, 0.95, 8):
        circle = Circle((0, 0), r, fill=False, edgecolor=LIGHT_GRAY, lw=0.5, ls="--")
        ax.add_patch(circle)

    # Binary tree embedding: root at origin, children spread toward boundary
    # Depth 0: root
    # Depth 1: 2 nodes
    # Depth 2: 4 nodes
    # Depth 3: 8 nodes (leaves)
    tree_nodes = {}
    depth_colors = [DARK_NAVY, MEDIUM_BLUE, ACCENT_GOLD, ACCENT_GREEN]
    depth_sizes = [120, 80, 50, 35]
    depth_radii = [0.0, 0.3, 0.58, 0.82]

    def place_tree(node_id, depth, angle_min, angle_max):
        if depth > 3:
            return
        angle = (angle_min + angle_max) / 2
        r = depth_radii[depth]
        x, y = r * np.cos(angle), r * np.sin(angle)
        tree_nodes[node_id] = (x, y, depth)

        left = 2 * node_id + 1
        right = 2 * node_id + 2
        place_tree(left, depth + 1, angle_min, angle)
        place_tree(right, depth + 1, angle, angle_max)

    place_tree(0, 0, np.pi / 6, 2 * np.pi + np.pi / 6)

    # Draw edges (approximate geodesics as straight lines in Poincare disk for clarity)
    for node_id, (x, y, d) in tree_nodes.items():
        for child_id in [2 * node_id + 1, 2 * node_id + 2]:
            if child_id in tree_nodes:
                cx, cy, cd = tree_nodes[child_id]
                ax.plot([x, cx], [y, cy], color=MEDIUM_BLUE, lw=1.0, alpha=0.5, zorder=1)

    # Draw nodes
    for node_id, (x, y, d) in tree_nodes.items():
        ax.scatter(x, y, c=depth_colors[d], s=depth_sizes[d],
                   zorder=5, edgecolors=DARK_NAVY, linewidths=0.7)

    # Annotations
    ax.text(0.0, -0.08, "root", ha="center", va="top", fontsize=7,
            color=DARK_NAVY, fontweight="bold")
    ax.annotate("specificity increases\ntoward boundary",
                xy=(0.75, 0.45), xytext=(1.1, 0.7),
                fontsize=7, color=TEXT_DARK, style="italic",
                arrowprops=dict(arrowstyle="->", color=TEXT_DARK, lw=0.8))

    # Conformal factor annotation
    ax.text(0.0, -1.18, r"Conformal factor $\lambda_x^c \to \infty$ as $\|x\| \to 1$",
            ha="center", fontsize=8, color=MEDIUM_BLUE, style="italic")

    # Legend
    for i, (label, color) in enumerate(zip(
            ["Depth 0 (root)", "Depth 1", "Depth 2", "Depth 3 (leaves)"],
            depth_colors)):
        ax.scatter([], [], c=color, s=40, label=label, edgecolors=DARK_NAVY, linewidths=0.5)
    ax.legend(loc="lower left", fontsize=7, framealpha=0.9)

    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Poincar\u00e9 Ball: Hierarchical Tree Embedding",
                 fontsize=12, fontweight="bold", color=DARK_NAVY, pad=15)

    _save(fig, "ch03-poincare-ball.png")


# ===================================================================
# Chapter 4: SPD Manifold Log-Euclidean Unfolding
# ===================================================================
def ch04_spd_manifold():
    _setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.0))

    # Left panel: SPD matrices as ellipses (eigenvalue visualization)
    matrices = [
        (np.array([[2.0, 0.5], [0.5, 0.5]]), "S\u2081", MEDIUM_BLUE),
        (np.array([[1.0, 0.0], [0.0, 3.0]]), "S\u2082", ACCENT_GOLD),
        (np.array([[0.5, -0.3], [-0.3, 1.5]]), "S\u2083", ACCENT_GREEN),
    ]

    positions = [(0.0, 0.0), (2.5, 0.5), (1.2, -1.0)]
    for (mat, label, color), (px, py) in zip(matrices, positions):
        eigvals, eigvecs = np.linalg.eigh(mat)
        angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
        w, h = 2 * np.sqrt(eigvals[1]), 2 * np.sqrt(eigvals[0])
        ell = Ellipse((px, py), w, h, angle=angle,
                       facecolor=color, edgecolor=DARK_NAVY,
                       alpha=0.3, lw=1.2)
        ax1.add_patch(ell)
        ell_line = Ellipse((px, py), w, h, angle=angle,
                            facecolor="none", edgecolor=color, lw=1.5)
        ax1.add_patch(ell_line)
        ax1.text(px, py, label, ha="center", va="center",
                 fontsize=9, fontweight="bold", color=DARK_NAVY)

    # Curved surface hint
    ax1.annotate("SPD(n)\n(curved)", xy=(1.2, 1.8), fontsize=8,
                 ha="center", color=MEDIUM_BLUE, style="italic")
    ax1.set_xlim(-1.8, 4.3)
    ax1.set_ylim(-2.5, 2.5)
    ax1.set_aspect("equal")
    ax1.axis("off")
    ax1.set_title("SPD Matrices as Ellipses", fontsize=10, fontweight="bold", color=DARK_NAVY)

    # Right panel: Log-Euclidean space (points)
    log_positions = [(0.0, 0.0), (2.0, 0.8), (1.0, -0.5)]
    for (mat, label, color), (lx, ly) in zip(matrices, log_positions):
        ax2.scatter(lx, ly, c=color, s=100, zorder=5, edgecolors=DARK_NAVY, linewidths=1.0)
        ax2.text(lx + 0.15, ly + 0.15, f"log({label})", fontsize=7, color=DARK_NAVY)

    # Grid to show flat space
    for x in np.arange(-0.5, 3.0, 0.5):
        ax2.axvline(x, color=LIGHT_GRAY, lw=0.5, zorder=0)
    for y in np.arange(-1.0, 1.5, 0.5):
        ax2.axhline(y, color=LIGHT_GRAY, lw=0.5, zorder=0)

    ax2.annotate("Sym(n)\n(flat)", xy=(2.0, -0.9), fontsize=8,
                 ha="center", color=MEDIUM_BLUE, style="italic")
    ax2.set_xlim(-0.8, 2.8)
    ax2.set_ylim(-1.2, 1.5)
    ax2.set_aspect("equal")
    ax2.axis("off")
    ax2.set_title("Log-Euclidean Space", fontsize=10, fontweight="bold", color=DARK_NAVY)

    # Arrow between panels
    fig.text(0.50, 0.50, "log \u2192", fontsize=12, fontweight="bold",
             color=ACCENT_GOLD, ha="center", va="center",
             transform=fig.transFigure)

    fig.suptitle("SPD Manifold \u2192 Log-Euclidean Unfolding",
                 fontsize=11, fontweight="bold", color=DARK_NAVY, y=1.05)
    fig.tight_layout()
    _save(fig, "ch04-spd-manifold.png")


# ===================================================================
# Chapter 5: TDA - Vietoris-Rips Filtration & Persistence Diagram
# ===================================================================
def ch05_tda_persistence():
    _setup_style()
    fig, axes = plt.subplots(1, 4, figsize=(6.5, 2.2))

    # Generate ring-shaped point cloud
    np.random.seed(7)
    n_pts = 24
    angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False) + np.random.normal(0, 0.15, n_pts)
    radii = 1.0 + np.random.normal(0, 0.08, n_pts)
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    epsilons = [0.3, 0.6, 1.0]
    titles = [f"\u03b5 = {e}" for e in epsilons] + ["Persistence\nDiagram"]

    for idx, (ax, eps) in enumerate(zip(axes[:3], epsilons)):
        # Draw edges
        from scipy.spatial.distance import pdist, squareform
        D = squareform(pdist(pts))
        for i in range(n_pts):
            for j in range(i + 1, n_pts):
                if D[i, j] <= eps:
                    ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]],
                            color=MEDIUM_BLUE, alpha=0.15, lw=0.6)
                    # Fill triangles at higher epsilon
                    if eps >= 0.6:
                        for k in range(j + 1, n_pts):
                            if D[i, k] <= eps and D[j, k] <= eps:
                                tri = plt.Polygon([pts[i], pts[j], pts[k]],
                                                   facecolor=MEDIUM_BLUE, alpha=0.04,
                                                   edgecolor="none")
                                ax.add_patch(tri)

        ax.scatter(pts[:, 0], pts[:, 1], c=ACCENT_GOLD, s=12, zorder=5,
                   edgecolors=DARK_NAVY, linewidths=0.3)
        ax.set_title(titles[idx], fontsize=8, color=DARK_NAVY)
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.8, 1.8)
        ax.set_aspect("equal")
        ax.axis("off")

    # Persistence diagram
    ax_pd = axes[3]
    # Simulated persistence pairs
    h0_pairs = [(0, e) for e in sorted(np.random.uniform(0.2, 0.5, 6))] + [(0, np.inf)]
    h1_pairs = [(0.55, 1.05)]  # the main loop
    noise_h1 = [(b, b + np.random.uniform(0.05, 0.15)) for b in np.random.uniform(0.3, 0.7, 3)]

    max_d = 1.4
    ax_pd.plot([0, max_d], [0, max_d], color=LIGHT_GRAY, lw=1.0, ls="--", zorder=0)

    for b, d in h0_pairs:
        if d == np.inf:
            ax_pd.scatter(b, max_d - 0.05, c=MEDIUM_BLUE, s=30, marker="^", zorder=5,
                          edgecolors=DARK_NAVY, linewidths=0.5)
        else:
            ax_pd.scatter(b, d, c=MEDIUM_BLUE, s=20, zorder=5,
                          edgecolors=DARK_NAVY, linewidths=0.5)

    for b, d in h1_pairs:
        ax_pd.scatter(b, d, c=ACCENT_RED, s=40, zorder=5, marker="s",
                      edgecolors=DARK_NAVY, linewidths=0.5)
    for b, d in noise_h1:
        ax_pd.scatter(b, d, c=ACCENT_RED, s=15, zorder=5, marker="s",
                      edgecolors=DARK_NAVY, linewidths=0.3, alpha=0.5)

    ax_pd.scatter([], [], c=MEDIUM_BLUE, s=20, label="H\u2080 (components)")
    ax_pd.scatter([], [], c=ACCENT_RED, s=20, marker="s", label="H\u2081 (loops)")
    ax_pd.legend(fontsize=5.5, loc="lower right", framealpha=0.9)
    ax_pd.set_xlabel("Birth", fontsize=7)
    ax_pd.set_ylabel("Death", fontsize=7)
    ax_pd.set_title(titles[3], fontsize=8, color=DARK_NAVY)
    ax_pd.set_xlim(-0.05, max_d)
    ax_pd.set_ylim(-0.05, max_d)
    ax_pd.tick_params(labelsize=6)

    # Annotate the main loop
    ax_pd.annotate("main loop", xy=(0.55, 1.05), xytext=(0.15, 1.15),
                   fontsize=5.5, color=ACCENT_RED, style="italic",
                   arrowprops=dict(arrowstyle="->", color=ACCENT_RED, lw=0.6))

    fig.suptitle("Vietoris\u2013Rips Filtration and Persistence Diagram",
                 fontsize=10, fontweight="bold", color=DARK_NAVY, y=1.08)
    fig.tight_layout()
    _save(fig, "ch05-tda-persistence.png")


# ===================================================================
# Chapter 6: Pathfinding on Decision Manifold
# ===================================================================
def ch06_pathfinding():
    _setup_style()
    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    # 2D cost landscape
    np.random.seed(3)
    nx_grid, ny_grid = 40, 30
    x = np.linspace(0, 8, nx_grid)
    y = np.linspace(0, 6, ny_grid)
    X, Y = np.meshgrid(x, y)
    # Cost field with a ridge (moral boundary)
    cost = 0.5 + 0.3 * np.sin(X * 0.5) + 0.2 * np.cos(Y * 0.7)
    boundary_y = 3.0
    cost += 3.0 * np.exp(-0.5 * ((Y - boundary_y) / 0.3) ** 2)

    im = ax.pcolormesh(X, Y, cost, cmap="YlOrBr", shading="auto", alpha=0.6)

    # Moral boundary line
    ax.axhline(boundary_y, color=ACCENT_RED, lw=2.0, ls="--", alpha=0.8)
    ax.text(7.0, boundary_y + 0.2, "moral boundary", fontsize=7,
            color=ACCENT_RED, fontweight="bold", ha="right")

    # Start and goal
    start = (1.0, 1.0)
    goal = (7.0, 5.0)
    ax.scatter(*start, c=ACCENT_GREEN, s=100, zorder=10, edgecolors=DARK_NAVY,
               linewidths=1.2, marker="o")
    ax.scatter(*goal, c=ACCENT_GOLD, s=100, zorder=10, edgecolors=DARK_NAVY,
               linewidths=1.2, marker="*")
    ax.text(start[0] - 0.1, start[1] - 0.4, "start", fontsize=8, color=ACCENT_GREEN, fontweight="bold")
    ax.text(goal[0] + 0.1, goal[1] - 0.4, "goal", fontsize=8, color=ACCENT_GOLD, fontweight="bold")

    # Naive path (crosses boundary)
    naive_x = np.linspace(1.0, 7.0, 20)
    naive_y = np.linspace(1.0, 5.0, 20)
    ax.plot(naive_x, naive_y, color=ACCENT_RED, lw=2.0, ls=":", alpha=0.7, zorder=8)
    ax.text(3.5, 2.2, "Euclidean\n(crosses boundary)", fontsize=6.5,
            color=ACCENT_RED, style="italic", ha="center")

    # Manifold-aware path (goes around)
    t = np.linspace(0, 1, 30)
    mani_x = 1.0 + 6.0 * t
    mani_y = 1.0 + 0.5 * t + 3.5 * t ** 2.5
    # Keep below boundary until past it
    mani_y = np.where(mani_x < 5.5, 1.0 + 1.5 * (mani_x - 1.0) / 4.5, mani_y)
    mani_y[-5:] = np.linspace(mani_y[-6], 5.0, 5)
    ax.plot(mani_x, mani_y, color=ACCENT_GREEN, lw=2.5, zorder=8)
    ax.text(4.0, 1.0, "manifold-aware\n(respects boundary)", fontsize=6.5,
            color=ACCENT_GREEN, style="italic", ha="center")

    ax.set_xlabel("Consequences", fontsize=9)
    ax.set_ylabel("Fairness", fontsize=9)
    ax.set_title("A* Pathfinding on Decision Manifold",
                 fontsize=11, fontweight="bold", color=DARK_NAVY)

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, label="Mahalanobis cost")
    cbar.ax.tick_params(labelsize=7)
    _save(fig, "ch06-pathfinding.png")


# ===================================================================
# Chapter 7: Nash vs BGE Comparison
# ===================================================================
def ch07_nash_vs_bge():
    _setup_style()
    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    # Decision options in 2D: Consequences vs Fairness
    offers = {
        "0%":  (9.0, 0.2),
        "10%": (7.5, 0.5),
        "20%": (6.0, 0.7),
        "30%": (4.5, 0.85),
        "40%": (3.0, 0.95),
        "50%": (1.5, 1.0),
    }

    for label, (cx, fy) in offers.items():
        ax.scatter(cx, fy, c=MEDIUM_BLUE, s=60, zorder=5,
                   edgecolors=DARK_NAVY, linewidths=0.8)
        ax.text(cx + 0.15, fy + 0.06, label, fontsize=7, color=TEXT_DARK)

    # Nash equilibrium (maximizes scalar utility = consequences)
    nash_x, nash_y = 9.0, 0.2
    ax.scatter(nash_x, nash_y, c=ACCENT_RED, s=150, zorder=6,
               edgecolors=DARK_NAVY, linewidths=1.5, marker="X")
    ax.annotate("Nash Equilibrium\n(scalar: max payoff)",
                xy=(nash_x, nash_y), xytext=(7.0, 0.6),
                fontsize=7, color=ACCENT_RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ACCENT_RED, lw=1.0))

    # BGE (minimizes multi-dimensional distance)
    bge_x, bge_y = 3.0, 0.95
    ax.scatter(bge_x, bge_y, c=ACCENT_GREEN, s=150, zorder=6,
               edgecolors=DARK_NAVY, linewidths=1.5, marker="D")
    ax.annotate("Bond Geodesic\nEquilibrium\n(multi-dimensional)",
                xy=(bge_x, bge_y), xytext=(1.0, 0.55),
                fontsize=7, color=ACCENT_GREEN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ACCENT_GREEN, lw=1.0))

    # Mahalanobis ellipses around BGE
    for r, alpha in [(0.8, 0.15), (1.5, 0.08)]:
        ell = Ellipse((bge_x, bge_y), r * 3.0, r * 0.4, angle=-10,
                       facecolor=ACCENT_GREEN, alpha=alpha, edgecolor="none")
        ax.add_patch(ell)

    ax.set_xlabel("Cost (Consequences)", fontsize=9)
    ax.set_ylabel("Fairness", fontsize=9)
    ax.set_title("Ultimatum Game: Nash vs Bond Geodesic Equilibrium",
                 fontsize=10, fontweight="bold", color=DARK_NAVY)
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 1.2)
    _save(fig, "ch07-nash-vs-bge.png")


# ===================================================================
# Chapter 8: Pareto Frontier with Dominance
# ===================================================================
def ch08_pareto_frontier():
    _setup_style()
    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    # Subset results: (num_dims, MAE)
    np.random.seed(15)
    results = []
    for k in range(1, 6):
        for _ in range(6):
            mae = 4.5 / k + np.random.uniform(-0.3, 0.5)
            results.append((k, mae))
    results = np.array(results)

    # Find Pareto frontier (minimize both k and MAE... actually minimize MAE, k is a cost)
    # Pareto: no other point has both <= k and <= MAE
    pareto_mask = np.zeros(len(results), dtype=bool)
    for i, (k_i, m_i) in enumerate(results):
        dominated = False
        for j, (k_j, m_j) in enumerate(results):
            if i != j and k_j <= k_i and m_j <= m_i and (k_j < k_i or m_j < m_i):
                dominated = True
                break
        pareto_mask[i] = not dominated

    # Shade dominated region
    pareto_pts = results[pareto_mask]
    pareto_pts = pareto_pts[pareto_pts[:, 0].argsort()]

    # Draw dominated and non-dominated
    non_pareto = results[~pareto_mask]
    ax.scatter(non_pareto[:, 0], non_pareto[:, 1], c=LIGHT_GRAY, s=40, zorder=3,
               edgecolors="#BBBBBB", linewidths=0.5, label="Dominated")
    ax.scatter(pareto_pts[:, 0], pareto_pts[:, 1], c=ACCENT_GOLD, s=70, zorder=5,
               edgecolors=DARK_NAVY, linewidths=1.0, label="Pareto-optimal")

    # Connect frontier
    ax.plot(pareto_pts[:, 0], pareto_pts[:, 1], color=ACCENT_GOLD, lw=2.0, ls="-", zorder=4)

    # Shade dominated region (above and to the right of frontier)
    fill_x = np.concatenate([[pareto_pts[0, 0]], pareto_pts[:, 0], [pareto_pts[-1, 0], pareto_pts[0, 0]]])
    fill_y = np.concatenate([[6.0], pareto_pts[:, 1], [6.0, 6.0]])
    ax.fill(fill_x, fill_y, color=ACCENT_RED, alpha=0.06)
    ax.text(3.5, 4.5, "dominated\nregion", fontsize=8, color=ACCENT_RED, alpha=0.5,
            ha="center", style="italic")

    # Labels for key Pareto points
    labels = {1: "{Complexity}", 2: "{Compl., Process}", 3: "{C, P, OO}"}
    for k_val, label in labels.items():
        pts_at_k = pareto_pts[pareto_pts[:, 0] == k_val]
        if len(pts_at_k) > 0:
            px, py = pts_at_k[0]
            ax.annotate(label, xy=(px, py), xytext=(px + 0.3, py + 0.3),
                        fontsize=6, color=TEXT_DARK,
                        arrowprops=dict(arrowstyle="->", color=TEXT_DARK, lw=0.5))

    ax.set_xlabel("Number of Dimensions (k)", fontsize=9)
    ax.set_ylabel("Mean Absolute Error (MAE)", fontsize=9)
    ax.set_title("Pareto Frontier: Dimension Count vs Prediction Error",
                 fontsize=10, fontweight="bold", color=DARK_NAVY)
    ax.legend(fontsize=7, framealpha=0.9)
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0, 5.5)
    _save(fig, "ch08-pareto-frontier.png")


# ===================================================================
# Chapter 9: Sharp vs Broad Minima + MRI Distribution
# ===================================================================
def ch09_robustness():
    _setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.0))

    # Left: Sharp vs Broad minima (1D loss landscape)
    x = np.linspace(-4, 4, 300)
    # Broad minimum at x=-1.5
    loss_broad = 0.5 * (x + 1.5) ** 2 / 3.0 + 0.5
    # Sharp minimum at x=1.5
    loss_sharp = 2.0 * (x - 1.5) ** 2 + 0.3
    # Combined landscape
    loss = np.minimum(loss_broad, loss_sharp) + 0.1 * np.sin(x * 3)

    ax1.plot(x, loss, color=DARK_NAVY, lw=2.0)
    ax1.fill_between(x, loss, alpha=0.05, color=MEDIUM_BLUE)

    # Highlight broad minimum region
    mask_broad = (x > -3.0) & (x < 0.0)
    ax1.fill_between(x[mask_broad], loss[mask_broad], alpha=0.2, color=ACCENT_GREEN)
    ax1.annotate("broad\n(robust)", xy=(-1.5, 0.55), xytext=(-3.2, 2.5),
                 fontsize=8, color=ACCENT_GREEN, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=ACCENT_GREEN, lw=1.0))

    # Highlight sharp minimum
    mask_sharp = (x > 1.0) & (x < 2.0)
    ax1.fill_between(x[mask_sharp], loss[mask_sharp], alpha=0.2, color=ACCENT_RED)
    ax1.annotate("sharp\n(fragile)", xy=(1.5, 0.4), xytext=(2.5, 2.5),
                 fontsize=8, color=ACCENT_RED, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=ACCENT_RED, lw=1.0))

    ax1.set_xlabel("Parameter space", fontsize=9)
    ax1.set_ylabel("Loss", fontsize=9)
    ax1.set_title("Loss Landscape Geometry", fontsize=10, fontweight="bold", color=DARK_NAVY)
    ax1.set_ylim(0, 4.0)

    # Right: MRI perturbation distribution
    np.random.seed(22)
    perturbations = np.concatenate([
        np.random.exponential(0.3, 200),
        np.random.exponential(1.0, 30),
    ])

    ax2.hist(perturbations, bins=30, color=MEDIUM_BLUE, edgecolor=WHITE,
             alpha=0.7, density=True)

    mean_val = np.mean(perturbations)
    p75 = np.percentile(perturbations, 75)
    p95 = np.percentile(perturbations, 95)

    for val, label, color in [(mean_val, "mean", ACCENT_GREEN),
                               (p75, "P75", ACCENT_GOLD),
                               (p95, "P95", ACCENT_RED)]:
        ax2.axvline(val, color=color, lw=2.0, ls="--")
        ax2.text(val + 0.05, ax2.get_ylim()[1] * 0.9, label,
                 fontsize=7, color=color, fontweight="bold", rotation=0)

    # MRI formula
    ax2.text(0.95, 0.7, "MRI = 0.5\u00b7mean\n    + 0.3\u00b7P75\n    + 0.2\u00b7P95",
             transform=ax2.transAxes, fontsize=7, color=TEXT_DARK,
             ha="right", va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.3", facecolor=VERY_LIGHT, edgecolor=LIGHT_GRAY))

    ax2.set_xlabel("MAE deviation under perturbation", fontsize=9)
    ax2.set_ylabel("Density", fontsize=9)
    ax2.set_title("MRI: Perturbation Distribution", fontsize=10, fontweight="bold", color=DARK_NAVY)

    fig.tight_layout()
    _save(fig, "ch09-robustness.png")


# ===================================================================
# Chapter 11: Subset Lattice Hasse Diagram
# ===================================================================
def ch11_hasse_diagram():
    _setup_style()
    import networkx as nx

    dims = ["Size", "Cmplx", "Halst", "OO"]
    n = len(dims)
    all_subsets = []
    for mask in range(1, 2 ** n):
        subset = tuple(i for i in range(n) if mask & (1 << i))
        all_subsets.append(subset)

    # Simulated MAE for each subset (lower = better)
    np.random.seed(10)
    mae_map = {}
    for s in all_subsets:
        base = 4.0 / (len(s) + 0.5)
        mae_map[s] = base + np.random.uniform(-0.3, 0.3)

    fig, ax = plt.subplots(figsize=(6.5, 5.0))

    # Layout: group by cardinality
    positions = {}
    for card in range(1, n + 1):
        card_subsets = [s for s in all_subsets if len(s) == card]
        n_at_card = len(card_subsets)
        for i, s in enumerate(card_subsets):
            x = (i - (n_at_card - 1) / 2) * 1.6
            y = card * 1.4
            positions[s] = (x, y)

    # Draw edges (subset differs by exactly one element)
    for s1 in all_subsets:
        for s2 in all_subsets:
            if len(s2) == len(s1) + 1 and set(s1).issubset(set(s2)):
                x1, y1 = positions[s1]
                x2, y2 = positions[s2]
                ax.plot([x1, x2], [y1, y2], color=LIGHT_GRAY, lw=0.8, zorder=0)

    # Color by MAE
    mae_vals = [mae_map[s] for s in all_subsets]
    norm = plt.Normalize(min(mae_vals), max(mae_vals))
    cmap = LinearSegmentedColormap.from_list("mae", [ACCENT_GREEN, ACCENT_GOLD, ACCENT_RED])

    for s in all_subsets:
        x, y = positions[s]
        mae = mae_map[s]
        color = cmap(norm(mae))
        ax.scatter(x, y, c=[color], s=280, zorder=5, edgecolors=DARK_NAVY, linewidths=0.8)
        label = ",".join(dims[i][0] for i in s)
        ax.text(x, y, label, ha="center", va="center", fontsize=5.5,
                fontweight="bold", color=DARK_NAVY)

    # Y-axis labels
    for card in range(1, n + 1):
        ax.text(-4.5, card * 1.4, f"k = {card}", fontsize=8, color=TEXT_DARK,
                ha="center", va="center", fontweight="bold")

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, label="MAE", pad=0.02)
    cbar.ax.tick_params(labelsize=7)

    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(0.5, n * 1.4 + 0.8)
    ax.axis("off")
    ax.set_title("Subset Lattice: Feature Combinations Colored by Error",
                 fontsize=10, fontweight="bold", color=DARK_NAVY, pad=15)

    _save(fig, "ch11-hasse-diagram.png")


# ===================================================================
# Chapter 12: Interaction Matrix Heatmap
# ===================================================================
def ch12_interaction_heatmap():
    _setup_style()
    fig, ax = plt.subplots(figsize=(5.0, 4.5))

    dims = ["Size", "Complexity", "Halstead", "OO", "Process"]
    n = len(dims)
    np.random.seed(8)

    # Simulated interaction matrix (symmetric, zero diagonal)
    Phi = np.random.uniform(-0.5, 0.5, (n, n))
    Phi = (Phi + Phi.T) / 2
    np.fill_diagonal(Phi, 0)
    # Make some strong interactions
    Phi[0, 2] = Phi[2, 0] = 0.8   # Size x Halstead = synergistic
    Phi[1, 4] = Phi[4, 1] = -0.6  # Complexity x Process = redundant
    Phi[3, 4] = Phi[4, 3] = 0.7   # OO x Process = synergistic

    cmap = LinearSegmentedColormap.from_list("interact",
        [MEDIUM_BLUE, WHITE, ACCENT_RED])
    im = ax.imshow(Phi, cmap=cmap, vmin=-1, vmax=1, aspect="equal")

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = Phi[i, j]
            text_color = WHITE if abs(val) > 0.55 else TEXT_DARK
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=text_color, fontweight="bold" if abs(val) > 0.5 else "normal")

    ax.set_xticks(range(n))
    ax.set_xticklabels(dims, fontsize=8, rotation=30, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(dims, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Interaction \u03a6", fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    # Legend text
    ax.text(n + 0.8, n - 0.5, "red = synergy\nblue = redundancy",
            fontsize=6.5, color=TEXT_DARK, style="italic")

    ax.set_title("Pairwise Dimension Interaction Matrix",
                 fontsize=10, fontweight="bold", color=DARK_NAVY, pad=12)
    fig.tight_layout()
    _save(fig, "ch12-interaction-heatmap.png")


# ===================================================================
# Chapter 13: D4 Dihedral Group Elements
# ===================================================================
def ch13_dihedral_grid():
    _setup_style()
    fig, axes = plt.subplots(2, 4, figsize=(6.5, 3.5))

    # L-shape on 5x5 grid
    grid = np.zeros((5, 5))
    grid[0, 0] = 1  # top of L
    grid[1, 0] = 1
    grid[2, 0] = 1
    grid[2, 1] = 1
    grid[2, 2] = 1

    def rot90(g):
        return np.rot90(g, -1)

    def reflect_h(g):
        return np.flipud(g)

    # D4 elements: e, r, r^2, r^3, s, sr, sr^2, sr^3
    transforms = [
        ("e (identity)", grid),
        ("r (90\u00b0)", rot90(grid)),
        ("r\u00b2 (180\u00b0)", rot90(rot90(grid))),
        ("r\u00b3 (270\u00b0)", rot90(rot90(rot90(grid)))),
        ("s (reflect \u2194)", reflect_h(grid)),
        ("sr (s\u00b7r)", rot90(reflect_h(grid))),
        ("sr\u00b2 (s\u00b7r\u00b2)", rot90(rot90(reflect_h(grid)))),
        ("sr\u00b3 (s\u00b7r\u00b3)", rot90(rot90(rot90(reflect_h(grid))))),
    ]

    cmap = LinearSegmentedColormap.from_list("d4", [WHITE, ACCENT_GOLD])

    for ax, (label, g) in zip(axes.flat, transforms):
        ax.imshow(g, cmap=cmap, vmin=0, vmax=1, aspect="equal")
        # Grid lines
        for i in range(6):
            ax.axhline(i - 0.5, color=LIGHT_GRAY, lw=0.5)
            ax.axvline(i - 0.5, color=LIGHT_GRAY, lw=0.5)
        ax.set_title(label, fontsize=6.5, color=DARK_NAVY, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(MEDIUM_BLUE)
            spine.set_linewidth(0.8)

    fig.suptitle("Dihedral Group D\u2084: All 8 Symmetries of an L-Shape",
                 fontsize=10, fontweight="bold", color=DARK_NAVY, y=1.03)
    fig.tight_layout()
    _save(fig, "ch13-dihedral-grid.png")


# ===================================================================
# Chapter 16: Six-Stage Pipeline Architecture
# ===================================================================
def ch16_pipeline_architecture():
    _setup_style()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 8)
    ax.axis("off")

    stages = [
        ("1. Subset\nEnumeration", "Evaluate all\nfeature subsets", ACCENT_GREEN, 0),
        ("2. Pareto\nFrontier", "Extract non-\ndominated configs", ACCENT_GOLD, 1),
        ("3. Sensitivity\nProfile", "Rank dimension\nimportance", MEDIUM_BLUE, 2),
        ("4. Model\nRobustness Index", "Quantify\nperturbation risk", MEDIUM_BLUE, 3),
        ("5. Adversarial\nSearch", "Find worst-case\nboundaries", ACCENT_RED, 4),
        ("6. Compositional\nTesting", "Discover dimension\ninteractions", ACCENT_GREEN, 5),
    ]

    box_w, box_h = 3.8, 0.9
    x_left = 1.5
    y_top = 7.0
    spacing = 1.15

    for label, desc, color, i in stages:
        y = y_top - i * spacing
        box = FancyBboxPatch(
            (x_left, y - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor=DARK_NAVY,
            linewidth=1.0, alpha=0.8
        )
        ax.add_patch(box)
        ax.text(x_left + 0.2, y, label, fontsize=8, fontweight="bold",
                color=WHITE, va="center")
        ax.text(x_left + box_w - 0.2, y, desc, fontsize=6.5,
                color=WHITE, va="center", ha="right", alpha=0.9)

        # Arrow to next
        if i < 5:
            y_next = y_top - (i + 1) * spacing
            ax.annotate("", xy=(x_left + box_w / 2, y_next + box_h / 2),
                         xytext=(x_left + box_w / 2, y - box_h / 2),
                         arrowprops=dict(arrowstyle="-|>", color=DARK_NAVY, lw=1.2))

    # Central evaluate_fn box
    eval_x = 7.0
    eval_y = 4.0
    eval_box = FancyBboxPatch(
        (eval_x, eval_y - 0.6), 2.8, 1.2,
        boxstyle="round,pad=0.1",
        facecolor=VERY_LIGHT, edgecolor=DARK_NAVY,
        linewidth=1.5, linestyle="--"
    )
    ax.add_patch(eval_box)
    ax.text(eval_x + 1.4, eval_y, "evaluate_fn\n(domain-specific)",
            ha="center", va="center", fontsize=8, fontweight="bold",
            color=DARK_NAVY, fontfamily="monospace")

    # Arrows from stages to evaluate_fn
    for i in [0, 2, 3, 4]:
        y = y_top - i * spacing
        ax.annotate("", xy=(eval_x, eval_y + 0.2),
                     xytext=(x_left + box_w, y),
                     arrowprops=dict(arrowstyle="->", color=TEXT_DARK,
                                     lw=0.6, ls="--", connectionstyle="arc3,rad=0.2"))

    # Output box
    out_y = y_top - 5 * spacing - 1.2
    out_box = FancyBboxPatch(
        (x_left, out_y - 0.4), box_w + 3.5, 0.8,
        boxstyle="round,pad=0.1",
        facecolor=DARK_NAVY, edgecolor=DARK_NAVY,
        linewidth=1.0, alpha=0.9
    )
    ax.add_patch(out_box)
    ax.text(x_left + (box_w + 3.5) / 2, out_y, "StructuralFuzzReport",
            ha="center", va="center", fontsize=9, fontweight="bold",
            color=WHITE, fontfamily="monospace")

    ax.annotate("", xy=(x_left + box_w / 2, out_y + 0.4),
                 xytext=(x_left + box_w / 2, y_top - 5 * spacing - box_h / 2),
                 arrowprops=dict(arrowstyle="-|>", color=DARK_NAVY, lw=1.2))

    ax.set_title("Structural Fuzzing: Six-Stage Pipeline",
                 fontsize=12, fontweight="bold", color=DARK_NAVY, pad=15)

    _save(fig, "ch16-pipeline-architecture.png")


# ===================================================================
# Chapter 20: Bioacoustics - Cetacean Coda + Poincare Embedding
# ===================================================================
def ch20_bioacoustics():
    _setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.2))

    # Left: Simulated spectrogram of whale clicks
    np.random.seed(5)
    n_clicks = 8
    t_total = 2.0  # seconds
    n_freq = 64
    n_time = 200
    spec = np.random.uniform(0, 0.1, (n_freq, n_time))

    # Add click events
    click_times = np.sort(np.random.uniform(0.1, 1.9, n_clicks))
    for ct in click_times:
        t_idx = int(ct / t_total * n_time)
        width = 3
        for dt in range(-width, width + 1):
            if 0 <= t_idx + dt < n_time:
                # Broadband click with some frequency structure
                click_spectrum = np.exp(-0.5 * ((np.arange(n_freq) - 35) / 15) ** 2)
                click_spectrum += 0.3 * np.exp(-0.5 * ((np.arange(n_freq) - 20) / 8) ** 2)
                spec[:, t_idx + dt] += click_spectrum * np.exp(-0.5 * (dt / 1.5) ** 2)

    ax1.imshow(spec, aspect="auto", origin="lower", cmap="YlOrBr",
               extent=[0, t_total, 0, 16], vmin=0, vmax=1.0)
    for ct in click_times:
        ax1.axvline(ct, color=ACCENT_GOLD, lw=0.5, alpha=0.5, ls="--")
    ax1.set_xlabel("Time (s)", fontsize=8)
    ax1.set_ylabel("Frequency (kHz)", fontsize=8)
    ax1.set_title("Sperm Whale Coda", fontsize=9, fontweight="bold", color=DARK_NAVY)
    ax1.tick_params(labelsize=6)

    # Right: Poincare disk with cetacean taxonomy
    boundary = Circle((0, 0), 1.0, fill=False, edgecolor=DARK_NAVY, lw=1.5)
    ax2.add_patch(boundary)

    taxonomy = {
        "Cetacea": (0.0, 0.0, DARK_NAVY, 9),
        "Odontoceti": (-0.2, 0.25, MEDIUM_BLUE, 7),
        "Mysticeti": (0.25, 0.2, MEDIUM_BLUE, 7),
        "Physeteridae": (-0.45, 0.5, ACCENT_GOLD, 6),
        "Delphinidae": (-0.1, 0.55, ACCENT_GREEN, 6),
        "Balaenopteridae": (0.4, 0.5, ACCENT_RED, 6),
        "P. macrocephalus": (-0.6, 0.72, ACCENT_GOLD, 5),
        "P. catodon": (-0.35, 0.75, ACCENT_GOLD, 5),
        "T. truncatus": (0.05, 0.78, ACCENT_GREEN, 5),
        "B. musculus": (0.55, 0.73, ACCENT_RED, 5),
    }

    edges = [
        ("Cetacea", "Odontoceti"), ("Cetacea", "Mysticeti"),
        ("Odontoceti", "Physeteridae"), ("Odontoceti", "Delphinidae"),
        ("Mysticeti", "Balaenopteridae"),
        ("Physeteridae", "P. macrocephalus"), ("Physeteridae", "P. catodon"),
        ("Delphinidae", "T. truncatus"),
        ("Balaenopteridae", "B. musculus"),
    ]

    for parent, child in edges:
        px, py, _, _ = taxonomy[parent]
        cx, cy, _, _ = taxonomy[child]
        ax2.plot([px, cx], [py, cy], color=LIGHT_GRAY, lw=0.8, zorder=0)

    for name, (x, y, color, size) in taxonomy.items():
        ax2.scatter(x, y, c=color, s=size * 8, zorder=5,
                    edgecolors=DARK_NAVY, linewidths=0.5)
        ax2.text(x, y - 0.08, name, fontsize=4.5, ha="center", va="top",
                 color=TEXT_DARK, style="italic" if "." in name else "normal")

    ax2.set_xlim(-1.15, 1.15)
    ax2.set_ylim(-0.4, 1.15)
    ax2.set_aspect("equal")
    ax2.axis("off")
    ax2.set_title("Cetacean Taxonomy\non Poincar\u00e9 Disk", fontsize=9,
                  fontweight="bold", color=DARK_NAVY)

    fig.suptitle("Bioacoustics: Spectral Analysis and Hyperbolic Embedding",
                 fontsize=10, fontweight="bold", color=DARK_NAVY, y=1.05)
    fig.tight_layout()
    _save(fig, "ch20-bioacoustics.png")


# ===================================================================
# Generate all illustrations
# ===================================================================
ALL_GENERATORS = [
    ch01_geometric_toolchain,
    ch02_euclidean_vs_mahalanobis,
    ch03_poincare_ball,
    ch04_spd_manifold,
    ch05_tda_persistence,
    ch06_pathfinding,
    ch07_nash_vs_bge,
    ch08_pareto_frontier,
    ch09_robustness,
    ch11_hasse_diagram,
    ch12_interaction_heatmap,
    ch13_dihedral_grid,
    ch16_pipeline_architecture,
    ch20_bioacoustics,
]

def generate_all():
    IMAGES_DIR.mkdir(exist_ok=True)
    print(f"Generating {len(ALL_GENERATORS)} illustrations...")
    for gen in ALL_GENERATORS:
        name = gen.__name__
        print(f"  {name}...")
        try:
            gen()
        except Exception as e:
            print(f"  ERROR in {name}: {e}")
    print("Done.")

if __name__ == "__main__":
    generate_all()
