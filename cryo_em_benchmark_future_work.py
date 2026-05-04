"""
cryo_em_benchmark.py
====================
Synthetic Latent Density for a Cryo-EM Conformational Heterogeneity Benchmark.

Adds the previously listed future-work items as far as possible in a single
self-contained script:
  - coupled z1-z2 latent density
  - exact 1D committor solver (numerical variational solution)
  - projection images with a simple synthetic CTF
  - heterogeneous image dataset + reconstruction diagnostics
  - high-dimensional TT/ALS-ready benchmark builder
  - PCA analysis of decoded volumes and simulated images

Run: python cryo_em_benchmark_future_work.py
"""

import os
import warnings
from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUT_DIR = os.environ.get("CRYO_EM_OUT_DIR", os.path.join(os.path.expanduser("~"), "Downloads"))
try:
    os.makedirs(OUT_DIR, exist_ok=True)
except PermissionError:
    OUT_DIR = os.path.join(os.getcwd(), "cryo_em_outputs")
    os.makedirs(OUT_DIR, exist_ok=True)


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


# ===========================================================================
# 1. LATENT DISTRIBUTION
# ===========================================================================
# z1 ~ double-well  (mixture of two Gaussians at ±1)
# z2 ~ Gaussian, but coupled to z1 by a soft curved seam
# z3 ~ N(0, 0.5²)
# z4 ~ N(0, 0.5²)
#
# The joint density is
#   p(z1,z2,z3,z4) ∝ p(z1) N(z2; α tanh(z1), σ2) N(z3;0,σ3) N(z4;0,σ4)
# which keeps the z1 marginal identical while curving the transition seam in
# the z1-z2 plane.

WELL_MU = 1.0
WELL_SIGMA = 0.35
MIX_WEIGHT = 0.5
Z2_SIGMA = 0.40
Z3_SIGMA = 0.50
Z4_SIGMA = 0.50
COUPLING_ALPHA = 0.55
COUPLING_MODE = "tanh"   # "tanh" or "linear"


def sample_z1(n, rng):
    which = rng.random(n) < MIX_WEIGHT
    return np.where(
        which,
        rng.normal(-WELL_MU, WELL_SIGMA, n),
        rng.normal(+WELL_MU, WELL_SIGMA, n),
    )


def log_p_z1(z1):
    lp_minus = np.log(MIX_WEIGHT) + norm.logpdf(z1, -WELL_MU, WELL_SIGMA)
    lp_plus = np.log(1 - MIX_WEIGHT) + norm.logpdf(z1, +WELL_MU, WELL_SIGMA)
    return np.logaddexp(lp_minus, lp_plus)


def coupling_mean(z1):
    if COUPLING_MODE == "linear":
        return COUPLING_ALPHA * np.asarray(z1)
    return COUPLING_ALPHA * np.tanh(np.asarray(z1))


def sample_latent(n, seed=42):
    rng = np.random.default_rng(seed)
    z1 = sample_z1(n, rng)
    z2 = rng.normal(coupling_mean(z1), Z2_SIGMA, n)
    z3 = rng.normal(0.0, Z3_SIGMA, n)
    z4 = rng.normal(0.0, Z4_SIGMA, n)
    return np.column_stack([z1, z2, z3, z4])


def joint_log_density_z1z2(z1, z2):
    """Analytic log-density for the coupled z1-z2 marginal."""
    return log_p_z1(z1) + norm.logpdf(z2, loc=coupling_mean(z1), scale=Z2_SIGMA)


# ---------------------------------------------------------------------------
# Soft committor proxy and exact 1D committor solver
# ---------------------------------------------------------------------------
def soft_committor(z1):
    """A smooth proxy q̃(z1) in [0,1]."""
    return 1.0 / (1.0 + np.exp(-4.0 * np.asarray(z1)))


def solve_committor_1d(x_grid, potential, a=None, b=None, beta=1.0):
    """Numerical 1D committor solver.

    For the overdamped 1D committor ODE,
        (exp(-βU) q')' = 0,
    the exact solution on [a,b] is
        q(x) = ∫_a^x exp(βU(y)) dy / ∫_a^b exp(βU(y)) dy.

    Parameters
    ----------
    x_grid : array
        Grid points where q is returned.
    potential : array or callable
        U(x) values on x_grid, or a function evaluated on x_grid.
    a, b : float
        Absorbing boundary locations. Defaults to the ends of x_grid.
    beta : float
        Inverse temperature.
    """
    x = np.asarray(x_grid)
    if callable(potential):
        U = np.asarray(potential(x))
    else:
        U = np.asarray(potential)
    if U.shape != x.shape:
        raise ValueError("potential must match x_grid shape")
    if a is None:
        a = float(x[0])
    if b is None:
        b = float(x[-1])

    # Restrict to the interval [a,b] and interpolate the result back.
    mask = (x >= a) & (x <= b)
    x2 = x[mask]
    U2 = U[mask]
    if len(x2) < 2:
        raise ValueError("Need at least two grid points in [a,b].")

    w = np.exp(beta * (U2 - np.min(U2)))
    cum = np.zeros_like(x2)
    cum[1:] = np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(x2))
    total = cum[-1]
    q2 = cum / total if total > 0 else np.zeros_like(cum)
    q2[0] = 0.0
    q2[-1] = 1.0

    q = np.interp(x, x2, q2)
    q[x <= a] = 0.0
    q[x >= b] = 1.0
    return q


def exact_soft_committor(z1_grid):
    """Exact 1D committor for the double-well marginal."""
    z = np.asarray(z1_grid)
    U = -log_p_z1(z)
    # Choose absorbing sets beyond the wells.
    return solve_committor_1d(z, U, a=float(z.min()), b=float(z.max()), beta=1.0)


# ===========================================================================
# 2. VISUALISATION OF THE LATENT DISTRIBUTION
# ===========================================================================

def plot_latent_2d(Z, out_dir):
    q = soft_committor(Z[:, 0])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Synthetic cryo-EM latent distribution  (d = 4)", fontsize=13)

    pairs = [(0, 1, "z₁", "z₂"), (0, 2, "z₁", "z₃"), (1, 2, "z₂", "z₃")]
    for ax, (i, j, xi, xj) in zip(axes, pairs):
        sc = ax.scatter(Z[:, i], Z[:, j], c=q, cmap="coolwarm", s=4, alpha=0.5, rasterized=True)
        ax.set_xlabel(xi, fontsize=11)
        ax.set_ylabel(xj, fontsize=11)
        ax.set_title(f"{xi} vs {xj}", fontsize=11)
        plt.colorbar(sc, ax=ax, label="q̃(z₁)")
    plt.tight_layout()
    save(fig, "latent_scatter_2d.png")


def plot_z1_marginal(Z, out_dir):
    z1 = Z[:, 0]
    xs = np.linspace(-2.5, 2.5, 400)
    p = np.exp(log_p_z1(xs))
    p /= np.trapezoid(p, xs)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(z1, bins=80, density=True, alpha=0.55, color="steelblue", label="samples")
    ax.plot(xs, p, "r-", lw=2, label="analytic p(z₁)")
    ax.axvline(0, ls="--", color="gray", lw=1)
    ax.set_xlabel("z₁  (conformational coordinate)", fontsize=11)
    ax.set_ylabel("density", fontsize=11)
    ax.set_title("Double-well marginal  p(z₁)", fontsize=12)
    ax.legend()
    plt.tight_layout()
    save(fig, "z1_marginal.png")


def plot_latent_3d(Z, out_dir, n_show=3000):
    idx = np.random.default_rng(0).choice(len(Z), size=min(n_show, len(Z)), replace=False)
    Zs = Z[idx]
    q = soft_committor(Zs[:, 0])

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(Zs[:, 0], Zs[:, 1], Zs[:, 2], c=q, cmap="coolwarm", s=6, alpha=0.6)
    ax.set_xlabel("z₁"); ax.set_ylabel("z₂"); ax.set_zlabel("z₃")
    ax.set_title("3D latent scatter  (coloured by q̃)", fontsize=12)
    fig.colorbar(sc, ax=ax, label="q̃(z₁)", shrink=0.6)
    plt.tight_layout()
    save(fig, "latent_scatter_3d.png")


def plot_contour_z1z2(out_dir, ngrid=200):
    z1v = np.linspace(-2.5, 2.5, ngrid)
    z2v = np.linspace(-1.7, 1.7, ngrid)
    Z1, Z2 = np.meshgrid(z1v, z2v)
    log_p = joint_log_density_z1z2(Z1, Z2)
    P = np.exp(log_p - np.max(log_p))

    fig, ax = plt.subplots(figsize=(6, 5))
    cf = ax.contourf(Z1, Z2, P, levels=30, cmap="viridis")
    ax.contour(Z1, Z2, P, levels=8, colors="white", linewidths=0.5, alpha=0.5)
    plt.colorbar(cf, ax=ax, label="p(z₁, z₂)  (unnorm.)")
    ax.set_xlabel("z₁"); ax.set_ylabel("z₂")
    ax.set_title("Coupled latent density  p(z₁, z₂)", fontsize=12)
    plt.tight_layout()
    save(fig, "latent_contour_z1z2.png")


def plot_committor_heatmap(out_dir, ngrid=500):
    z1v = np.linspace(-2.5, 2.5, ngrid)
    log_p = log_p_z1(z1v)
    energy = -log_p + np.max(log_p)
    q_proxy = soft_committor(z1v)
    q_exact = exact_soft_committor(z1v)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax1.plot(z1v, energy, "b-", lw=2.5, label=r"$-\log\,p(z_1)$  (energy)")
    ax1.fill_between(z1v, energy, alpha=0.15, color="blue")
    ax1.set_ylabel("Free energy  [a.u.]", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.set_title("Double-well free energy landscape", fontsize=12)
    ax1.axvline(0, ls="--", color="gray", lw=1)

    ax2.plot(z1v, q_proxy, "r--", lw=2, label=r"$\tilde{q}(z_1)$  proxy")
    ax2.plot(z1v, q_exact, "k-", lw=2.5, label=r"$q(z_1)$  exact 1D solve")
    ax2.set_xlabel("z₁  (conformational coordinate)", fontsize=11)
    ax2.set_ylabel("q̃", fontsize=11)
    ax2.set_ylim(-0.05, 1.05)
    ax2.axvline(0, ls="--", color="gray", lw=1)
    ax2.axhline(0.5, ls=":", color="orange", lw=1.5, label="q̃ = 0.5  (TS)")
    ax2.legend(fontsize=10)
    ax2.set_title("Soft committor proxy vs exact 1D committor", fontsize=12)

    plt.tight_layout()
    save(fig, "energy_and_committor.png")


# ===========================================================================
# 3. TOY 3-D DECODER  (20 x 20 x 20 voxel grid)
# ===========================================================================
VOX = 20
SIGMA_BLOB = 1.5


def make_grid():
    ax = np.linspace(-1, 1, VOX)
    X, Y, Z_ = np.meshgrid(ax, ax, ax, indexing="ij")
    return np.stack([X, Y, Z_], axis=-1)


GRID = make_grid()


def gaussian_blob(grid, centre, amplitude=1.0, sigma=SIGMA_BLOB):
    centre = np.asarray(centre)
    sigma_u = sigma * (2.0 / VOX)
    sq_dist = np.sum((grid - centre) ** 2, axis=-1)
    return amplitude * np.exp(-sq_dist / (2 * sigma_u ** 2))


def decode(z1, z2=0.0):
    """Toy decoder: z1 controls hinge opening, z2 adds a small tilt."""
    vol = np.zeros((VOX, VOX, VOX))
    vol += gaussian_blob(GRID, [0.0, 0.0, 0.0], amplitude=1.0, sigma=2.0)

    angle = np.clip(z1, -1.5, 1.5) * (np.pi / 3.0)
    tilt = np.clip(z2, -1.0, 1.0) * 0.15
    bx = 0.55 * np.cos(angle)
    by = 0.55 * np.sin(angle)
    bz = tilt
    vol += gaussian_blob(GRID, [bx, by, bz], amplitude=0.85, sigma=1.8)

    cx = bx + 0.38 * np.cos(angle)
    cy = by + 0.38 * np.sin(angle)
    cz = bz + 0.05
    vol += gaussian_blob(GRID, [cx, cy, cz], amplitude=0.55, sigma=1.3)

    vol += gaussian_blob(GRID, [-0.5, 0.2, 0.1], amplitude=0.40, sigma=1.0)
    vol = gaussian_filter(vol, sigma=0.6)
    return vol / vol.max()


def central_slices(vol, title=""):
    mid = VOX // 2
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    slices = [vol[mid, :, :], vol[:, mid, :], vol[:, :, mid]]
    labels = ["YZ slice (x=mid)", "XZ slice (y=mid)", "XY slice (z=mid)"]
    for ax, sl, lab in zip(axes, slices, labels):
        ax.imshow(sl, cmap="gray", origin="lower", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(lab, fontsize=9)
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    return fig


def plot_representative_volumes(out_dir):
    states = [("closed", -1.10, 0.0), ("intermediate", 0.00, 0.0), ("open", +1.10, 0.0)]
    all_vols = {}
    for name, z1, z2 in states:
        vol = decode(z1, z2)
        all_vols[name] = vol
        fig = central_slices(vol, title=f"State: {name}  (z₁={z1:.2f})")
        save(fig, f"volume_{name}.png")

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    mid = VOX // 2
    for row, (name, z1, z2) in enumerate(states):
        vol = all_vols[name]
        for col, (sl, lab) in enumerate([(vol[mid, :, :], "YZ"), (vol[:, mid, :], "XZ"), (vol[:, :, mid], "XY")]):
            axes[row, col].imshow(sl, cmap="gray", origin="lower", vmin=0, vmax=1, interpolation="nearest")
            axes[row, col].set_title(f"{name}  {lab}", fontsize=8)
            axes[row, col].axis("off")
    fig.suptitle("Representative volumes – closed / intermediate / open", fontsize=12)
    plt.tight_layout()
    save(fig, "volumes_panel.png")
    return all_vols


# ===========================================================================
# 4. REACTION-COORDINATE VISUALISATION
# ===========================================================================

def plot_volume_trajectory(out_dir, n_frames=7):
    z1_vals = np.linspace(-1.3, 1.3, n_frames)
    mid = VOX // 2
    fig, axes = plt.subplots(1, n_frames, figsize=(2.2 * n_frames, 3))
    for ax, z1 in zip(axes, z1_vals):
        vol = decode(z1, z2=0.0)
        q = soft_committor(z1)
        ax.imshow(vol[:, :, mid], cmap="gray", origin="lower", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"z₁={z1:+.2f}\nq̃={q:.2f}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Volume XY-slice trajectory  (z₁ sweeping left→right well)", fontsize=11)
    plt.tight_layout()
    save(fig, "volume_trajectory.png")


# ===========================================================================
# 5. PROJECTION IMAGES + SYNTHETIC CTF
# ===========================================================================

def project_volume(vol, axis=2):
    """Simple projection by summing along a viewing axis."""
    img = np.sum(vol, axis=axis)
    img = img / (img.max() + 1e-12)
    return img


def apply_synthetic_ctf(img, defocus=1.8, phase_shift=0.2, b_factor=0.02, envelope_sigma=0.15):
    """A lightweight CTF-like filter in Fourier space.

    This is not a physical microscope model, but it captures the oscillatory
    band-pass behaviour and mild attenuation expected from a CTF-convolved
    projection.
    """
    n, m = img.shape
    fy = np.fft.fftfreq(n)
    fx = np.fft.fftfreq(m)
    FX, FY = np.meshgrid(fx, fy)
    r2 = FX**2 + FY**2
    ctf = -np.sin(np.pi * defocus * r2 + phase_shift) * np.exp(-b_factor * r2)
    envelope = np.exp(-(r2 / (2 * envelope_sigma**2)))
    filt = ctf * envelope

    F = np.fft.fft2(img)
    out = np.real(np.fft.ifft2(F * filt))
    out = out - out.min()
    out = out / (out.max() + 1e-12)
    return out


def simulate_micrograph(z1, z2=0.0, seed=0, noise_sigma=0.05):
    rng = np.random.default_rng(seed)
    vol = decode(z1, z2)
    img = project_volume(vol, axis=2)
    img = apply_synthetic_ctf(img, defocus=1.5 + 0.5 * np.tanh(z1), phase_shift=0.2)
    img = img + rng.normal(0.0, noise_sigma, size=img.shape)
    img = img - img.min()
    img = img / (img.max() + 1e-12)
    return img


def plot_projection_examples(out_dir):
    states = [("closed", -1.10, 0.0), ("intermediate", 0.00, 0.0), ("open", +1.10, 0.0)]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    for ax, (name, z1, z2) in zip(axes, states):
        img = simulate_micrograph(z1, z2, seed=123)
        ax.imshow(img, cmap="gray", origin="lower", interpolation="nearest")
        ax.set_title(f"{name}\nz₁={z1:+.2f}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Synthetic CTF-convolved projection images", fontsize=12)
    plt.tight_layout()
    save(fig, "projection_examples.png")


# ===========================================================================
# 6. HETEROGENEOUS IMAGE DATASET + RECONSTRUCTION / PCA
# ===========================================================================

def make_image_dataset(Z, n_images=2000, seed=11):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Z), size=min(n_images, len(Z)), replace=False)
    Zs = Z[idx]
    imgs = np.empty((len(Zs), VOX, VOX), dtype=float)
    for i, z in enumerate(Zs):
        imgs[i] = simulate_micrograph(z[0], z[1], seed=seed + i)
    return Zs, imgs


def pca_scores(X, n_components=2):
    X = np.asarray(X)
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    scores = Xc @ Vt[:n_components].T
    total = np.sum(S**2)
    evr = (S[:n_components]**2) / total if total > 0 else np.zeros(n_components)
    return scores, Vt[:n_components], evr, X.mean(axis=0)


def fit_linear_regression(X, y):
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1, 1)
    X1 = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return coef.ravel()


def predict_linear_regression(coef, X):
    X1 = np.column_stack([np.ones(len(X)), X])
    return X1 @ coef


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def plot_image_pca_and_reconstruction(Z, out_dir, n_images=1600):
    Zs, imgs = make_image_dataset(Z, n_images=n_images)
    X = imgs.reshape(len(imgs), -1)
    q = soft_committor(Zs[:, 0])

    # PCA on images
    scores, comps, evr, mean_img = pca_scores(X, n_components=2)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sc = axes[0].scatter(scores[:, 0], scores[:, 1], c=Zs[:, 0], cmap="coolwarm", s=10, alpha=0.7)
    axes[0].set_xlabel(f"PC1 ({evr[0]*100:.1f}% var)")
    axes[0].set_ylabel(f"PC2 ({evr[1]*100:.1f}% var)")
    axes[0].set_title("PCA of simulated micrographs")
    plt.colorbar(sc, ax=axes[0], label="z₁")

    sc2 = axes[1].scatter(scores[:, 0], scores[:, 1], c=q, cmap="viridis", s=10, alpha=0.7)
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].set_title("PCA coloured by exact reaction coordinate")
    plt.colorbar(sc2, ax=axes[1], label="q(z₁)")
    plt.tight_layout()
    save(fig, "image_pca.png")

    # Reconstruction: linear decoder from pixels to z1
    n = len(X)
    perm = np.random.default_rng(0).permutation(n)
    split = int(0.75 * n)
    tr, te = perm[:split], perm[split:]
    coef = fit_linear_regression(X[tr], Zs[tr, 0])
    pred = predict_linear_regression(coef, X[te])
    r2 = r2_score(Zs[te, 0], pred)
    print(f"  image-to-z1 linear reconstruction R² = {r2:.3f}")

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.scatter(Zs[te, 0], pred, c=q[te], cmap="coolwarm", s=12, alpha=0.7)
    lims = [min(Zs[te, 0].min(), pred.min()), max(Zs[te, 0].max(), pred.max())]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlabel("true z₁")
    ax.set_ylabel("reconstructed z₁")
    ax.set_title(f"Linear image reconstruction  (R²={r2:.2f})")
    plt.tight_layout()
    save(fig, "reconstruction_scatter.png")

    # Store dataset for later experiments
    np.save(os.path.join(out_dir, "simulated_images.npy"), imgs)
    np.save(os.path.join(out_dir, "simulated_image_latents.npy"), Zs)
    print(f"  saved → {os.path.join(out_dir, 'simulated_images.npy')}")
    print(f"  saved → {os.path.join(out_dir, 'simulated_image_latents.npy')}")


# ===========================================================================
# 7. DENSITY CORRELATION PLOT
# ===========================================================================

def plot_density_correlation(Z, out_dir):
    print("  computing volume projections (this may take ~10 s)...")
    n_sub = 250
    rng = np.random.default_rng(7)
    idx = rng.choice(len(Z), size=n_sub, replace=False)
    Zs = Z[idx]

    v_open = decode(1.1, 0)
    v_closed = decode(-1.1, 0)
    template = v_open - v_closed
    template /= np.linalg.norm(template)

    scores = []
    for z in Zs:
        vol = decode(z[0], z[1])
        scores.append(float(np.sum(vol * template)))
    scores = np.array(scores)
    q = soft_committor(Zs[:, 0])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(Zs[:, 0], scores, c=q, cmap="coolwarm", s=12, alpha=0.7)
    axes[0].set_xlabel("z₁", fontsize=11)
    axes[0].set_ylabel("open/closed projection score", fontsize=11)
    axes[0].set_title("Volume projection score vs z₁", fontsize=12)

    axes[1].scatter(q, scores, c=Zs[:, 0], cmap="RdBu_r", s=12, alpha=0.7)
    axes[1].set_xlabel("q̃(z₁)  exact / proxy", fontsize=11)
    axes[1].set_ylabel("open/closed projection score", fontsize=11)
    axes[1].set_title("Volume score vs reaction coordinate", fontsize=12)
    plt.tight_layout()
    save(fig, "density_correlation.png")


# ===========================================================================
# 8. TT / ALS-READY HIGH-DIMENSIONAL BENCHMARK
# ===========================================================================

def sample_latent_high_dim(n, dim=8, seed=123, coupling=0.25):
    """A simple high-dimensional benchmark distribution.

    The first coordinate is double-well, the second is softly coupled to it,
    and the remaining coordinates are Gaussian nuisance dimensions.
    """
    rng = np.random.default_rng(seed)
    z1 = sample_z1(n, rng)
    z2 = rng.normal(coupling * np.tanh(z1), Z2_SIGMA, n)
    rest = [rng.normal(0.0, 0.50, n) for _ in range(max(dim - 2, 0))]
    cols = [z1, z2] + rest[: max(dim - 2, 0)]
    return np.column_stack(cols)


def build_tt_ready_benchmark(out_dir, dim=8, n=50000, grid_points=25):
    """Prepare a tensor-train friendly benchmark and save compact diagnostics.

    This does not require the external TT/ALS solver to be installed. It creates
    a high-dimensional sample set and coarse grids that can be fed to a TT code
    later. If such a solver is available, this is the natural insertion point.
    """
    Z = sample_latent_high_dim(n, dim=dim)
    q = soft_committor(Z[:, 0])
    np.savez(os.path.join(out_dir, f"tt_ready_benchmark_d{dim}.npz"), Z=Z, q=q)
    print(f"  saved → {os.path.join(out_dir, f'tt_ready_benchmark_d{dim}.npz')}")

    # Coarse 2D diagnostic for the coupled seam.
    z1v = np.linspace(-2.5, 2.5, grid_points)
    z2v = np.linspace(-1.7, 1.7, grid_points)
    Z1, Z2 = np.meshgrid(z1v, z2v)
    P = np.exp(joint_log_density_z1z2(Z1, Z2))
    P /= P.max()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(Z1, Z2, P, levels=20, cmap="magma")
    ax.set_xlabel("z₁")
    ax.set_ylabel("z₂")
    ax.set_title(f"TT-ready coupled density slice  (d={dim})")
    plt.tight_layout()
    save(fig, f"tt_ready_density_d{dim}.png")
    return Z, q


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 60)
    print("Synthetic cryo-EM latent density benchmark")
    print("=" * 60)

    N = 10_000
    print(f"\n[1] Sampling {N} latent configurations...")
    Z = sample_latent(N)
    print(f"    Z shape: {Z.shape}, z1 range: [{Z[:,0].min():.2f}, {Z[:,0].max():.2f}]")

    print("\n[2] Generating latent-space visualisations...")
    plot_z1_marginal(Z, OUT_DIR)
    plot_latent_2d(Z, OUT_DIR)
    plot_latent_3d(Z, OUT_DIR)
    plot_contour_z1z2(OUT_DIR)
    plot_committor_heatmap(OUT_DIR)

    print("\n[3] Generating representative 3D volumes...")
    all_vols = plot_representative_volumes(OUT_DIR)
    for name, vol in all_vols.items():
        path = os.path.join(OUT_DIR, f"volume_{name}.npy")
        np.save(path, vol)
        print(f"    saved → {path}")

    print("\n[4] Generating volume trajectory along reaction coordinate...")
    plot_volume_trajectory(OUT_DIR)

    print("\n[5] Projection-image examples...")
    plot_projection_examples(OUT_DIR)

    print("\n[6] Heterogeneous image dataset + reconstruction/PCA...")
    plot_image_pca_and_reconstruction(Z, OUT_DIR, n_images=800)

    print("\n[7] Density correlation plot (subset of samples)...")
    plot_density_correlation(Z, OUT_DIR)

    print("\n[8] TT/ALS-ready high-dimensional benchmark diagnostics...")
    build_tt_ready_benchmark(OUT_DIR, dim=8, n=12000, grid_points=30)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(
        """
IMPLEMENTED SUCCESSFULLY
------------------------
✓ 4D latent distribution with a coupled z1-z2 seam
✓ Direct sampler (mixture-of-Gaussians for z1, conditional Gaussian z2)
✓ Analytic joint density for z1-z2 and latent contour plots
✓ Proxy committor plus exact 1D numerical committor solver
✓ 2D scatter / contour plots of latent pairs (z1-z2, z1-z3, z2-z3)
✓ 3D scatter (z1, z2, z3) coloured by committor
✓ Energy landscape + exact/proxy committor 1D plot
✓ 20×20×20 voxel toy decoder with hinge-motion molecule model
✓ Central-slice visualisation for closed / intermediate / open states
✓ 7-frame volume trajectory along z1
✓ Synthetic projection images with a CTF-like Fourier filter
✓ Heterogeneous image dataset, PCA, and image→z1 reconstruction
✓ Open/closed template projection score vs z1
✓ TT/ALS-ready high-dimensional benchmark dataset and diagnostics
✓ All figures and .npy/.npz outputs saved to the output directory

INTENTIONAL SIMPLIFICATIONS
----------------------------
• The synthetic CTF is a lightweight Fourier-domain approximation
• The exact committor is solved in 1D only, because the benchmark latent
  reaction coordinate is 1D; a future true TT/ALS solve can be plugged in at
  the marked benchmark builder
• The decoder is analytic and not learned from data
• The image reconstruction is linear, meant as a sanity-check baseline
"""
    )


if __name__ == "__main__":
    main()
