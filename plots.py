"""Analysis / visualization (TASK 3).

Every function consumes one or more ``Prediction`` objects, so a neural
predictor, ePBL and persistence all plot through the same calls. ``quantity``
selects the state (M) or tendency (dM, the diff along time) view; ``split``
selects train or val.
"""
import numpy as np
import matplotlib.pyplot as plt


def _as_list(preds):
    return list(preds) if isinstance(preds, (list, tuple)) else [preds]


def _np(x):
    return x.detach().numpy() if hasattr(x, "detach") else np.asarray(x)


# --------------------------------------------------------------------------- #
# Truth-vs-prediction scatter (state or tendency)
# --------------------------------------------------------------------------- #
def scatter(*preds, split="val", quantity="state", ax=None, lim=None,
            s=2, alpha=0.3, one_to_one=True):
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    for pr in preds:
        p, t = pr.split(split).flat(quantity)
        ax.scatter(_np(p), _np(t), s=s, alpha=alpha, label=pr.label)
    if one_to_one and lim is not None:
        ax.plot(lim, lim, "r--", lw=1)
    if lim is not None:
        ax.set_xlim(lim); ax.set_ylim(lim)
    sub = "dM" if quantity == "tendency" else "M"
    ax.set_xlabel(f"${sub}_{{pred}}$"); ax.set_ylabel(f"${sub}_{{true}}$")
    ax.legend()
    return ax


# --------------------------------------------------------------------------- #
# Loss trajectories
# --------------------------------------------------------------------------- #
def loss_trajectory(*preds, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    for pr in preds:
        c = pr.config
        if c is None:
            continue
        ax.plot(c.train_loss_trajectory, label=f"{pr.label} train")
        ax.plot(c.test_loss_trajectory, ls="--", label=f"{pr.label} test")
    ax.set_yscale("log"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend()
    return ax


# --------------------------------------------------------------------------- #
# RMSE summary table
# --------------------------------------------------------------------------- #
def rmse_table(*preds, split="val", quantities=("state", "tendency")):
    import pandas as pd
    rows = [{"label": pr.label,
             **{q: pr.split(split).rmse(q) for q in quantities}} for pr in preds]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Per-case timeseries (M truth/pred, optionally with feature panels)
# --------------------------------------------------------------------------- #
def case_timeseries(preds, cases, split="val", quantity="state", features=None,
                    case_meta=None, figsize=(10, 3)):
    preds = _as_list(preds)
    features = features or []
    ref = preds[0].split(split)          # feature panels + truth come from the first pred
    figs = []
    for i in cases:
        n_panels = 1 + len(features)
        fig, axes = plt.subplots(n_panels, 1, figsize=(figsize[0], figsize[1] * n_panels),
                                 sharex=True, tight_layout=True, squeeze=False)
        axes = axes[:, 0]
        if case_meta is not None:
            c = case_meta[f"case_{i + 1}"]
            fig.suptitle(f"case {i+1}: temp_grad {c.get('temp_grad')}, tx {c.get('tx')}, "
                         f"lat {c.get('lat')}, hf {c.get('heat_flux')}")
        for ax, fname in zip(axes, features):
            j = ref.feature_names.index(fname)
            ax.plot(_np(ref.features[i, :, j]), color="black")
            ax.set_ylabel(fname)
        m_ax = axes[-1]
        for pr in preds:
            _, p = pr.split(split).series(quantity)
            m_ax.plot(_np(p[i]), label=f"{pr.label} $M_{{pred}}$")
        truth, _ = preds[0].split(split).series(quantity)
        m_ax.plot(_np(truth[i]), color="black", label="$M_{true}$")
        m_ax.set_xlabel("Time step [30 min]")
        m_ax.set_ylabel("dM" if quantity == "tendency" else "M")
        m_ax.legend()
        figs.append(fig)
    return figs


# --------------------------------------------------------------------------- #
# Profile hovmoller (for full-profile datasets)
# --------------------------------------------------------------------------- #
def hovmoller(pred, case, feature, split="val", ax=None, cmap="plasma", log=True):
    from matplotlib.colors import LogNorm
    s = pred.split(split)
    prefix = feature + "_"
    cols = [(k, n) for k, n in enumerate(s.feature_names) if n.startswith(prefix)]
    if not cols:
        raise ValueError(f"No profile columns for feature {feature!r}.")
    depth = np.array([float(n[len(prefix):]) for _, n in cols])
    order = np.argsort(depth)
    idx = [cols[o][0] for o in order]
    field = _np(s.features[case])[:, idx].T          # (depth, time)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3))
    norm = LogNorm() if log else None
    mesh = ax.pcolormesh(np.arange(field.shape[1]), depth[order], field,
                         norm=norm, cmap=cmap)
    ax.invert_yaxis()
    ax.set_xlabel("Time step [30 min]"); ax.set_ylabel(feature)
    plt.colorbar(mesh, ax=ax)
    return ax
