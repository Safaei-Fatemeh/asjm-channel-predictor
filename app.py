"""ASJM Channel Predictor — web tool for forward and inverse ASJM channel planning.

Run locally:
    streamlit run app.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from paper_utils import add_physics_features, channel_depth_width  # noqa: E402

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf  # noqa: E402

# ----------------------------------------------------------------------------
# Page config + custom styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="ASJM Channel Predictor",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
  header {visibility: hidden;}

  :root {
    --brand-1: #0b3d91;
    --brand-2: #1f77b4;
    --brand-accent: #ff7043;
    --bg-soft: #f4f7fb;
    --text-soft: #4a5568;
  }

  .hero {
    background: linear-gradient(120deg, #0b3d91 0%, #1f77b4 60%, #2ca5d6 100%);
    padding: 1.6rem 2rem;
    border-radius: 14px;
    color: #ffffff;
    margin-bottom: 1.4rem;
    box-shadow: 0 8px 24px rgba(11, 61, 145, 0.18);
  }
  .hero h1 {margin: 0; font-size: 1.85rem; font-weight: 700; letter-spacing: -0.01em;}
  .hero p {margin: 0.45rem 0 0 0; font-size: 0.98rem; opacity: 0.95;}
  .hero .pill {
    display: inline-block; background: rgba(255,255,255,0.15);
    padding: 2px 10px; border-radius: 999px;
    font-size: 0.82rem; margin-right: 6px; margin-top: 8px;
    border: 1px solid rgba(255,255,255,0.25);
  }

  .kpi {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-left: 4px solid var(--brand-2); border-radius: 10px;
    padding: 0.85rem 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.04);
  }
  .kpi .label {font-size: 0.78rem; text-transform: uppercase;
    letter-spacing: 0.06em; color: var(--text-soft); margin: 0;}
  .kpi .value {font-size: 1.7rem; font-weight: 700; color: var(--brand-1);
    margin: 0.15rem 0 0 0; line-height: 1.1;}
  .kpi .sub {font-size: 0.78rem; color: var(--text-soft); margin: 2px 0 0 0;}
  .kpi.accent {border-left-color: var(--brand-accent);}
  .kpi.accent .value {color: var(--brand-accent);}

  .section-h {
    display: flex; align-items: center; gap: 8px;
    font-size: 1.15rem; font-weight: 600; color: var(--brand-1);
    margin: 1.0rem 0 0.6rem 0; padding-bottom: 4px;
    border-bottom: 2px solid #e2e8f0;
  }
  .section-h .emoji {font-size: 1.25rem;}

  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f4f7fb 0%, #e8eef7 100%);
  }
  section[data-testid="stSidebar"] h1,
  section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3 {color: var(--brand-1);}

  .stButton > button {
    background: linear-gradient(120deg, var(--brand-2), var(--brand-1));
    color: white; border: none; border-radius: 8px;
    padding: 0.55rem 1.4rem; font-weight: 600;
    box-shadow: 0 4px 10px rgba(11,61,145,0.2);
    transition: transform 0.08s, box-shadow 0.12s;
  }
  .stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 14px rgba(11,61,145,0.28);
  }

  .footer-card {
    margin-top: 2rem; padding: 0.9rem 1.1rem;
    background: var(--bg-soft); border-radius: 10px;
    border: 1px solid #e2e8f0; color: var(--text-soft);
    font-size: 0.85rem;
  }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Hero banner
st.markdown(
    """
    <div class="hero">
      <h1>⚙️ ASJM Channel Predictor</h1>
      <p>Plan and verify abrasive-slurry-jet micro-channels <b>before you cut</b>. Predict
      the channel a recipe will produce, or get a recipe for a channel you want —
      ranked, with uncertainty. Powered by a physics-informed ANN trained on 270
      measured profiles.</p>
      <span class="pill">P ∈ [193, 275] MPa</span>
      <span class="pill">SOD ∈ [1, 5] mm</span>
      <span class="pill">V ∈ [500, 1500] mm/min</span>
      <span class="pill">N ∈ [1, 10] passes</span>
    </div>
    """,
    unsafe_allow_html=True,
)


ENVELOPE = {
    "P":   (193.0, 275.0,  "MPa"),
    "SOD": (1.0,   5.0,    "mm"),
    "V":   (500.0, 1500.0, "mm/min"),
    "N":   (1,     10,     "passes"),
}


# ----------------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------------
@st.cache_resource
def load_forward_model():
    model = tf.keras.models.load_model(ROOT / "forward_pi_ann.h5", compile=False)
    sc = np.load(ROOT / "forward_pi_scalers.npz")
    return model, sc


@st.cache_data
def load_x_positions():
    raw = pd.read_csv(ROOT / "Experimental_profiles_fitted.csv", header=None)
    return raw.iloc[4:, 0].to_numpy(dtype=float)


def predict_profile(model, sc, P, SOD, V, N):
    raw = np.array([[P, SOD, V, N]], dtype=float)
    feats = add_physics_features(raw)
    X_s = (feats - sc["X_mean"]) / sc["X_scale"]
    Y_s = model(X_s, training=False).numpy()
    return (Y_s * sc["y_scale"] + sc["y_mean"])[0]


def predict_batch(model, sc, params):
    feats = add_physics_features(params)
    X_s = (feats - sc["X_mean"]) / sc["X_scale"]
    Y_s = model(X_s, training=False).numpy()
    return Y_s * sc["y_scale"] + sc["y_mean"]


def project_to_envelope(params):
    out = params.copy()
    out[:, 0] = np.clip(out[:, 0], *ENVELOPE["P"][:2])
    out[:, 1] = np.clip(out[:, 1], *ENVELOPE["SOD"][:2])
    out[:, 2] = np.clip(out[:, 2], *ENVELOPE["V"][:2])
    out[:, 3] = np.clip(np.round(out[:, 3]), ENVELOPE["N"][0], ENVELOPE["N"][1])
    return out


def random_search_inverse(model, sc, x_pos, target_depth_mm, target_width_mm,
                           K=50, n_random=5000, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.uniform(*ENVELOPE["P"][:2],   size=n_random)
    S = rng.uniform(*ENVELOPE["SOD"][:2], size=n_random)
    V = rng.uniform(*ENVELOPE["V"][:2],   size=n_random)
    N = rng.integers(ENVELOPE["N"][0], ENVELOPE["N"][1] + 1, size=n_random)
    cands = project_to_envelope(np.stack([P, S, V, N.astype(float)], axis=1))
    profs = predict_batch(model, sc, cands)
    depths = np.abs(profs.min(axis=1))
    widths = np.zeros(profs.shape[0])
    for i, prof in enumerate(profs):
        _, w = channel_depth_width(prof, x_pos)
        widths[i] = w
    err = (np.abs(depths - target_depth_mm) / max(target_depth_mm, 1e-6)
           + np.abs(widths - target_width_mm) / max(target_width_mm, 1e-6))
    order = np.argsort(err)[:K]
    return cands[order], profs[order], depths[order], widths[order], err[order]


def kpi_card(col, label, value, sub="", accent=False):
    cls = "kpi accent" if accent else "kpi"
    col.markdown(
        f'<div class="{cls}"><p class="label">{label}</p>'
        f'<p class="value">{value}</p><p class="sub">{sub}</p></div>',
        unsafe_allow_html=True,
    )


def styled_profile_plot(x_pos, profile, title, depth_mm, width_mm):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fbfcfe")
    ax.fill_between(x_pos, profile, 0, where=(profile <= 0), alpha=0.25, color="#1f77b4")
    ax.plot(x_pos, profile, lw=2.2, color="#0b3d91", label="predicted profile")
    ax.axhline(0, color="#222", lw=0.8)
    ax.set_xlabel("Lateral position x (mm)", fontsize=11)
    ax.set_ylabel("Surface height y (mm)  (negative = into workpiece)", fontsize=10)
    ax.set_title(title, fontsize=12, weight="bold", color="#0b3d91", pad=12)
    ax.grid(True, alpha=0.25, ls="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=9)
    ax.text(0.02, 0.06,
            f"depth {depth_mm*1000:.1f} µm   width {width_mm*1000:.1f} µm",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(facecolor="white", edgecolor="#cbd5e0", boxstyle="round,pad=0.35"))
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔧 Mode")
    mode = st.radio(
        " ",
        ["▶ Forward — predict channel from parameters",
         "◀ Inverse — recommend parameters from target"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("#### 📊 Model")
    st.markdown(
        "- **Architecture:** PI-ANN (128→64, dropout 0.2)\n"
        "- **Features:** 9 engineered physics features\n"
        "- **Training:** 270 samples, monotonicity penalty λ = 0.05\n"
        "- **Inference:** ≈ 100 ms / sample (CPU)"
    )
    st.markdown("---")
    st.caption("v1.0 · always validate with a pilot cut before production")


with st.spinner("Loading trained PI-ANN…"):
    model, sc = load_forward_model()
    x_pos = load_x_positions()


# ----------------------------------------------------------------------------
# FORWARD MODE
# ----------------------------------------------------------------------------
if mode.startswith("▶"):
    st.markdown('<div class="section-h"><span class="emoji">▶</span> Forward prediction</div>',
                unsafe_allow_html=True)
    st.caption("Adjust the four ASJM process parameters; the predicted channel cross-section "
               "updates instantly.")

    c1, c2 = st.columns([1, 2.2], gap="large")

    with c1:
        st.markdown("##### Process parameters")
        P = st.slider("💧 Pump pressure P (MPa)",  *ENVELOPE["P"][:2],   value=234.0, step=1.0)
        SOD = st.slider("📏 Stand-off distance SOD (mm)", *ENVELOPE["SOD"][:2], value=3.0, step=0.1)
        V = st.slider("➡️ Traverse speed V (mm/min)", *ENVELOPE["V"][:2], value=1000.0, step=10.0)
        N = st.slider("🔁 Number of passes N", ENVELOPE["N"][0], ENVELOPE["N"][1], value=5, step=1)

    profile = predict_profile(model, sc, P, SOD, V, float(N))
    depth_mm, width_mm = channel_depth_width(profile, x_pos)
    ar = depth_mm / width_mm if width_mm > 1e-6 else 0.0

    with c2:
        title = (f"P = {P:.0f} MPa   |   SOD = {SOD:.2f} mm   |   "
                 f"V = {V:.0f} mm/min   |   N = {int(N)} passes")
        fig = styled_profile_plot(x_pos, profile, title, depth_mm, width_mm)
        st.pyplot(fig, clear_figure=True)

    st.markdown('<div class="section-h"><span class="emoji">📐</span> Predicted geometry</div>',
                unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    kpi_card(k1, "Depth", f"{depth_mm*1000:.1f} µm", "from predicted profile minimum")
    kpi_card(k2, "Width (FWHM)", f"{width_mm*1000:.1f} µm", "at half maximum depth")
    kpi_card(k3, "Aspect ratio", f"{ar:.2f}",
             "high-AR (> 1.5) is the hardest regime", accent=(ar >= 1.5))
    bin_label = ("low" if ar < 0.5 else ("medium" if ar < 1.5 else "high"))
    kpi_card(k4, "AR bin", bin_label,
             "follows §4.1 stratification", accent=(bin_label == "high"))


# ----------------------------------------------------------------------------
# INVERSE MODE
# ----------------------------------------------------------------------------
else:
    st.markdown('<div class="section-h"><span class="emoji">◀</span> Inverse parameter recommendation</div>',
                unsafe_allow_html=True)
    st.caption("Specify a desired channel depth and width. The app searches the OMAX operating "
               "envelope and returns the K best envelope-feasible parameter sets, with their "
               "empirical spread quantifying the inverse problem's many-to-one ambiguity.")

    c1, c2 = st.columns([1, 2.2], gap="large")

    with c1:
        st.markdown("##### Target geometry")
        target_depth_um = st.number_input("🎯 Target depth (µm)", min_value=10.0, max_value=2000.0,
                                          value=400.0, step=10.0)
        target_width_um = st.number_input("🎯 Target width FWHM (µm)", min_value=100.0, max_value=2000.0,
                                          value=400.0, step=10.0)
        st.markdown("##### Search settings")
        K = st.slider("Number of candidates K", 1, 50, value=10)
        n_random = st.select_slider("Search budget", options=[1000, 2000, 5000, 10000, 20000],
                                    value=5000)
        seed = st.number_input("Random seed", value=0, step=1)
        run = st.button("🔍 Search candidate parameters", type="primary", use_container_width=True)

    if run:
        with st.spinner(f"Evaluating {n_random} envelope-feasible candidates…"):
            cands, profs, depths, widths, errs = random_search_inverse(
                model, sc, x_pos,
                target_depth_um / 1000.0, target_width_um / 1000.0,
                K=K, n_random=int(n_random), seed=int(seed),
            )

        with c2:
            best_d, best_w = depths[0], widths[0]
            title = (f"K = {K} candidate reconstructions   |   "
                     f"target {target_depth_um:.0f} µm × {target_width_um:.0f} µm")
            fig, ax = plt.subplots(figsize=(8, 4.8))
            fig.patch.set_facecolor("#ffffff")
            ax.set_facecolor("#fbfcfe")
            for i, prof in enumerate(profs[:min(K, 10)]):
                ax.plot(x_pos, prof, lw=1.0, alpha=0.55,
                        color="#1f77b4", label=f"#{i+1}" if i < 5 else None)
            ax.plot(x_pos, profs[0], lw=2.4, color="#0b3d91", label="best candidate")
            ax.axhline(-target_depth_um / 1000.0, color="#ff7043", ls="--", lw=1.4,
                       label=f"target depth {target_depth_um:.0f} µm")
            ax.axhline(0, color="#222", lw=0.8)
            ax.set_xlabel("Lateral position x (mm)", fontsize=11)
            ax.set_ylabel("Surface height y (mm)", fontsize=11)
            ax.set_title(title, fontsize=12, weight="bold", color="#0b3d91", pad=12)
            ax.grid(True, alpha=0.25, ls="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)

        st.markdown('<div class="section-h"><span class="emoji">🏆</span> Best candidate</div>',
                    unsafe_allow_html=True)
        b = cands[0]
        k1, k2, k3, k4 = st.columns(4)
        kpi_card(k1, "Pressure P",   f"{b[0]:.1f} MPa")
        kpi_card(k2, "Stand-off SOD", f"{b[1]:.2f} mm")
        kpi_card(k3, "Speed V",      f"{b[2]:.0f} mm/min")
        kpi_card(k4, "Passes N",     f"{int(b[3])}")
        e1, e2, e3 = st.columns(3)
        kpi_card(e1, "Predicted depth", f"{best_d*1000:.1f} µm",
                 f"target {target_depth_um:.0f} µm", accent=True)
        kpi_card(e2, "Predicted width", f"{best_w*1000:.1f} µm",
                 f"target {target_width_um:.0f} µm", accent=True)
        kpi_card(e3, "Combined rel. error", f"{errs[0]*100:.2f} %",
                 "smaller is better", accent=True)

        st.markdown('<div class="section-h"><span class="emoji">📋</span> Top-K candidates</div>',
                    unsafe_allow_html=True)
        df = pd.DataFrame({
            "Rank": np.arange(1, len(cands) + 1),
            "P (MPa)":   cands[:, 0].round(1),
            "SOD (mm)":  cands[:, 1].round(2),
            "V (mm/min)": cands[:, 2].round(0),
            "N (passes)": cands[:, 3].astype(int),
            "Depth (µm)": (depths * 1000).round(1),
            "Width (µm)": (widths * 1000).round(1),
            "Rel. error": errs.round(4),
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-h"><span class="emoji">📈</span> Candidate spread (uncertainty)</div>',
                    unsafe_allow_html=True)
        spread = pd.DataFrame({
            "Parameter": ["P (MPa)", "SOD (mm)", "V (mm/min)", "N (passes)"],
            "Best (#1)": [b[0], b[1], b[2], int(b[3])],
            "Mean": [df["P (MPa)"].mean(), df["SOD (mm)"].mean(),
                      df["V (mm/min)"].mean(), df["N (passes)"].mean()],
            "Std (uncertainty)": [df["P (MPa)"].std(), df["SOD (mm)"].std(),
                                   df["V (mm/min)"].std(), df["N (passes)"].std()],
            "Range": [f"{df['P (MPa)'].min():.0f} – {df['P (MPa)'].max():.0f}",
                       f"{df['SOD (mm)'].min():.1f} – {df['SOD (mm)'].max():.1f}",
                       f"{df['V (mm/min)'].min():.0f} – {df['V (mm/min)'].max():.0f}",
                       f"{df['N (passes)'].min()} – {df['N (passes)'].max()}"],
        })
        st.dataframe(spread.round(2), use_container_width=True, hide_index=True)
        st.info(
            "ℹ️ A small std means the inverse problem has a unique solution at this target. "
            "A large std means several physically distinct recipes can hit the same target — "
            "pick the candidate that best matches secondary objectives such as cycle time or "
            "abrasive consumption."
        )


st.markdown(
    """
    <div class="footer-card">
      <b>Free to use.</b> The model is a planning aid — always validate with a pilot cut
      before production. Predictions are only valid inside the trained operating envelope.
      <br><b>Source code:</b> <a href="https://github.com/Safaei-Fatemeh/asjm-channel-predictor" target="_blank">github.com/Safaei-Fatemeh/asjm-channel-predictor</a>.
      If this tool helps your work, please cite the accompanying paper.
    </div>
    """,
    unsafe_allow_html=True,
)
