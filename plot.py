"""
visualize.py
────────────
Streamlit dashboard for contrastive trajectory results.

Usage:
    streamlit run visualize.py
"""

import os, glob
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import uniform_filter1d

import config as cfg

st.set_page_config(page_title="Trajectory Viewer", layout="wide")

# ── Helpers ───────────────────────────────────────────────────────────────────
COLORS = ["#7B61FF", "#00C9A7", "#FF6B6B", "#FFA600",
          "#2196F3", "#FF5722", "#9C27B0", "#4CAF50"]

def smooth(arr, w):
    if w <= 1 or len(arr) < w:
        return arr
    return uniform_filter1d(arr.astype(float), size=w, mode="nearest")

def load_result(path: str) -> dict:
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}

def scalar(v):
    """Unwrap 0-d numpy arrays to Python scalars."""
    return v.item() if hasattr(v, "item") else v

def pair_label(r):
    return f'{scalar(r["benign"])}  vs  {scalar(r["harmful"])}'


# ── Load results ──────────────────────────────────────────────────────────────
results_dir = st.sidebar.text_input("Results dir", value=cfg.RESULTS_DIR)
files = sorted(glob.glob(os.path.join(results_dir, "*.npz")))

if not files:
    st.warning(f"No .npz files found in `{results_dir}`. Run `run_experiments.py` first.")
    st.stop()

all_results = {os.path.basename(f): load_result(f) for f in files}

# ── Sidebar controls ──────────────────────────────────────────────────────────
st.sidebar.markdown("### Models")
selected_files = [
    fname for fname in all_results
    if st.sidebar.checkbox(fname, value=True)
]

if not selected_files:
    st.warning("Select at least one model.")
    st.stop()

results = {f: all_results[f] for f in selected_files}

# Prompt pair selector (intersection of available pairs)
all_pairs = sorted({scalar(r["pair"]) for r in results.values()})
sel_pair  = st.sidebar.selectbox("Prompt pair", all_pairs)
results   = {f: r for f, r in results.items()
             if scalar(r["pair"]) == sel_pair}

smooth_w = st.sidebar.slider("Smooth window", 1, 15, cfg.SMOOTH_WINDOW)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_enc, tab_gen, tab_cross = st.tabs(["Encoding", "Generation", "Cross-model"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ENCODING
# ══════════════════════════════════════════════════════════════════════════════
with tab_enc:
    st.subheader("Prompt Encoding — layer trajectories")

    # pick token position (use positions from first result as reference)
    ref        = next(iter(results.values()))
    positions  = ref["enc_positions"].tolist()
    toks_b     = ref["enc_toks_b"].tolist()
    toks_h     = ref["enc_toks_h"].tolist()
    pos_labels = [f"pos {p}: '{toks_b[i]}' vs '{toks_h[i]}'"
                  for i, p in enumerate(positions)]
    sel_pos_label = st.selectbox("Token position", pos_labels)
    pos_idx       = pos_labels.index(sel_pos_label)

    metrics = [
        ("enc_l2",     "L2 Distance",         "L2"),
        ("enc_cos",    "Cosine Similarity",    "cosine sim"),
        ("enc_l2_vel", "L2 Velocity",          "d(L2)/d(layer)"),
    ]

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=[m[1] for m in metrics])

    for ci, (key, title, ylab) in enumerate(metrics, 1):
        for mi, (fname, r) in enumerate(results.items()):
            arr = smooth(r[key][pos_idx], smooth_w)
            fig.add_trace(
                go.Scatter(x=list(range(len(arr))), y=arr,
                           name=scalar(r["model"]),
                           line=dict(color=COLORS[mi % len(COLORS)]),
                           showlegend=(ci == 1)),
                row=1, col=ci,
            )
        fig.update_yaxes(title_text=ylab, row=1, col=ci)
        fig.update_xaxes(title_text="Layer", row=1, col=ci)

    fig.update_layout(height=380, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # Turbulence — one subplot per model (B vs H overlay)
    st.markdown("#### Directional Turbulence (benign vs harmful per model)")
    n_models = len(results)
    fig2 = make_subplots(rows=1, cols=n_models,
                         subplot_titles=[scalar(r["model"])
                                         for r in results.values()])

    for mi, (fname, r) in enumerate(results.items(), 1):
        tb = smooth(r["enc_turb_b"][pos_idx], smooth_w)
        th = smooth(r["enc_turb_h"][pos_idx], smooth_w)
        fig2.add_trace(go.Scatter(x=list(range(len(tb))), y=tb,
                                  name="benign", line=dict(color="#00C9A7"),
                                  showlegend=(mi == 1)), row=1, col=mi)
        fig2.add_trace(go.Scatter(x=list(range(len(th))), y=th,
                                  name="harmful", line=dict(color="#FF6B6B"),
                                  showlegend=(mi == 1)), row=1, col=mi)
        fig2.update_xaxes(title_text="Layer", row=1, col=mi)
        fig2.update_yaxes(title_text="cos(δᵢ, δᵢ₊₁)", row=1, col=mi)

    fig2.update_layout(height=350, margin=dict(t=40, b=20))
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — GENERATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_gen:
    st.subheader("Generation — contrastive dynamics")

    gen_metrics = [
        ("gen_l2",       "Contrastive L2",      "L2"),
        ("gen_cos",      "Contrastive Cosine",   "cosine sim"),
        ("gen_dir_stab", "Direction Stability",  "cosine sim"),
    ]

    fig3 = make_subplots(rows=1, cols=3,
                         subplot_titles=[m[1] for m in gen_metrics])

    for ci, (key, title, ylab) in enumerate(gen_metrics, 1):
        for mi, (fname, r) in enumerate(results.items()):
            arr = smooth(r[key], smooth_w)
            fig3.add_trace(
                go.Scatter(x=list(range(len(arr))), y=arr,
                           name=scalar(r["model"]),
                           line=dict(color=COLORS[mi % len(COLORS)]),
                           showlegend=(ci == 1)),
                row=1, col=ci,
            )
        fig3.update_yaxes(title_text=ylab, row=1, col=ci)
        fig3.update_xaxes(title_text="Gen step", row=1, col=ci)

    fig3.update_layout(height=380, margin=dict(t=40, b=20))
    st.plotly_chart(fig3, use_container_width=True)

    # Directional turbulence B vs H
    st.markdown("#### Directional Turbulence during generation")
    fig4 = make_subplots(rows=1, cols=n_models,
                         subplot_titles=[scalar(r["model"])
                                         for r in results.values()])

    for mi, (fname, r) in enumerate(results.items(), 1):
        tb = smooth(r["gen_turb_b"], smooth_w)
        th = smooth(r["gen_turb_h"], smooth_w)
        fig4.add_trace(go.Scatter(x=list(range(len(tb))), y=tb,
                                  name="benign", line=dict(color="#00C9A7"),
                                  showlegend=(mi == 1)), row=1, col=mi)
        fig4.add_trace(go.Scatter(x=list(range(len(th))), y=th,
                                  name="harmful", line=dict(color="#FF6B6B"),
                                  showlegend=(mi == 1)), row=1, col=mi)
        fig4.update_xaxes(title_text="Gen step", row=1, col=mi)
        fig4.update_yaxes(title_text="turbulence", row=1, col=mi)

    fig4.update_layout(height=350, margin=dict(t=40, b=20))
    st.plotly_chart(fig4, use_container_width=True)

    # Generated text
    with st.expander("Generated text"):
        for fname, r in results.items():
            st.markdown(f"**{scalar(r['model'])}**")
            st.markdown(f"- Benign → *{scalar(r['gen_text_b'])}*")
            st.markdown(f"- Harmful → *{scalar(r['gen_text_h'])}*")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CROSS-MODEL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
with tab_cross:
    st.subheader("Cross-model summary")

    model_names = [scalar(r["model"]) for r in results.values()]

    col1, col2 = st.columns(2)

    # ── Heatmap: mean L2 per layer (encoding) ────────────────────────────────
    with col1:
        st.markdown("##### Mean encoding L2 per layer")
        heat_enc = np.array([
            r["enc_l2"][pos_idx] for r in results.values()
        ])
        fig5 = go.Figure(go.Heatmap(
            z=heat_enc,
            x=[f"L{i}" for i in range(heat_enc.shape[1])],
            y=model_names,
            colorscale="Viridis",
            colorbar=dict(title="L2"),
        ))
        fig5.update_layout(height=300, margin=dict(t=20, b=20))
        st.plotly_chart(fig5, use_container_width=True)

    # ── Heatmap: contrastive L2 per gen step ─────────────────────────────────
    with col2:
        st.markdown("##### Contrastive L2 per gen step")
        max_t    = max(len(r["gen_l2"]) for r in results.values())
        heat_gen = np.full((len(results), max_t), np.nan)
        for i, r in enumerate(results.values()):
            arr = r["gen_l2"]
            heat_gen[i, :len(arr)] = arr

        fig6 = go.Figure(go.Heatmap(
            z=heat_gen,
            x=[f"t{i}" for i in range(max_t)],
            y=model_names,
            colorscale="Plasma",
            colorbar=dict(title="L2"),
        ))
        fig6.update_layout(height=300, margin=dict(t=20, b=20))
        st.plotly_chart(fig6, use_container_width=True)

    # ── Bar charts: scalar summaries ─────────────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("##### Avg direction stability")
        vals = [r["gen_dir_stab"].mean() for r in results.values()]
        fig7 = go.Figure(go.Bar(
            x=model_names, y=vals,
            marker_color=COLORS[:len(model_names)],
        ))
        fig7.update_layout(height=300, yaxis_title="mean cosine sim",
                           margin=dict(t=20, b=20))
        st.plotly_chart(fig7, use_container_width=True)

    with col4:
        st.markdown("##### Avg contrastive L2")
        vals = [r["gen_l2"].mean() for r in results.values()]
        fig8 = go.Figure(go.Bar(
            x=model_names, y=vals,
            marker_color=COLORS[:len(model_names)],
        ))
        fig8.update_layout(height=300, yaxis_title="mean L2",
                           margin=dict(t=20, b=20))
        st.plotly_chart(fig8, use_container_width=True)