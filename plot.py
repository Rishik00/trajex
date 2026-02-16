"""
visualize.py
────────────
Streamlit dashboard for contrastive trajectory results.
Auto-exports a standalone HTML file (tabbed layout) on every run.

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

DARK = dict(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="#fafafa")

def smooth(arr, w):
    if w <= 1 or len(arr) < w:
        return arr
    return uniform_filter1d(arr.astype(float), size=w, mode="nearest")

def load_result(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}

def scalar(v):
    return v.item() if hasattr(v, "item") else v

def dark(fig):
    fig.update_layout(**DARK)
    return fig


# ── HTML export ───────────────────────────────────────────────────────────────
def fig_to_html(fig):
    return fig.to_html(full_html=False, include_plotlyjs=False)

def build_html(sections, title):
    tab_names = list(sections.keys())
    nav   = "\n".join(
        f'<button class="tab-btn" onclick="show(\'{i}\')" id="btn-{i}">{n}</button>'
        for i, n in enumerate(tab_names)
    )
    panes = ""
    for i, (name, figs) in enumerate(sections.items()):
        inner  = "\n".join(fig_to_html(f) for f in figs)
        panes += (f'<div class="tab-pane" id="pane-{i}" '
                  f'style="display:{"block" if i==0 else "none"}">{inner}</div>\n')
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  body {{ font-family:sans-serif; background:#0e1117; color:#fafafa; margin:0; padding:16px; }}
  h1   {{ font-size:1.2rem; margin-bottom:12px; }}
  .tab-btn {{ background:#1e2130; color:#ccc; border:none; padding:8px 18px;
              margin-right:4px; cursor:pointer; border-radius:4px 4px 0 0; }}
  .tab-btn.active {{ background:#7B61FF; color:#fff; }}
  .tab-pane {{ padding-top:12px; }}
</style>
<script>
  function show(id) {{
    document.querySelectorAll('.tab-pane').forEach(p=>p.style.display='none');
    document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
    document.getElementById('pane-'+id).style.display='block';
    document.getElementById('btn-'+id).classList.add('active');
  }}
  window.onload = ()=>show(0);
</script>
</head><body>
<h1>{title}</h1><div>{nav}</div>{panes}
</body></html>"""

def export_html(sections, pair_name):
    html  = build_html(sections, f"Trajectory Viewer — {pair_name}")
    fname = os.path.join(cfg.RESULTS_DIR, f"report__{pair_name}.html")
    with open(fname, "w", encoding="utf-8") as f:
        f.write(html)
    return fname


# ── Load results ──────────────────────────────────────────────────────────────
results_dir = st.sidebar.text_input("Results dir", value=cfg.RESULTS_DIR)
files = sorted(glob.glob(os.path.join(results_dir, "*.npz")))

if not files:
    st.warning(f"No .npz files found in `{results_dir}`. Run `run_experiments.py` first.")
    st.stop()

all_results = {os.path.basename(f): load_result(f) for f in files}

# ── Sidebar controls ──────────────────────────────────────────────────────────
st.sidebar.markdown("### Models")
selected_files = [f for f in all_results if st.sidebar.checkbox(f, value=True)]

if not selected_files:
    st.warning("Select at least one model.")
    st.stop()

results  = {f: all_results[f] for f in selected_files}
all_pairs = sorted({scalar(r["pair"]) for r in results.values()})
sel_pair  = st.sidebar.selectbox("Prompt pair", all_pairs)
results   = {f: r for f, r in results.items() if scalar(r["pair"]) == sel_pair}
smooth_w  = st.sidebar.slider("Smooth window", 1, 15, cfg.SMOOTH_WINDOW)
n_models  = len(results)

html_sections = {"Encoding": [], "Attention Divergence": [], "Generation": [], "Cross-model": []}

tab_enc, tab_div, tab_gen, tab_cross = st.tabs(
    ["Encoding", "Attention Divergence", "Generation", "Cross-model"]
)

ref      = next(iter(results.values()))
positions = ref["enc_positions"].tolist()
toks_b   = ref["enc_toks_b"].tolist()
toks_h   = ref["enc_toks_h"].tolist()
pos_labels = [f"pos {p}: '{toks_b[i]}' vs '{toks_h[i]}'"
              for i, p in enumerate(positions)]


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ENCODING
# ══════════════════════════════════════════════════════════════════════════════
with tab_enc:
    st.subheader("Prompt Encoding — layer trajectories")
    sel_pos   = st.selectbox("Token position", pos_labels, key="enc_pos")
    pos_idx   = pos_labels.index(sel_pos)

    # L2 / cosine / velocity
    metrics = [
        ("enc_l2",     "L2 Distance",      "L2"),
        ("enc_cos",    "Cosine Similarity", "cosine sim"),
        ("enc_l2_vel", "L2 Velocity",       "d(L2)/d(layer)"),
    ]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[m[1] for m in metrics])
    for ci, (key, _, ylab) in enumerate(metrics, 1):
        for mi, (fname, r) in enumerate(results.items()):
            arr = smooth(r[key][pos_idx], smooth_w)
            fig.add_trace(go.Scatter(x=list(range(len(arr))), y=arr,
                                     name=scalar(r["model"]),
                                     line=dict(color=COLORS[mi % len(COLORS)]),
                                     showlegend=(ci == 1)), row=1, col=ci)
        fig.update_yaxes(title_text=ylab, row=1, col=ci)
        fig.update_xaxes(title_text="Layer", row=1, col=ci)
    fig.update_layout(height=380, margin=dict(t=40, b=20), **DARK)
    st.plotly_chart(fig, use_container_width=True)
    html_sections["Encoding"].append(fig)

    # Directional turbulence
    st.markdown("#### Directional Turbulence (benign vs harmful per model)")
    fig2 = make_subplots(rows=1, cols=n_models,
                         subplot_titles=[scalar(r["model"]) for r in results.values()])
    for mi, (fname, r) in enumerate(results.items(), 1):
        tb = smooth(r["enc_turb_b"][pos_idx], smooth_w)
        th = smooth(r["enc_turb_h"][pos_idx], smooth_w)
        fig2.add_trace(go.Scatter(x=list(range(len(tb))), y=tb, name="benign",
                                  line=dict(color="#00C9A7"), showlegend=(mi==1)),
                       row=1, col=mi)
        fig2.add_trace(go.Scatter(x=list(range(len(th))), y=th, name="harmful",
                                  line=dict(color="#FF6B6B"), showlegend=(mi==1)),
                       row=1, col=mi)
        fig2.update_xaxes(title_text="Layer", row=1, col=mi)
        fig2.update_yaxes(title_text="cos(δᵢ, δᵢ₊₁)", row=1, col=mi)
    fig2.update_layout(height=350, margin=dict(t=40, b=20), **DARK)
    st.plotly_chart(fig2, use_container_width=True)
    html_sections["Encoding"].append(fig2)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ATTENTION DIVERGENCE
# ══════════════════════════════════════════════════════════════════════════════
with tab_div:
    st.subheader("Attention Divergence — token sink/source dynamics")
    st.caption("div[l, i] = Σⱼ A[j,i] − Σⱼ A[i,j]  |  positive = sink, negative = source")

    # Token selector — use benign tokens as labels (shared prefix)
    tok_labels = [f"{i}: '{toks_b[i]}'" for i in range(len(toks_b))]
    sel_toks   = st.multiselect("Tokens to highlight", tok_labels,
                                default=tok_labels[:min(4, len(tok_labels))],
                                key="div_toks")
    sel_tok_idx = [tok_labels.index(t) for t in sel_toks]

    # ── Divergence across layers per selected token ───────────────────────────
    st.markdown("#### Divergence per layer")
    for mi, (fname, r) in enumerate(results.items()):
        mname = scalar(r["model"])
        div_b = r["enc_div_b"]   # (al, S)
        div_h = r["enc_div_h"]

        fig_d = make_subplots(rows=1, cols=2,
                              subplot_titles=[f"{mname} — benign",
                                             f"{mname} — harmful"])
        for ti, tok_i in enumerate(sel_tok_idx):
            col = COLORS[ti % len(COLORS)]
            lab = tok_labels[tok_i]
            if tok_i < div_b.shape[1]:
                ab = smooth(div_b[:, tok_i], smooth_w)
                ah = smooth(div_h[:, tok_i], smooth_w) if tok_i < div_h.shape[1] else ab
                fig_d.add_trace(go.Scatter(x=list(range(len(ab))), y=ab,
                                           name=lab, line=dict(color=col),
                                           showlegend=True), row=1, col=1)
                fig_d.add_trace(go.Scatter(x=list(range(len(ah))), y=ah,
                                           name=lab, line=dict(color=col),
                                           showlegend=False), row=1, col=2)
        fig_d.add_hline(y=0, line_dash="dash", line_color="gray",
                        opacity=0.4, row=1, col=1)
        fig_d.add_hline(y=0, line_dash="dash", line_color="gray",
                        opacity=0.4, row=1, col=2)
        fig_d.update_xaxes(title_text="Layer")
        fig_d.update_yaxes(title_text="divergence")
        fig_d.update_layout(height=350, margin=dict(t=40, b=20), **DARK)
        st.plotly_chart(fig_d, use_container_width=True)
        html_sections["Attention Divergence"].append(fig_d)

    # ── Contrastive divergence diff (harmful - benign) ────────────────────────
    st.markdown("#### Contrastive divergence (harmful − benign) per layer")
    fig_cd = make_subplots(rows=1, cols=n_models,
                           subplot_titles=[scalar(r["model"]) for r in results.values()])
    for mi, (fname, r) in enumerate(results.items(), 1):
        S    = min(r["enc_div_b"].shape[1], r["enc_div_h"].shape[1])
        diff = r["enc_div_h"][:, :S] - r["enc_div_b"][:, :S]   # (al, S)
        for ti, tok_i in enumerate(sel_tok_idx):
            if tok_i < S:
                arr = smooth(diff[:, tok_i], smooth_w)
                fig_cd.add_trace(
                    go.Scatter(x=list(range(len(arr))), y=arr,
                               name=tok_labels[tok_i],
                               line=dict(color=COLORS[ti % len(COLORS)]),
                               showlegend=(mi == 1)),
                    row=1, col=mi)
        fig_cd.add_hline(y=0, line_dash="dash", line_color="gray",
                         opacity=0.4, row=1, col=mi)
        fig_cd.update_xaxes(title_text="Layer", row=1, col=mi)
        fig_cd.update_yaxes(title_text="Δ divergence", row=1, col=mi)
    fig_cd.update_layout(height=350, margin=dict(t=40, b=20), **DARK)
    st.plotly_chart(fig_cd, use_container_width=True)
    html_sections["Attention Divergence"].append(fig_cd)

    # ── Divergence velocity (rate of change across layers) ────────────────────
    st.markdown("#### Divergence velocity d(div)/d(layer)")
    for mi, (fname, r) in enumerate(results.items()):
        mname  = scalar(r["model"])
        vel_b  = r["enc_div_vel_b"]   # (al, S)
        vel_h  = r["enc_div_vel_h"]

        fig_v = make_subplots(rows=1, cols=2,
                              subplot_titles=[f"{mname} — benign",
                                             f"{mname} — harmful"])
        for ti, tok_i in enumerate(sel_tok_idx):
            col = COLORS[ti % len(COLORS)]
            lab = tok_labels[tok_i]
            if tok_i < vel_b.shape[1]:
                vb = smooth(vel_b[:, tok_i], smooth_w)
                vh = smooth(vel_h[:, tok_i], smooth_w) if tok_i < vel_h.shape[1] else vb
                fig_v.add_trace(go.Scatter(x=list(range(len(vb))), y=vb,
                                           name=lab, line=dict(color=col),
                                           showlegend=True), row=1, col=1)
                fig_v.add_trace(go.Scatter(x=list(range(len(vh))), y=vh,
                                           name=lab, line=dict(color=col),
                                           showlegend=False), row=1, col=2)
        fig_v.add_hline(y=0, line_dash="dash", line_color="gray",
                        opacity=0.4, row=1, col=1)
        fig_v.add_hline(y=0, line_dash="dash", line_color="gray",
                        opacity=0.4, row=1, col=2)
        fig_v.update_xaxes(title_text="Layer")
        fig_v.update_yaxes(title_text="d(div)/d(layer)")
        fig_v.update_layout(height=350, margin=dict(t=40, b=20), **DARK)
        st.plotly_chart(fig_v, use_container_width=True)
        html_sections["Attention Divergence"].append(fig_v)

    # ── Heatmap: divergence across all tokens × layers (single model) ─────────
    st.markdown("#### Divergence heatmap (all tokens × layers)")
    sel_model_heat = st.selectbox("Model", list(results.keys()), key="div_heat_model")
    r_h     = results[sel_model_heat]
    S       = min(r_h["enc_div_b"].shape[1], r_h["enc_div_h"].shape[1])
    heat_b  = r_h["enc_div_b"][:, :S]
    heat_h  = r_h["enc_div_h"][:, :S]
    heat_diff = heat_h - heat_b

    col_a, col_b_st, col_c = st.columns(3)
    for col_st, heat, title in [
        (col_a,    heat_b,    "Benign"),
        (col_b_st, heat_h,    "Harmful"),
        (col_c,    heat_diff, "Δ (H−B)"),
    ]:
        fh = go.Figure(go.Heatmap(
            z=heat,
            x=[f"{i}:{toks_b[i]}" if i < len(toks_b) else str(i) for i in range(S)],
            y=[f"L{l}" for l in range(heat.shape[0])],
            colorscale="RdBu", zmid=0,
            colorbar=dict(title="div"),
        ))
        fh.update_layout(title=title, height=400,
                         margin=dict(t=40, b=20), **DARK)
        col_st.plotly_chart(fh, use_container_width=True)
        html_sections["Attention Divergence"].append(fh)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — GENERATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_gen:
    st.subheader("Generation — contrastive dynamics")

    gen_metrics = [
        ("gen_l2",       "Contrastive L2",     "L2"),
        ("gen_cos",      "Contrastive Cosine",  "cosine sim"),
        ("gen_dir_stab", "Direction Stability", "cosine sim"),
    ]
    fig3 = make_subplots(rows=1, cols=3,
                         subplot_titles=[m[1] for m in gen_metrics])
    for ci, (key, _, ylab) in enumerate(gen_metrics, 1):
        for mi, (fname, r) in enumerate(results.items()):
            arr = smooth(r[key], smooth_w)
            fig3.add_trace(go.Scatter(x=list(range(len(arr))), y=arr,
                                      name=scalar(r["model"]),
                                      line=dict(color=COLORS[mi % len(COLORS)]),
                                      showlegend=(ci == 1)), row=1, col=ci)
        fig3.update_yaxes(title_text=ylab, row=1, col=ci)
        fig3.update_xaxes(title_text="Gen step", row=1, col=ci)
    fig3.update_layout(height=380, margin=dict(t=40, b=20), **DARK)
    st.plotly_chart(fig3, use_container_width=True)
    html_sections["Generation"].append(fig3)

    st.markdown("#### Directional Turbulence during generation")
    fig4 = make_subplots(rows=1, cols=n_models,
                         subplot_titles=[scalar(r["model"]) for r in results.values()])
    for mi, (fname, r) in enumerate(results.items(), 1):
        tb = smooth(r["gen_turb_b"], smooth_w)
        th = smooth(r["gen_turb_h"], smooth_w)
        fig4.add_trace(go.Scatter(x=list(range(len(tb))), y=tb, name="benign",
                                  line=dict(color="#00C9A7"), showlegend=(mi==1)),
                       row=1, col=mi)
        fig4.add_trace(go.Scatter(x=list(range(len(th))), y=th, name="harmful",
                                  line=dict(color="#FF6B6B"), showlegend=(mi==1)),
                       row=1, col=mi)
        fig4.update_xaxes(title_text="Gen step", row=1, col=mi)
        fig4.update_yaxes(title_text="turbulence", row=1, col=mi)
    fig4.update_layout(height=350, margin=dict(t=40, b=20), **DARK)
    st.plotly_chart(fig4, use_container_width=True)
    html_sections["Generation"].append(fig4)

    with st.expander("Generated text"):
        for fname, r in results.items():
            st.markdown(f"**{scalar(r['model'])}**")
            st.markdown(f"- Benign  → *{scalar(r['gen_text_b'])}*")
            st.markdown(f"- Harmful → *{scalar(r['gen_text_h'])}*")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CROSS-MODEL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
with tab_cross:
    st.subheader("Cross-model summary")
    model_names = [scalar(r["model"]) for r in results.values()]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Mean encoding L2 per layer")
        heat_enc = np.array([r["enc_l2"][pos_idx] for r in results.values()])
        fig5 = go.Figure(go.Heatmap(z=heat_enc,
                                    x=[f"L{i}" for i in range(heat_enc.shape[1])],
                                    y=model_names, colorscale="Viridis",
                                    colorbar=dict(title="L2")))
        fig5.update_layout(height=300, margin=dict(t=20, b=20), **DARK)
        st.plotly_chart(fig5, use_container_width=True)
        html_sections["Cross-model"].append(fig5)

    with col2:
        st.markdown("##### Contrastive L2 per gen step")
        max_t    = max(len(r["gen_l2"]) for r in results.values())
        heat_gen = np.full((len(results), max_t), np.nan)
        for i, r in enumerate(results.values()):
            arr = r["gen_l2"]; heat_gen[i, :len(arr)] = arr
        fig6 = go.Figure(go.Heatmap(z=heat_gen,
                                    x=[f"t{i}" for i in range(max_t)],
                                    y=model_names, colorscale="Plasma",
                                    colorbar=dict(title="L2")))
        fig6.update_layout(height=300, margin=dict(t=20, b=20), **DARK)
        st.plotly_chart(fig6, use_container_width=True)
        html_sections["Cross-model"].append(fig6)

    # Mean attention divergence per model (avg over tokens and layers)
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("##### Mean attention divergence magnitude")
        vals_b = [np.abs(r["enc_div_b"]).mean() for r in results.values()]
        vals_h = [np.abs(r["enc_div_h"]).mean() for r in results.values()]
        fig7 = go.Figure([
            go.Bar(name="benign",  x=model_names, y=vals_b, marker_color="#00C9A7"),
            go.Bar(name="harmful", x=model_names, y=vals_h, marker_color="#FF6B6B"),
        ])
        fig7.update_layout(barmode="group", height=300, yaxis_title="|div| mean",
                           margin=dict(t=20, b=20), **DARK)
        st.plotly_chart(fig7, use_container_width=True)
        html_sections["Cross-model"].append(fig7)

    with col4:
        st.markdown("##### Avg direction stability")
        vals = [r["gen_dir_stab"].mean() for r in results.values()]
        fig8 = go.Figure(go.Bar(x=model_names, y=vals,
                                marker_color=COLORS[:n_models]))
        fig8.update_layout(height=300, yaxis_title="mean cosine sim",
                           margin=dict(t=20, b=20), **DARK)
        st.plotly_chart(fig8, use_container_width=True)
        html_sections["Cross-model"].append(fig8)


# ── Auto-export HTML ──────────────────────────────────────────────────────────
out_path = export_html(html_sections, sel_pair)
st.sidebar.success(f"Exported → `{out_path}`")