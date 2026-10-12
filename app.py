"""Streamlit web UI for TMQA.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import altair as alt
import matplotlib
import numpy as np
import pandas as pd
import streamlit as st

from tmqa.hdr_io import read_hdr, read_ldr
from tmqa.synthetic import make_scene
from tmqa.tmqi import luminance, tmqi
from tmqa.tonemap import OPERATORS, tonemap

st.set_page_config(page_title="TMQA - Tone-Mapping Quality", page_icon="🌅", layout="wide")

OPERATOR_LABELS = {
    "linear": "Linear",
    "logarithmic": "Logarithmic",
    "reinhard_global": "Reinhard (global)",
    "reinhard_local": "Reinhard (local)",
    "drago": "Drago",
    "durand": "Durand (bilateral)",
    "ward": "Ward (histogram)",
    "fattal": "Fattal (gradient)",
}


# --------------------------------------------------------------------------
# cached computation
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_synthetic(seed: int, height: int, width: int) -> np.ndarray:
    return make_scene(height, width, seed)


@st.cache_data(show_spinner=False)
def load_uploaded_hdr(data: bytes, suffix: str) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
        fh.write(data)
    try:
        return read_hdr(fh.name)
    finally:
        Path(fh.name).unlink(missing_ok=True)


@st.cache_data(show_spinner=False)
def run_operator(hdr: np.ndarray, op: str):
    ldr = tonemap(hdr, op)
    res = tmqi(hdr, ldr)
    return ldr.astype(np.uint8), res.as_dict(), [m.astype(np.float32) for m in res.s_maps]


def hdr_false_color(hdr: np.ndarray) -> np.ndarray:
    """Log-luminance false-colour visualisation of an HDR image."""
    L = np.log10(np.maximum(luminance(hdr), 1e-6))
    lo, hi = np.percentile(L, [0.5, 99.9])
    n = np.clip((L - lo) / max(hi - lo, 1e-9), 0, 1)
    return (matplotlib.colormaps["inferno"](n)[..., :3] * 255).astype(np.uint8)


def smap_image(m: np.ndarray) -> np.ndarray:
    return (matplotlib.colormaps["viridis"](np.clip(m, 0, 1))[..., :3] * 255).astype(np.uint8)


# --------------------------------------------------------------------------
# sidebar
# --------------------------------------------------------------------------
st.sidebar.title("🌅 TMQA")
st.sidebar.caption("Objective quality assessment of tone-mapped HDR images (TMQI)")

source = st.sidebar.radio("HDR source", ["Synthetic scene", "Upload HDR file"])
hdr = None
scene_name = ""
if source == "Synthetic scene":
    seed = st.sidebar.number_input("Scene seed", 0, 999, 1)
    size = st.sidebar.select_slider(
        "Resolution", options=["256 x 320", "384 x 512", "512 x 768"], value="384 x 512"
    )
    h, w = (int(v) for v in size.split(" x "))
    hdr = load_synthetic(int(seed), h, w)
    scene_name = f"synthetic_{seed:03d}"
else:
    up = st.sidebar.file_uploader("Radiance .hdr / .pfm / .exr", type=["hdr", "pic", "pfm", "exr"])
    if up is not None:
        hdr = load_uploaded_hdr(up.getvalue(), Path(up.name).suffix)
        scene_name = Path(up.name).stem

operators = st.sidebar.multiselect(
    "Tone-mapping operators",
    list(OPERATORS),
    default=list(OPERATORS),
    format_func=OPERATOR_LABELS.get,
)

st.sidebar.markdown(
    "---\n**TMQI** = 0.8012·S^0.3046 + 0.1988·N^0.7088\n\n"
    "- **S** multi-scale structural fidelity\n- **N** statistical naturalness"
)

# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
st.title("Quality Assessment of HDR Tone-Mapped Images")

if hdr is None:
    st.info("Upload an HDR image in the sidebar (e.g. a scene from the TMQI database) to begin.")
    st.stop()
if min(hdr.shape[:2]) < 176:
    st.error("TMQI needs images of at least 176 px on the shorter side (5 scales x 11 px window).")
    st.stop()
if not operators:
    st.warning("Select at least one tone-mapping operator.")
    st.stop()

dr = hdr.max() / max(hdr[hdr > 0].min(), 1e-12)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Scene", scene_name)
c2.metric("Resolution", f"{hdr.shape[1]} x {hdr.shape[0]}")
c3.metric("Dynamic range", f"{dr:.2g} : 1")
c4.metric("Stops", f"{np.log2(dr):.1f}")

progress = st.progress(0.0, text="Tone-mapping and scoring...")
results = {}
for i, op in enumerate(operators):
    results[op] = run_operator(hdr, op)
    progress.progress((i + 1) / len(operators), text=f"Scored {OPERATOR_LABELS[op]}")
progress.empty()

df = pd.DataFrame([{"operator": op, **results[op][1]} for op in operators])
df["label"] = df["operator"].map(OPERATOR_LABELS)
df = df.sort_values("Q", ascending=False).reset_index(drop=True)
df.insert(0, "rank", np.arange(1, len(df) + 1))

tab_bench, tab_inspect, tab_score = st.tabs(
    ["📊 Operator benchmark", "🔍 Inspect structural fidelity", "🖼️ Score your own image"]
)

# ---- benchmark ------------------------------------------------------------
with tab_bench:
    best = df.iloc[0]
    st.success(
        f"**Best operator: {best['label']}** — TMQI Q = {best['Q']:.4f} "
        f"(S = {best['S']:.4f}, N = {best['N']:.4f})"
    )
    left, right = st.columns([1, 1])
    with left:
        long = df.melt(id_vars=["label"], value_vars=["Q", "S", "N"], var_name="score")
        chart = (
            alt.Chart(long)
            .mark_bar()
            .encode(
                y=alt.Y("label:N", sort=list(df["label"]), title=None),
                x=alt.X("value:Q", scale=alt.Scale(domain=[0, 1]), title=None),
                color=alt.Color(
                    "score:N",
                    scale=alt.Scale(domain=["Q", "S", "N"], range=["#2a6fdb", "#20a39e", "#f4a259"]),
                    legend=alt.Legend(orient="top", title=None),
                ),
                yOffset="score:N",
                tooltip=["label", "score", alt.Tooltip("value:Q", format=".4f")],
            )
            .properties(height=60 * len(df))
        )
        st.altair_chart(chart, use_container_width=True)
    with right:
        st.dataframe(
            df[["rank", "label", "Q", "S", "N"]].rename(columns={"label": "operator"}),
            hide_index=True,
            use_container_width=True,
            column_config={
                "rank": st.column_config.NumberColumn("#", width="small"),
                "Q": st.column_config.ProgressColumn("TMQI Q", format="%.4f", min_value=0, max_value=1),
                "S": st.column_config.NumberColumn("S", format="%.4f"),
                "N": st.column_config.NumberColumn("N", format="%.4f"),
            },
        )
        st.download_button(
            "Download scores (CSV)",
            df.drop(columns=["label"]).to_csv(index=False),
            file_name=f"tmqi_{scene_name}.csv",
        )

    st.subheader("Tone-mapped results")
    cols = st.columns(4)
    for i, row in df.iterrows():
        with cols[i % 4]:
            st.image(
                results[row["operator"]][0],
                caption=f"#{row['rank']} {row['label']} · Q={row['Q']:.3f}",
                use_container_width=True,
            )
    with st.expander("HDR luminance (log false colour)"):
        st.image(hdr_false_color(hdr), use_container_width=True)

# ---- inspect --------------------------------------------------------------
with tab_inspect:
    op = st.selectbox(
        "Operator", list(df["operator"]), format_func=OPERATOR_LABELS.get, key="inspect_op"
    )
    ldr, scores, smaps = results[op]
    a, b = st.columns(2)
    a.image(ldr, caption=f"{OPERATOR_LABELS[op]} (tone-mapped)", use_container_width=True)
    b.image(hdr_false_color(hdr), caption="HDR log-luminance", use_container_width=True)
    st.markdown(
        "**Structural fidelity maps per scale** (bright = local structure preserved; "
        "scale 1 is the finest)."
    )
    mcols = st.columns(len(smaps))
    for k, (col, m) in enumerate(zip(mcols, smaps)):
        col.image(smap_image(m), caption=f"Scale {k + 1}: S{k + 1} = {scores[f'S{k + 1}']:.3f}",
                  use_container_width=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("TMQI (Q)", f"{scores['Q']:.4f}")
    m2.metric("Structural fidelity (S)", f"{scores['S']:.4f}")
    m3.metric("Naturalness (N)", f"{scores['N']:.4f}")

# ---- score uploads --------------------------------------------------------
with tab_score:
    st.write(
        "Upload tone-mapped versions of the current HDR scene (same resolution), e.g. the "
        "eight images of a TMQI database scene, to score and rank them."
    )
    files = st.file_uploader(
        "Tone-mapped images", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
        accept_multiple_files=True,
    )
    if files:
        rows, images = [], {}
        for f in files:
            with tempfile.NamedTemporaryFile(suffix=Path(f.name).suffix, delete=False) as fh:
                fh.write(f.getvalue())
            ldr = read_ldr(fh.name)
            Path(fh.name).unlink(missing_ok=True)
            if ldr.shape[:2] != hdr.shape[:2]:
                st.error(f"{f.name}: size {ldr.shape[1]}x{ldr.shape[0]} does not match the HDR image")
                continue
            r = tmqi(hdr, ldr)
            rows.append({"image": f.name, "Q": r.Q, "S": r.S, "N": r.N})
            images[f.name] = ldr.astype(np.uint8)
        if rows:
            up_df = pd.DataFrame(rows).sort_values("Q", ascending=False)
            st.dataframe(up_df, hide_index=True, use_container_width=True)
            cols = st.columns(4)
            for i, row in enumerate(up_df.itertuples()):
                cols[i % 4].image(images[row.image], caption=f"{row.image} · Q={row.Q:.3f}")
