"""Streamlit frontend for the bodymeasure pipeline.

Run with:

    streamlit run bodymeasure/ui/streamlit_app.py

Or, with config + corrector overrides via environment variables:

    BODYMEASURE_CONFIG=configs/default.yaml \
    BODYMEASURE_CORRECTOR=checkpoints/v1/corrector.pt \
    BODYMEASURE_DEVICE=cuda \
    streamlit run bodymeasure/ui/streamlit_app.py

What the UI does:
    - Sidebar: configuration (gender, height, optional side image, device)
    - Main area: front-image upload + (optional) side-image upload
    - On submit: runs the pipeline, then shows
        * the predicted measurements with uncertainty bands
        * the QC report (warnings / failures / metrics)
        * the segmentation overlay
        * the 3D fitted mesh (interactive plotly viewer)
        * raw JSON for debugging

Design choices:
    - Pipeline is loaded ONCE at first request and cached via @st.cache_resource
      so subsequent runs in the same session are fast.
    - Files are written to a tempdir per-request because the pipeline reads
      from disk paths (so EXIF parsing works correctly).
    - The 3D viewer uses plotly Mesh3d — no extra dependencies beyond what's
      already in requirements.txt.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Make the package importable when streamlit cd's into a different directory
import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from bodymeasure import __version__
from bodymeasure.core.config import Config, default_config
from bodymeasure.core.pipeline import Pipeline
from bodymeasure.core.types import Gender, PredictionResult
from bodymeasure.finetune.corrector import ShapeCorrector


# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="bodymeasure",
    page_icon="📏",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📏 bodymeasure")
st.caption(
    f"v{__version__} — image-based body measurement. "
    "Outputs are estimates with uncertainty bands, not validated truths. "
    "Read the README for details."
)


# ----------------------------------------------------------------------------
# Pipeline loading (cached)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading pipeline (one-time, can take ~30s)...")
def _load_pipeline(config_path: str | None, corrector_path: str | None, device: str) -> Pipeline:
    cfg = Config.from_yaml(Path(config_path)) if config_path else default_config()
    corrector = None
    if corrector_path:
        corrector = ShapeCorrector(Path(corrector_path), device=device)
    return Pipeline(cfg, device=device, corrector=corrector)


# ----------------------------------------------------------------------------
# Sidebar — configuration
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")

    config_default = os.environ.get("BODYMEASURE_CONFIG", "")
    config_path = st.text_input(
        "Config YAML path",
        value=config_default,
        help="Leave empty to use built-in defaults. Pointing this at configs/mock.yaml "
             "lets the UI run without any model downloads."
    ).strip() or None

    corrector_default = os.environ.get("BODYMEASURE_CORRECTOR", "")
    corrector_path = st.text_input(
        "Fine-tuned corrector path",
        value=corrector_default,
        help="Optional. Path to a corrector.pt produced by scripts/run_finetune.py. "
             "Leave empty to run without correction."
    ).strip() or None

    device = st.selectbox(
        "Device",
        options=["cpu", "cuda", "mps"],
        index=["cpu", "cuda", "mps"].index(os.environ.get("BODYMEASURE_DEVICE", "cpu")),
        help="Use 'cuda' if you have an NVIDIA GPU, 'mps' on Apple Silicon."
    )

    st.divider()
    st.header("Subject")
    gender = st.selectbox(
        "Gender",
        options=["neutral", "male", "female"],
        index=0,
        help="Affects which SMPL-X gendered model is used. 'neutral' is a fallback; "
             "male/female are noticeably more accurate for shape recovery."
    )
    height_cm = st.number_input(
        "Height (cm)",
        min_value=80.0,
        max_value=240.0,
        value=170.0,
        step=0.5,
        help="Used as a hard constraint during optimization. "
             "Must be measured, not guessed — accuracy of all other measurements "
             "depends on this being correct."
    )

    st.divider()
    if st.button("🔄 Reload pipeline", help="Clear cache and re-load all models"):
        _load_pipeline.clear()
        st.rerun()


# ----------------------------------------------------------------------------
# Main — image upload
# ----------------------------------------------------------------------------
col_upload_a, col_upload_b = st.columns(2)
with col_upload_a:
    st.subheader("Front view (required)")
    front_file = st.file_uploader(
        "Front-facing photo",
        type=["jpg", "jpeg", "png"],
        key="front_uploader",
        help="Standing relaxed, arms slightly out from sides, full body in frame, "
             "camera roughly at chest height, ~3m away."
    )
    if front_file:
        st.image(front_file, caption="Front view", use_container_width=True)

with col_upload_b:
    st.subheader("Side view (recommended)")
    side_file = st.file_uploader(
        "Side-facing photo (optional but improves circumferences)",
        type=["jpg", "jpeg", "png"],
        key="side_uploader",
        help="Same setup as front, rotated 90°. Without a side view, "
             "circumferences (chest/waist/hip) have ~2x the uncertainty."
    )
    if side_file:
        st.image(side_file, caption="Side view", use_container_width=True)


# ----------------------------------------------------------------------------
# Run button
# ----------------------------------------------------------------------------
run = st.button(
    "📐 Compute measurements",
    type="primary",
    disabled=front_file is None,
    use_container_width=True,
)


# ----------------------------------------------------------------------------
# Helper: render the result
# ----------------------------------------------------------------------------
def _measurement_table(result: PredictionResult) -> list[dict]:
    rows = []
    for m in result.measurements:
        if np.isnan(m.value_cm):
            value_str = "—"
            range_str = "—"
        else:
            value_str = f"{m.value_cm:.1f} cm"
            lo = m.value_cm - m.uncertainty_cm
            hi = m.value_cm + m.uncertainty_cm
            range_str = f"{lo:.1f} – {hi:.1f} cm"
        rows.append({
            "Measurement": m.name.replace("_", " ").title(),
            "Estimate": value_str,
            "68% Interval": range_str,
            "Method": m.method,
        })
    return rows


def _render_mesh_3d(result: PredictionResult):
    """Interactive 3D plotly viewer for the fitted mesh."""
    import plotly.graph_objects as go

    fb = result.fitted_body
    if fb is None:
        st.info("No fitted mesh available.")
        return
    v = fb.vertices_m
    f = fb.faces

    fig = go.Figure(data=[
        go.Mesh3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2],
            i=f[:, 0], j=f[:, 1], k=f[:, 2],
            color="lightblue",
            opacity=0.85,
            flatshading=False,
            lighting=dict(ambient=0.4, diffuse=0.8, specular=0.2),
            lightposition=dict(x=100, y=200, z=100),
            name="body",
        )
    ])
    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
            camera=dict(eye=dict(x=0.0, y=0.0, z=2.5), up=dict(x=0, y=1, z=0)),
        ),
        height=560,
        margin=dict(l=0, r=0, t=0, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


def _segmentation_overlay(image_bytes: bytes, mask: np.ndarray | None) -> np.ndarray | None:
    """Tint the image red wherever the mask says background."""
    if mask is None:
        return None
    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = np.array(pil)
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = img.copy()
    bg = mask < 128
    overlay[bg] = (overlay[bg] * 0.4 + np.array([255, 80, 80]) * 0.6).astype(np.uint8)
    return overlay


# ----------------------------------------------------------------------------
# Run the pipeline
# ----------------------------------------------------------------------------
if run and front_file is not None:
    try:
        pipeline = _load_pipeline(config_path, corrector_path, device)
    except Exception as e:
        st.error(f"Failed to load pipeline: {e}")
        st.stop()

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        front_path = td_path / front_file.name
        front_bytes = front_file.getvalue()
        front_path.write_bytes(front_bytes)

        side_path = None
        side_bytes = None
        if side_file is not None:
            side_path = td_path / side_file.name
            side_bytes = side_file.getvalue()
            side_path.write_bytes(side_bytes)

        with st.spinner("Running pipeline... (CPU: ~30-90s, GPU: ~5-15s)"):
            try:
                result = pipeline.predict(
                    front_image_path=front_path,
                    side_image_path=side_path,
                    height_cm=height_cm,
                    gender=Gender(gender),
                )
            except RuntimeError as e:
                st.error(f"Pipeline rejected the input: {e}")
                st.info(
                    "Common causes: photo too low resolution, no person detected, "
                    "person clipped at frame edges, or pose too far from standing relaxed. "
                    "See the README for the recommended capture setup."
                )
                st.stop()
            except Exception as e:
                st.exception(e)
                st.stop()

    # ------- Results -------
    st.success("Done.")

    # ---- Top-line measurements ----
    st.header("Measurements")
    rows = _measurement_table(result)
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.caption(
        "Intervals are ±1σ (about a 68% confidence band given the model's prior uncertainty). "
        "Until you fine-tune on real ground-truth data, these are model-derived priors, "
        "not validated bounds — treat them as rough guidance."
    )

    # ---- QC report ----
    qc = result.qc
    qc_color = "green" if qc.passed else "red"
    qc_label = "PASS" if qc.passed else "FAIL"
    st.markdown(f"**Quality check:** :{qc_color}[{qc_label}]")

    if qc.warnings:
        with st.expander(f"⚠️  {len(qc.warnings)} warning(s)", expanded=False):
            for w in qc.warnings:
                st.write(f"- {w}")
    if qc.failures:
        with st.expander(f"❌ {len(qc.failures)} failure(s)", expanded=True):
            for f in qc.failures:
                st.write(f"- {f}")
    if qc.metrics:
        with st.expander("QC metrics", expanded=False):
            st.json(qc.metrics)

    # ---- Visual tabs ----
    st.header("Visualizations")
    tab_mesh, tab_seg, tab_meta, tab_raw = st.tabs(
        ["3D mesh", "Segmentation", "Metadata", "Raw JSON"]
    )

    with tab_mesh:
        try:
            _render_mesh_3d(result)
        except Exception as e:
            st.warning(f"3D viewer unavailable: {e}")

    with tab_seg:
        # Best-effort: re-run the parser on the front image so we can show its mask.
        # (We don't keep the parsing object around in PredictionResult — it's not
        # part of the user-facing surface.)
        try:
            from bodymeasure.io.image_io import load_image
            bgr, _ = load_image(front_path) if False else (cv2.cvtColor(np.array(Image.open(io.BytesIO(front_bytes)).convert("RGB")), cv2.COLOR_RGB2BGR), {})
            pipeline._ensure_loaded()
            det = pipeline._detector.detect(bgr)
            if det is not None:
                parsing = pipeline._parser.parse(bgr, det.bbox_xyxy)
                overlay = _segmentation_overlay(front_bytes, parsing.full_mask)
                if overlay is not None:
                    st.image(overlay, caption="Background tinted red", use_container_width=True)
                else:
                    st.info("No mask available from the configured parser.")
            else:
                st.info("No person detected for visualization.")
        except Exception as e:
            st.info(f"Segmentation preview unavailable: {e}")

    with tab_meta:
        st.json(result.metadata)

    with tab_raw:
        payload = {
            "measurements": result.measurements_dict(),
            "qc": {
                "passed": qc.passed,
                "warnings": qc.warnings,
                "failures": qc.failures,
                "metrics": qc.metrics,
            },
            "metadata": result.metadata,
        }
        text = json.dumps(payload, indent=2)
        st.code(text, language="json")
        st.download_button(
            "Download as JSON",
            data=text,
            file_name="bodymeasure_result.json",
            mime="application/json",
        )

elif not run and front_file is None:
    st.info(
        "👈 Upload a front-facing photo to begin. "
        "A side-facing photo is optional but significantly improves accuracy of "
        "circumference measurements (chest, waist, hip)."
    )
