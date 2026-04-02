
# import sys
# import os
# import urllib.request
# from pathlib import Path

# # ensure core import path
# root_dir = Path(__file__).resolve().parent
# if str(root_dir) not in sys.path:
#     sys.path.append(str(root_dir))

# import io
# import cv2
# import numpy as np
# import trimesh
# import streamlit as st
# import plotly.graph_objects as go
# from PIL import Image

# from ultralytics import YOLO
# import torch
# import smplx

# from core.pipeline import (
#     BodyMeasurementPipeline,
#     ISegmentationModel,
#     IPoseModel,
#     ISMPlReconstructor,
# )

# from core.mesh_geometry import (
#     scale_mesh,
#     extract_measurements,
# )

# st.set_page_config(
#     page_title="3D Body Measurement",
#     page_icon="🧍",
#     layout="wide",
# )

# # =============================
# # session state
# # =============================
# if "analyzed" not in st.session_state:
#     st.session_state.analyzed = False

# def reset_analysis_state():
#     st.session_state.analyzed = False

# # =============================
# # Download Utility
# # =============================
# def download_model_file(url, dest_path, desc):
#     """Automatically creates directories and downloads required model weights."""
#     dest_path = Path(dest_path)
#     if not dest_path.parent.exists():
#         dest_path.parent.mkdir(parents=True, exist_ok=True)
        
#     if not dest_path.exists():
#         try:
#             with st.spinner(desc):
#                 urllib.request.urlretrieve(url, str(dest_path))
#         except Exception as e:
#             st.error(f"Failed to download {dest_path.name}: {str(e)}")

# # =============================
# # segmentation model
# # =============================
# class YOLOSegmentationModel(ISegmentationModel):
#     def __init__(self):
#         self.model = YOLO("yolov8n-seg.pt")

#     def segment(self, img):
#         mask = np.zeros(img.shape[:2], dtype=np.uint8)
#         results = self.model(img, verbose=False)

#         if results[0].masks is not None:
#             for i, cls in enumerate(results[0].boxes.cls):
#                 if int(cls) == 0:  # Class 0 is 'person'
#                     contour = results[0].masks.xy[i]
#                     if len(contour) > 0:
#                         contour = np.array(contour, dtype=np.int32)
#                         cv2.fillPoly(mask, [contour], 255)
#                     return mask
#         return mask

# # =============================
# # pose model (OpenCV DNN)
# # =============================
# class OpenCVPoseModel(IPoseModel):
#     def __init__(self):
#         self.model_dir = root_dir / "models" / "openpose"
#         self.proto_file = self.model_dir / "pose_deploy_linevec.prototxt"
#         self.weights_file = self.model_dir / "pose_iter_440000.caffemodel"
        
#         proto_url = "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/coco/pose_deploy_linevec.prototxt"
#         weights_url = "https://huggingface.co/camenduru/openpose/resolve/main/models/pose/coco/pose_iter_440000.caffemodel"

#         download_model_file(proto_url, self.proto_file, "Downloading OpenPose architecture (prototxt)...")
#         download_model_file(weights_url, self.weights_file, "Downloading OpenPose weights (200MB). This may take a moment...")

#         self.net = None
#         try:
#             self.net = cv2.dnn.readNetFromCaffe(str(self.proto_file), str(self.weights_file))
#         except Exception as e:
#             self.load_error = str(e)

#     def estimate_pose(self, img):
#         keypoints = {}
#         if self.net is None:
#             st.warning(f"⚠️ OpenCV Pose Model failed to load: {self.load_error}. Returning dummy keypoints.")
#             return {"joint_0": (img.shape[1]/2, img.shape[0]/2)}

#         frame_height, frame_width = img.shape[:2]
#         in_width = 368
#         in_height = int((in_width / frame_width) * frame_height)
        
#         inp_blob = cv2.dnn.blobFromImage(img, 1.0 / 255, (in_width, in_height),
#                                         (0, 0, 0), swapRB=False, crop=False)
#         self.net.setInput(inp_blob)
#         output = self.net.forward()

#         H = output.shape[2]
#         W = output.shape[3]
        
#         for i in range(18):
#             prob_map = output[0, i, :, :]
#             min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)
#             if prob > 0.1:
#                 x = (frame_width * point[0]) / W
#                 y = (frame_height * point[1]) / H
#                 keypoints[f"joint_{i}"] = (float(x), float(y))
#         return keypoints

# # =============================
# # smpl reconstruction
# # =============================
# class PyTorchSMPLReconstructor(ISMPlReconstructor):
#     def __init__(self):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
#         self.model_path = Path("C:/Projects/body_measurement/models/smpl")
#         # Updated to look for the .pkl file
#         self.model_file = self.model_path / "SMPL_NEUTRAL.pkl"
        
#         if not self.model_path.exists():
#             self.model_path.mkdir(parents=True, exist_ok=True)
#             with open(self.model_path / "README.txt", "w") as f:
#                 f.write("Please place your SMPL neutral .pkl models here.\n"
#                         "Download them from https://smpl.is.tue.mpg.de/ after registering.")

#         self.smpl = None
#         self.load_error = None
        
#         if not self.model_file.exists():
#             folder_contents = [f.name for f in self.model_path.iterdir()]
#             self.load_error = Exception(
#                 f"File 'SMPL_NEUTRAL.pkl' is missing!\n\n"
#                 f"Expected exactly at: {self.model_file}\n"
#                 f"Currently, this folder only contains: {folder_contents}"
#             )
#             return
        
#         try:
#             self.smpl = smplx.create(
#                 str(self.model_file),
#                 model_type="smpl",
#                 gender="neutral",
#                 ext="pkl", # Explicitly set extension to pkl
#             ).to(self.device)
#         except Exception as e:
#             self.load_error = e 

#     def reconstruct(self, img, mask, keypoints):
#         if self.smpl is None:
#             st.error(f"🚨 SMPL configuration error at: `{self.model_path}`")
#             st.warning("⚠️ **RAW ERROR TRACEBACK:**")
#             st.exception(self.load_error) 
#             st.info("**Action Required:**\n"
#                     "1. Go to https://smpl.is.tue.mpg.de/\n"
#                     "2. Download 'SMPL python v.1.1.0' and extract it.\n"
#                     "3. Find 'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl' in the smpl/models/ folder.\n"
#                     "4. Copy it to your local models/smpl folder and rename it to 'SMPL_NEUTRAL.pkl'.")
#             st.stop()
            
#         betas = torch.zeros([1, 10]).to(self.device)
#         body_pose = torch.zeros([1, 69]).to(self.device)
#         global_orient = torch.zeros([1, 3]).to(self.device)

#         output = self.smpl(betas=betas, body_pose=body_pose, global_orient=global_orient)
#         vertices = output.vertices.detach().cpu().numpy().squeeze()
#         mesh = trimesh.Trimesh(vertices=vertices, faces=self.smpl.faces)

#         min_y = mesh.bounds[0][1]
#         mesh.apply_translation([0, -min_y, 0])
#         return mesh

# # =============================
# # pipeline loader
# # =============================
# @st.cache_resource
# def load_pipeline():
#     return BodyMeasurementPipeline(
#         seg_model=YOLOSegmentationModel(),
#         pose_model=OpenCVPoseModel(),
#         smpl_model=PyTorchSMPLReconstructor(),
#     )

# pipeline = load_pipeline()

# # =============================
# # visualization
# # =============================
# def render_mesh(mesh):
#     v = mesh.vertices
#     f = mesh.faces
#     fig = go.Figure(data=[go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2], opacity=1.0)])
#     fig.update_layout(height=600, scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='data'))
#     return fig

# def render_wireframe(mesh):
#     vertices = mesh.vertices
#     edges = mesh.edges_unique
#     xs, ys, zs = [], [], []

#     for e in edges:
#         xs += [vertices[e[0], 0], vertices[e[1], 0], None]
#         ys += [vertices[e[0], 1], vertices[e[1], 1], None]
#         zs += [vertices[e[0], 2], vertices[e[1], 2], None]

#     fig = go.Figure(data=[go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color='black', width=1))])
#     fig.update_layout(height=600, scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='data'))
#     return fig

# def overlay_mask(img, mask):
#     overlay = img.copy()
#     overlay[mask == 255] = [0, 255, 0]
#     return cv2.addWeighted(overlay, 0.4, img, 0.6, 0)

# # =============================
# # UI
# # =============================
# st.title("3D Body Measurement from Image")

# with st.sidebar:
#     uploaded_file = st.file_uploader("Upload front image", type=["jpg", "png", "jpeg"], on_change=reset_analysis_state)
#     height_cm = st.number_input("Height (cm)", min_value=120.0, max_value=220.0, value=170.0)
#     if st.button("Run"):
#         st.session_state.analyzed = True

# if uploaded_file and st.session_state.analyzed:
#     img = Image.open(uploaded_file).convert("RGB")
#     img_np = np.array(img)
#     img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

#     with st.spinner("Analyzing human pose & geometry..."):
#         mask = pipeline.seg_model.segment(img_bgr)
#         keypoints = pipeline.pose_model.estimate_pose(img_bgr)
#         mesh = pipeline.smpl_model.reconstruct(img_bgr, mask, keypoints)
#         mesh = scale_mesh(mesh, height_cm)
#         measurements = extract_measurements(mesh)

#     tab1, tab2, tab3, tab4 = st.tabs(["Segmentation", "3D Mesh", "Wireframe", "Measurements"])
#     with tab1: st.image(overlay_mask(img_np, mask), use_container_width=True)
#     with tab2: st.plotly_chart(render_mesh(mesh), use_container_width=True)
#     with tab3: st.plotly_chart(render_wireframe(mesh), use_container_width=True)
#     with tab4: st.json(measurements)

import sys
import os
import urllib.request
from pathlib import Path

# ensure core import path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import io
import cv2
import numpy as np
import trimesh
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

from ultralytics import YOLO
import torch
import smplx

from core.pipeline import (
    BodyMeasurementPipeline,
    ISegmentationModel,
    IPoseModel,
    ISMPlReconstructor,
)

from core.mesh_geometry import (
    scale_mesh,
    extract_measurements,
)

st.set_page_config(
    page_title="3D Body Measurement",
    page_icon="🧍",
    layout="wide",
)

# =============================
# session state
# =============================
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False

def reset_analysis_state():
    st.session_state.analyzed = False

# =============================
# Download Utility
# =============================
def download_model_file(url, dest_path, desc):
    """Automatically creates directories and downloads required model weights."""
    dest_path = Path(dest_path)
    if not dest_path.parent.exists():
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
    if not dest_path.exists():
        try:
            with st.spinner(desc):
                urllib.request.urlretrieve(url, str(dest_path))
        except Exception as e:
            st.error(f"Failed to download {dest_path.name}: {str(e)}")

# =============================
# segmentation model
# =============================
class YOLOSegmentationModel(ISegmentationModel):
    def __init__(self):
        self.model = YOLO("yolov8n-seg.pt")

    def segment(self, img):
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        results = self.model(img, verbose=False)

        if results[0].masks is not None:
            for i, cls in enumerate(results[0].boxes.cls):
                if int(cls) == 0:  # Class 0 is 'person'
                    contour = results[0].masks.xy[i]
                    if len(contour) > 0:
                        contour = np.array(contour, dtype=np.int32)
                        cv2.fillPoly(mask, [contour], 255)
                    return mask
        return mask

# =============================
# pose model (OpenCV DNN)
# =============================
class OpenCVPoseModel(IPoseModel):
    def __init__(self):
        self.model_dir = root_dir / "models" / "openpose"
        self.proto_file = self.model_dir / "pose_deploy_linevec.prototxt"
        self.weights_file = self.model_dir / "pose_iter_440000.caffemodel"
        
        proto_url = "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/coco/pose_deploy_linevec.prototxt"
        weights_url = "https://huggingface.co/camenduru/openpose/resolve/main/models/pose/coco/pose_iter_440000.caffemodel"

        download_model_file(proto_url, self.proto_file, "Downloading OpenPose architecture (prototxt)...")
        download_model_file(weights_url, self.weights_file, "Downloading OpenPose weights (200MB). This may take a moment...")

        self.net = None
        try:
            self.net = cv2.dnn.readNetFromCaffe(str(self.proto_file), str(self.weights_file))
        except Exception as e:
            self.load_error = str(e)

    def estimate_pose(self, img):
        keypoints = {}
        if self.net is None:
            st.warning(f"⚠️ OpenCV Pose Model failed to load: {self.load_error}. Returning dummy keypoints.")
            return {"joint_0": (img.shape[1]/2, img.shape[0]/2)}

        frame_height, frame_width = img.shape[:2]
        in_width = 368
        in_height = int((in_width / frame_width) * frame_height)
        
        inp_blob = cv2.dnn.blobFromImage(img, 1.0 / 255, (in_width, in_height),
                                        (0, 0, 0), swapRB=False, crop=False)
        self.net.setInput(inp_blob)
        output = self.net.forward()

        H = output.shape[2]
        W = output.shape[3]
        
        for i in range(18):
            prob_map = output[0, i, :, :]
            min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)
            if prob > 0.1:
                x = (frame_width * point[0]) / W
                y = (frame_height * point[1]) / H
                keypoints[f"joint_{i}"] = (float(x), float(y))
        return keypoints

# =============================
# smpl reconstruction
# =============================
class PyTorchSMPLReconstructor(ISMPlReconstructor):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model_path = Path("C:/Projects/body_measurement/models/smpl")
        self.model_file = self.model_path / "SMPL_NEUTRAL.pkl"
        
        if not self.model_path.exists():
            self.model_path.mkdir(parents=True, exist_ok=True)
            with open(self.model_path / "README.txt", "w") as f:
                f.write("Please place your SMPL neutral .pkl models here.\n"
                        "Download them from https://smpl.is.tue.mpg.de/ after registering.")

        self.smpl = None
        self.load_error = None
        
        if not self.model_file.exists():
            folder_contents = [f.name for f in self.model_path.iterdir()]
            self.load_error = Exception(
                f"File 'SMPL_NEUTRAL.pkl' is missing!\n\n"
                f"Expected exactly at: {self.model_file}\n"
                f"Currently, this folder only contains: {folder_contents}"
            )
            return
        
        try:
            self.smpl = smplx.create(
                str(self.model_file),
                model_type="smpl",
                gender="neutral",
                ext="pkl",
            ).to(self.device)
        except Exception as e:
            self.load_error = e 

    def reconstruct(self, img, mask, keypoints):
        if self.smpl is None:
            st.error(f"🚨 SMPL configuration error at: `{self.model_path}`")
            st.warning("⚠️ **RAW ERROR TRACEBACK:**")
            st.exception(self.load_error) 
            st.info("**Action Required:**\n"
                    "1. Go to https://smpl.is.tue.mpg.de/\n"
                    "2. Download 'SMPL python v.1.1.0' and extract it.\n"
                    "3. Find 'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl' in the smpl/models/ folder.\n"
                    "4. Copy it to your local models/smpl folder and rename it to 'SMPL_NEUTRAL.pkl'.")
            st.stop()
            
        betas = torch.zeros([1, 10]).to(self.device)
        body_pose = torch.zeros([1, 69]).to(self.device)
        global_orient = torch.zeros([1, 3]).to(self.device)

        output = self.smpl(betas=betas, body_pose=body_pose, global_orient=global_orient)
        vertices = output.vertices.detach().cpu().numpy().squeeze()
        mesh = trimesh.Trimesh(vertices=vertices, faces=self.smpl.faces)

        min_y = mesh.bounds[0][1]
        mesh.apply_translation([0, -min_y, 0])
        return mesh

# =============================
# pipeline loader
# =============================
@st.cache_resource
def load_pipeline():
    return BodyMeasurementPipeline(
        seg_model=YOLOSegmentationModel(),
        pose_model=OpenCVPoseModel(),
        smpl_model=PyTorchSMPLReconstructor(),
    )

pipeline = load_pipeline()

# =============================
# visualization
# =============================
def render_mesh(mesh):
    v = mesh.vertices
    f = mesh.faces
    fig = go.Figure(data=[go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2], opacity=1.0)])
    fig.update_layout(height=600, scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='data'))
    return fig

def render_wireframe(mesh):
    vertices = mesh.vertices
    edges = mesh.edges_unique
    xs, ys, zs = [], [], []

    for e in edges:
        xs += [vertices[e[0], 0], vertices[e[1], 0], None]
        ys += [vertices[e[0], 1], vertices[e[1], 1], None]
        zs += [vertices[e[0], 2], vertices[e[1], 2], None]

    fig = go.Figure(data=[go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color='black', width=1))])
    fig.update_layout(
        height=600, 
        paper_bgcolor='white', 
        plot_bgcolor='white',
        margin=dict(l=0, r=0, b=0, t=0), 
        scene=dict(
            bgcolor='white', 
            xaxis=dict(visible=False, showbackground=False), 
            yaxis=dict(visible=False, showbackground=False), 
            zaxis=dict(visible=False, showbackground=False), 
            aspectmode='data'
        )
    )
    return fig

def overlay_mask(img, mask):
    overlay = img.copy()
    overlay[mask == 255] = [0, 255, 0]
    return cv2.addWeighted(overlay, 0.4, img, 0.6, 0)

# =============================
# UI
# =============================
st.title("3D Body Measurement from Image")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload front image", type=["jpg", "png", "jpeg"], on_change=reset_analysis_state)
    height_cm = st.number_input("Height (cm)", min_value=120.0, max_value=220.0, value=170.0)
    if st.button("Run"):
        st.session_state.analyzed = True

if uploaded_file and st.session_state.analyzed:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    with st.spinner("Analyzing human pose & geometry..."):
        mask = pipeline.seg_model.segment(img_bgr)
        keypoints = pipeline.pose_model.estimate_pose(img_bgr)
        mesh = pipeline.smpl_model.reconstruct(img_bgr, mask, keypoints)
        mesh = scale_mesh(mesh, height_cm)
        measurements = extract_measurements(mesh)

    tab1, tab2, tab3, tab4 = st.tabs(["Segmentation", "3D Mesh", "Wireframe", "Measurements"])
    
    with tab1: 
        # Using columns to constrain the image size and center it
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(overlay_mask(img_np, mask), use_container_width=True)
            
    with tab2: st.plotly_chart(render_mesh(mesh), use_container_width=True)
    with tab3: st.plotly_chart(render_wireframe(mesh), use_container_width=True)
    with tab4: st.json(measurements)