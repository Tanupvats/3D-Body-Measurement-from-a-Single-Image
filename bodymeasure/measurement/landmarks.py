

from __future__ import annotations

# ---------- Body landmarks (SMPL-X 10475 vertex mesh) ----------
# Pelvis / hips
SMPLX_HIP_LEFT = 6571
SMPLX_HIP_RIGHT = 3327
SMPLX_HIP_FRONT = 3119      # belly-button-ish front pelvis
SMPLX_HIP_BACK = 5614

# Waist (narrow point of the torso)
SMPLX_WAIST_FRONT = 5333    # natural waist front
SMPLX_WAIST_BACK = 5346

# Chest / bust
SMPLX_CHEST_FRONT = 3076    # nipple line approx
SMPLX_CHEST_BACK = 3017

# Shoulders
SMPLX_SHOULDER_LEFT = 4481
SMPLX_SHOULDER_RIGHT = 7144
SMPLX_NECK_FRONT = 3458
SMPLX_NECK_BACK = 1305

# Limbs — used for LINEAR (not girth) measurements
SMPLX_HEAD_TOP = 8980
SMPLX_HEEL_LEFT = 8852
SMPLX_HEEL_RIGHT = 5601
SMPLX_TOE_LEFT = 5778
SMPLX_TOE_RIGHT = 8463

SMPLX_WRIST_LEFT = 4824
SMPLX_WRIST_RIGHT = 7547

SMPLX_KNEE_LEFT = 4756
SMPLX_KNEE_RIGHT = 7470
SMPLX_ANKLE_LEFT = 5882
SMPLX_ANKLE_RIGHT = 8576

# Crotch (for inseam start)
SMPLX_CROTCH = 5500

# Bicep (for upper-arm girth)
SMPLX_BICEP_LEFT = 4220
SMPLX_BICEP_RIGHT = 6904

# Thigh — at gluteal fold
SMPLX_THIGH_LEFT = 4115
SMPLX_THIGH_RIGHT = 6816

# Calf max
SMPLX_CALF_LEFT = 4505
SMPLX_CALF_RIGHT = 7204


# Pairs that define a slicing plane (two vertices that define plane height
# and the body axis). For each girth measurement we pick a pair and the
# plane normal is the body axis (Y by default in canonical pose).
GIRTH_LANDMARKS = {
    # name -> (vertex_id_a, vertex_id_b, axis_letter)
    "chest":     (SMPLX_CHEST_FRONT, SMPLX_CHEST_BACK, "y"),
    "waist":     (SMPLX_WAIST_FRONT, SMPLX_WAIST_BACK, "y"),
    "hip":       (SMPLX_HIP_FRONT,   SMPLX_HIP_BACK,   "y"),
    "neck":      (SMPLX_NECK_FRONT,  SMPLX_NECK_BACK,  "y"),
    # Limb girths slice perpendicular to limb axis; landmark pairs define
    # the height of the slice but the limb axis is approximated via joints.
    "thigh_left":   (SMPLX_THIGH_LEFT, SMPLX_THIGH_LEFT, "limb_left_thigh"),
    "thigh_right":  (SMPLX_THIGH_RIGHT, SMPLX_THIGH_RIGHT, "limb_right_thigh"),
    "calf_left":    (SMPLX_CALF_LEFT, SMPLX_CALF_LEFT, "limb_left_calf"),
    "calf_right":   (SMPLX_CALF_RIGHT, SMPLX_CALF_RIGHT, "limb_right_calf"),
    "bicep_left":   (SMPLX_BICEP_LEFT, SMPLX_BICEP_LEFT, "limb_left_upper_arm"),
    "bicep_right":  (SMPLX_BICEP_RIGHT, SMPLX_BICEP_RIGHT, "limb_right_upper_arm"),
}


# Linear measurements: (name, start_vertex, end_vertex, geodesic?)
LINEAR_LANDMARKS = {
    # Stature: head-top to floor (we use heel as a proxy for floor in
    # canonical pose).
    "stature":            (SMPLX_HEAD_TOP, SMPLX_HEEL_LEFT, False),
    "shoulder_breadth":   (SMPLX_SHOULDER_LEFT, SMPLX_SHOULDER_RIGHT, False),
    "arm_length_left":    ("joint:left_shoulder", "joint:left_wrist", False),
    "arm_length_right":   ("joint:right_shoulder", "joint:right_wrist", False),
    "inseam_left":        (SMPLX_CROTCH, SMPLX_ANKLE_LEFT, False),
    "inseam_right":       (SMPLX_CROTCH, SMPLX_ANKLE_RIGHT, False),
    "outseam_left":       (SMPLX_HIP_LEFT, SMPLX_ANKLE_LEFT, False),
    "outseam_right":      (SMPLX_HIP_RIGHT, SMPLX_ANKLE_RIGHT, False),
}


# Joint indices in the SMPL-X joint output (when the regressor produces
# the standard 55-joint SMPL-X joint layout).
SMPLX_JOINTS = {
    "pelvis": 0,
    "left_hip": 1, "right_hip": 2,
    "spine1": 3,
    "left_knee": 4, "right_knee": 5,
    "spine2": 6,
    "left_ankle": 7, "right_ankle": 8,
    "spine3": 9,
    "left_foot": 10, "right_foot": 11,
    "neck": 12,
    "left_collar": 13, "right_collar": 14,
    "head": 15,
    "left_shoulder": 16, "right_shoulder": 17,
    "left_elbow": 18, "right_elbow": 19,
    "left_wrist": 20, "right_wrist": 21,
}
