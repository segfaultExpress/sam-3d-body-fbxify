from sam_3d_body.metadata.mhr70 import pose_info

PROFILES = {
    "mhr": {
        "lod0_path": "mapping/mhr/lod0.fbx",
        "lod1_path": "mapping/mhr/lod1.fbx",
        "lod2_path": "mapping/mhr/lod2.fbx",
        "lod3_path": "mapping/mhr/lod3.fbx",
        "lod4_path": "mapping/mhr/lod4.fbx",
        "lod5_path": "mapping/mhr/lod5.fbx",
        "lod6_path": "mapping/mhr/lod6.fbx",
        "joint_mapping_path": "mapping/mhr/mhr_armature_joints_mapping.json",
        "armature_rest_pose_path": "mapping/mhr/mhr_armature_skeleton_rest_pose.json",
    },
    "mixamo": {
        "joint_mapping_path": "mapping/mixamo/mixamo_armature_joints_mapping.json",
        "armature_rest_pose_path": "mapping/mixamo/mixamo_armature_skeleton_rest_pose.json",
    },
    "unity": {
        "joint_mapping_path": "mapping/unity/unity_armature_joints_mapping.json",
        "armature_rest_pose_path": "mapping/unity/unity_armature_skeleton_rest_pose.json",
    },
}

JOINT_NAMES = [
    'body_world', 'root', 'l_upleg', 'l_lowleg', 'l_foot', 'l_talocrural', 'l_subtalar', 
    'l_transversetarsal', 'l_ball', 'l_lowleg_twist1_proc', 'l_lowleg_twist2_proc', 'l_lowleg_twist3_proc', 
    'l_lowleg_twist4_proc', 'l_upleg_twist0_proc', 'l_upleg_twist1_proc', 'l_upleg_twist2_proc', 'l_upleg_twist3_proc', 
    'l_upleg_twist4_proc', 'r_upleg', 'r_lowleg', 'r_foot', 'r_talocrural', 'r_subtalar', 'r_transversetarsal', 'r_ball', 
    'r_lowleg_twist1_proc', 'r_lowleg_twist2_proc', 'r_lowleg_twist3_proc', 'r_lowleg_twist4_proc', 'r_upleg_twist0_proc', 
    'r_upleg_twist1_proc', 'r_upleg_twist2_proc', 'r_upleg_twist3_proc', 'r_upleg_twist4_proc', 'c_spine0', 'c_spine1', 
    'c_spine2', 'c_spine3', 'r_clavicle', 'r_uparm', 'r_lowarm', 'r_wrist_twist', 'r_wrist', 'r_pinky0', 'r_pinky1', 'r_pinky2', 
    'r_pinky3', 'r_pinky_null', 'r_ring1', 'r_ring2', 'r_ring3', 'r_ring_null', 'r_middle1', 'r_middle2', 'r_middle3', 
    'r_middle_null', 'r_index1', 'r_index2', 'r_index3', 'r_index_null', 'r_thumb0', 'r_thumb1', 'r_thumb2', 'r_thumb3', 
    'r_thumb_null', 'r_lowarm_twist1_proc', 'r_lowarm_twist2_proc', 'r_lowarm_twist3_proc', 'r_lowarm_twist4_proc', 
    'r_uparm_twist0_proc', 'r_uparm_twist1_proc', 'r_uparm_twist2_proc', 'r_uparm_twist3_proc', 'r_uparm_twist4_proc', 
    'l_clavicle', 'l_uparm', 'l_lowarm', 'l_wrist_twist', 'l_wrist', 'l_pinky0', 'l_pinky1', 'l_pinky2', 'l_pinky3', 
    'l_pinky_null', 'l_ring1', 'l_ring2', 'l_ring3', 'l_ring_null', 'l_middle1', 'l_middle2', 'l_middle3', 'l_middle_null', 
    'l_index1', 'l_index2', 'l_index3', 'l_index_null', 'l_thumb0', 'l_thumb1', 'l_thumb2', 'l_thumb3', 'l_thumb_null', 
    'l_lowarm_twist1_proc', 'l_lowarm_twist2_proc', 'l_lowarm_twist3_proc', 'l_lowarm_twist4_proc', 'l_uparm_twist0_proc', 
    'l_uparm_twist1_proc', 'l_uparm_twist2_proc', 'l_uparm_twist3_proc', 'l_uparm_twist4_proc', 'c_neck', 'c_neck_twist1_proc', 
    'c_neck_twist0_proc', 'c_head', 'c_jaw', 'c_teeth', 'c_jaw_null', 'c_tongue0', 'c_tongue1', 'c_tongue2', 'c_tongue3', 
    'c_tongue4', 'r_eye', 'r_eye_null', 'l_eye', 'l_eye_null', 'c_head_null'
]

JOINT_NAMES_TO_INDEX = {name: index for index, name in enumerate(JOINT_NAMES)}

MHR_KEYPOINT_INDEX = {info["name"]: idx for idx, info in pose_info["keypoint_info"].items()}
