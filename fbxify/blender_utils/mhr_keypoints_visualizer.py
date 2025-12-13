"""
MHR Keypoints Visualizer for Blender
Creates a 3D skeleton visualization with proper colors and connections
Based on MHR70 (70 keypoints) dataset structure from SAM 3D Body
"""

import bpy
import bmesh
import mathutils
import math
from mathutils import Vector
import json
import sys
import os

# Add the project root to the path to import sam_3d_body metadata
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from sam_3d_body.metadata.mhr70 import pose_info
except ImportError:
    # Fallback: try to load from mhr_info.txt if import fails
    mhr_info_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mhr_info.txt")
    if os.path.exists(mhr_info_path):
        with open(mhr_info_path, 'r') as f:
            pose_info = eval(f.read())
    else:
        raise ImportError("Could not import pose_info from sam_3d_body.metadata.mhr70 or load from mhr_info.txt")


class MHRKeypointsVisualizer:
    """Class for managing MHR skeleton visualization in Blender"""
    
    def __init__(self):
        # Scale factors for keypoint positions
        self.x_scale = 1
        self.y_scale = 1
        self.z_scale = 1
        self.keypoint_scale = [0.4, 0.4, 0.4]
        self.line_scale_xy = [0.5, 0.5]
        self.keypoint_objects = {}  # Dict to track keypoint objects by name
        self.line_objects = {}      # Dict to track line objects by name
        self.parent_axis = None     # Parent object for all skeleton components
        
        # Load MHR keypoint and skeleton info from pose_info
        self.MHR_KEYPOINTS = {}
        self.MHR_SKELETON = []
        self.T_POSE_POSITIONS = {}
        
        # Build keypoint dictionary from pose_info
        for kp_id, kp_data in pose_info['keypoint_info'].items():
            self.MHR_KEYPOINTS[kp_id] = {
                "name": kp_data['name'],
                "color": kp_data['color']
            }
        
        # Build skeleton connections from pose_info
        for skel_id, skel_data in pose_info['skeleton_info'].items():
            link = skel_data['link']
            # Find keypoint IDs from names
            start_id = None
            end_id = None
            for kp_id, kp_data in pose_info['keypoint_info'].items():
                if kp_data['name'] == link[0]:
                    start_id = kp_id
                if kp_data['name'] == link[1]:
                    end_id = kp_id
            if start_id is not None and end_id is not None:
                self.MHR_SKELETON.append((start_id, end_id))
        
        # Create default T-pose positions (simplified, can be improved)
        self._create_default_t_pose_positions()

    def _create_default_t_pose_positions(self):
        """Create default T-pose positions for all keypoints"""
        # Body keypoints (0-16)
        self.T_POSE_POSITIONS[0] = (0, 0, 1.7)      # nose
        self.T_POSE_POSITIONS[1] = (-0.05, 0, 1.72) # left_eye
        self.T_POSE_POSITIONS[2] = (0.05, 0, 1.72)  # right_eye
        self.T_POSE_POSITIONS[3] = (-0.08, 0, 1.7)  # left_ear
        self.T_POSE_POSITIONS[4] = (0.08, 0, 1.7)   # right_ear
        self.T_POSE_POSITIONS[5] = (-0.3, 0, 1.5)   # left_shoulder
        self.T_POSE_POSITIONS[6] = (0.3, 0, 1.5)    # right_shoulder
        self.T_POSE_POSITIONS[7] = (-0.5, 0, 1.4)   # left_elbow
        self.T_POSE_POSITIONS[8] = (0.5, 0, 1.4)    # right_elbow
        self.T_POSE_POSITIONS[9] = (-0.1, 0, 1.0)   # left_hip
        self.T_POSE_POSITIONS[10] = (0.1, 0, 1.0)   # right_hip
        self.T_POSE_POSITIONS[11] = (-0.1, 0, 0.5)  # left_knee
        self.T_POSE_POSITIONS[12] = (0.1, 0, 0.5)   # right_knee
        self.T_POSE_POSITIONS[13] = (-0.1, 0, 0.0) # left_ankle
        self.T_POSE_POSITIONS[14] = (0.1, 0, 0.0)   # right_ankle
        
        # Feet (15-20)
        self.T_POSE_POSITIONS[15] = (-0.15, 0, 0.0) # left_big_toe
        self.T_POSE_POSITIONS[16] = (-0.05, 0, 0.0) # left_small_toe
        self.T_POSE_POSITIONS[17] = (-0.1, -0.1, 0.0) # left_heel
        self.T_POSE_POSITIONS[18] = (0.15, 0, 0.0)  # right_big_toe
        self.T_POSE_POSITIONS[19] = (0.05, 0, 0.0)  # right_small_toe
        self.T_POSE_POSITIONS[20] = (0.1, -0.1, 0.0) # right_heel
        
        # Right hand (21-41)
        self.T_POSE_POSITIONS[41] = (0.7, 0, 1.3)   # right_wrist
        # Right thumb
        self.T_POSE_POSITIONS[24] = (0.75, 0.05, 1.32) # right_thumb_third_joint
        self.T_POSE_POSITIONS[23] = (0.8, 0.1, 1.34)  # right_thumb2
        self.T_POSE_POSITIONS[22] = (0.85, 0.15, 1.36) # right_thumb3
        self.T_POSE_POSITIONS[21] = (0.9, 0.2, 1.38)  # right_thumb4
        # Right forefinger
        self.T_POSE_POSITIONS[28] = (0.75, -0.05, 1.32) # right_forefinger_third_joint
        self.T_POSE_POSITIONS[27] = (0.8, -0.1, 1.34)  # right_forefinger2
        self.T_POSE_POSITIONS[26] = (0.85, -0.15, 1.36) # right_forefinger3
        self.T_POSE_POSITIONS[25] = (0.9, -0.2, 1.38)  # right_forefinger4
        # Right middle finger
        self.T_POSE_POSITIONS[32] = (0.75, -0.1, 1.3)  # right_middle_finger_third_joint
        self.T_POSE_POSITIONS[31] = (0.8, -0.15, 1.28) # right_middle_finger2
        self.T_POSE_POSITIONS[30] = (0.85, -0.2, 1.26) # right_middle_finger3
        self.T_POSE_POSITIONS[29] = (0.9, -0.25, 1.24) # right_middle_finger4
        # Right ring finger
        self.T_POSE_POSITIONS[36] = (0.75, -0.15, 1.28) # right_ring_finger_third_joint
        self.T_POSE_POSITIONS[35] = (0.8, -0.2, 1.26)  # right_ring_finger2
        self.T_POSE_POSITIONS[34] = (0.85, -0.25, 1.24) # right_ring_finger3
        self.T_POSE_POSITIONS[33] = (0.9, -0.3, 1.22)  # right_ring_finger4
        # Right pinky
        self.T_POSE_POSITIONS[40] = (0.75, -0.2, 1.26)  # right_pinky_finger_third_joint
        self.T_POSE_POSITIONS[39] = (0.8, -0.25, 1.24)  # right_pinky_finger2
        self.T_POSE_POSITIONS[38] = (0.85, -0.3, 1.22) # right_pinky_finger3
        self.T_POSE_POSITIONS[37] = (0.9, -0.35, 1.2)  # right_pinky_finger4
        
        # Left hand (42-62)
        self.T_POSE_POSITIONS[62] = (-0.7, 0, 1.3)  # left_wrist
        # Left thumb
        self.T_POSE_POSITIONS[45] = (-0.75, -0.05, 1.32) # left_thumb_third_joint
        self.T_POSE_POSITIONS[44] = (-0.8, -0.1, 1.34)  # left_thumb2
        self.T_POSE_POSITIONS[43] = (-0.85, -0.15, 1.36) # left_thumb3
        self.T_POSE_POSITIONS[42] = (-0.9, -0.2, 1.38)  # left_thumb4
        # Left forefinger
        self.T_POSE_POSITIONS[49] = (-0.75, 0.05, 1.32) # left_forefinger_third_joint
        self.T_POSE_POSITIONS[48] = (-0.8, 0.1, 1.34)   # left_forefinger2
        self.T_POSE_POSITIONS[47] = (-0.85, 0.15, 1.36) # left_forefinger3
        self.T_POSE_POSITIONS[46] = (-0.9, 0.2, 1.38)   # left_forefinger4
        # Left middle finger
        self.T_POSE_POSITIONS[53] = (-0.75, 0.1, 1.3)   # left_middle_finger_third_joint
        self.T_POSE_POSITIONS[52] = (-0.8, 0.15, 1.28)  # left_middle_finger2
        self.T_POSE_POSITIONS[51] = (-0.85, 0.2, 1.26)  # left_middle_finger3
        self.T_POSE_POSITIONS[50] = (-0.9, 0.25, 1.24)  # left_middle_finger4
        # Left ring finger
        self.T_POSE_POSITIONS[57] = (-0.75, 0.15, 1.28) # left_ring_finger_third_joint
        self.T_POSE_POSITIONS[56] = (-0.8, 0.2, 1.26)  # left_ring_finger2
        self.T_POSE_POSITIONS[55] = (-0.85, 0.25, 1.24) # left_ring_finger3
        self.T_POSE_POSITIONS[54] = (-0.9, 0.3, 1.22)   # left_ring_finger4
        # Left pinky
        self.T_POSE_POSITIONS[61] = (-0.75, 0.2, 1.26)  # left_pinky_finger_third_joint
        self.T_POSE_POSITIONS[60] = (-0.8, 0.25, 1.24)  # left_pinky_finger2
        self.T_POSE_POSITIONS[59] = (-0.85, 0.3, 1.22) # left_pinky_finger3
        self.T_POSE_POSITIONS[58] = (-0.9, 0.35, 1.2)   # left_pinky_finger4
        
        # Extra keypoints (63-69)
        self.T_POSE_POSITIONS[63] = (-0.45, 0, 1.45)  # left_olecranon
        self.T_POSE_POSITIONS[64] = (0.45, 0, 1.45)   # right_olecranon
        self.T_POSE_POSITIONS[65] = (-0.4, 0, 1.35)   # left_cubital_fossa
        self.T_POSE_POSITIONS[66] = (0.4, 0, 1.35)    # right_cubital_fossa
        self.T_POSE_POSITIONS[67] = (-0.25, 0, 1.55)  # left_acromion
        self.T_POSE_POSITIONS[68] = (0.25, 0, 1.55)  # right_acromion
        self.T_POSE_POSITIONS[69] = (0, 0, 1.6)       # neck

    def get_material_count(self):
        """Get the current number of materials in the scene"""
        return len(bpy.data.materials)

    def _create_keypoint_sphere(self, position, color, name, radius=0.02):
        """Create a colored sphere for a keypoint"""
        bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=position)
        sphere = bpy.context.active_object
        sphere.name = f"Keypoint_{name}"
        
        # Check if material already exists
        mat_name = f"Mat_{name}"
        if mat_name in bpy.data.materials:
            mat = bpy.data.materials[mat_name]
        else:
            # Create material with the specified color
            mat = bpy.data.materials.new(name=mat_name)
            mat.use_nodes = True
            
            # Get the Principled BSDF node
            principled = mat.node_tree.nodes["Principled BSDF"]
            
            # Set the base color
            principled.inputs[0].default_value = (
                color[0]/255, color[1]/255, color[2]/255, 1.0
            )
            
            # Make it emissive for better visibility
            try:
                if "Emission Color" in principled.inputs:
                    principled.inputs["Emission Color"].default_value = (
                        color[0]/255, color[1]/255, color[2]/255, 1.0
                    )
                elif "Emission" in principled.inputs:
                    principled.inputs["Emission"].default_value = (
                        color[0]/255, color[1]/255, color[2]/255, 1.0
                    )
                
                if "Emission Strength" in principled.inputs:
                    principled.inputs["Emission Strength"].default_value = 0.5
                elif "Emission Weight" in principled.inputs:
                    principled.inputs["Emission Weight"].default_value = 0.5
            except (KeyError, AttributeError):
                try:
                    if len(principled.inputs) > 19:
                        principled.inputs[19].default_value = (
                            color[0]/255, color[1]/255, color[2]/255, 1.0
                        )
                    if len(principled.inputs) > 20:
                        principled.inputs[20].default_value = 0.5
                except (TypeError, AttributeError):
                    pass
        
        # Assign material to the sphere
        sphere.data.materials.append(mat)
        
        return sphere

    def _draw_line(self, sphere1_pos, sphere2_pos, color, name, radius=0.01):
        """Create a cylinder that connects two spheres like a line"""
        # Convert positions to vectors
        pos1 = Vector(sphere1_pos)
        pos2 = Vector(sphere2_pos)
        
        # Calculate the vector from sphere1 to sphere2
        direction = pos2 - pos1
        distance = direction.length
                
        # Create cylinder at origin
        bpy.ops.mesh.primitive_cylinder_add(
            radius=radius,
            depth=1.0,
            location=(0, 0, 0)
        )
        cylinder = bpy.context.active_object
        cylinder.name = f"Line_{name}"
        
        # Position cylinder at midpoint
        midpoint = (pos1 + pos2) / 2
        cylinder.location = midpoint
        
        # Rotate cylinder to align with direction
        if distance > 0:
            direction.normalize()
            z_axis = Vector((0, 0, 1))
                        
            cylinder.rotation_mode = 'XYZ'
            
            if abs(direction.dot(z_axis)) > 0.999:
                if direction.dot(z_axis) < 0:
                    cylinder.rotation_euler = (math.pi, 0, 0)
            else:
                rotation_axis = z_axis.cross(direction)
                rotation_axis.normalize()
                angle = z_axis.angle(direction)
                
                from mathutils import Matrix
                rotation_matrix = Matrix.Rotation(angle, 4, rotation_axis)
                translation_matrix = Matrix.Translation(midpoint)
                cylinder.matrix_world = translation_matrix @ rotation_matrix
        
        # Set scale AFTER matrix transformation
        cylinder.scale = (self.line_scale_xy[0], self.line_scale_xy[1], distance)
        
        # Force update the object
        bpy.context.view_layer.update()
        
        # Check if material already exists
        mat_name = f"LineMat_{name}"
        if mat_name in bpy.data.materials:
            mat = bpy.data.materials[mat_name]
        else:
            # Create material
            mat = bpy.data.materials.new(name=mat_name)
            mat.use_nodes = True
            
            principled = mat.node_tree.nodes["Principled BSDF"]
            principled.inputs[0].default_value = (
                color[0]/255, color[1]/255, color[2]/255, 1.0
            )
            
            try:
                if "Emission Color" in principled.inputs:
                    principled.inputs["Emission Color"].default_value = (
                        color[0]/255, color[1]/255, color[2]/255, 1.0
                    )
                elif "Emission" in principled.inputs:
                    principled.inputs["Emission"].default_value = (
                        color[0]/255, color[1]/255, color[2]/255, 1.0
                    )
                
                if "Emission Strength" in principled.inputs:
                    principled.inputs["Emission Strength"].default_value = 0.3
                elif "Emission Weight" in principled.inputs:
                    principled.inputs["Emission Weight"].default_value = 0.3
            except (KeyError, AttributeError):
                try:
                    if len(principled.inputs) > 19:
                        principled.inputs[19].default_value = (
                            color[0]/255, color[1]/255, color[2]/255, 1.0
                        )
                    if len(principled.inputs) > 20:
                        principled.inputs[20].default_value = 0.3
                except (TypeError, AttributeError):
                    pass
        
        # Assign material to the cylinder
        cylinder.data.materials.append(mat)
        
        return cylinder

    def create_skeleton(self):
        """Create the complete MHR skeleton in Blender"""
                
        # Create parent axis at origin
        bpy.ops.object.empty_add(type='ARROWS', location=(0, 0, 0))
        self.parent_axis = bpy.context.active_object
        self.parent_axis.name = "MHR_Skeleton"
        self.parent_axis.scale = (1, 1, 1)
        
        # Create keypoints
        for kp_id, kp_data in self.MHR_KEYPOINTS.items():
            if kp_id in self.T_POSE_POSITIONS:
                position = self.T_POSE_POSITIONS[kp_id]
                color = kp_data["color"]
                name = kp_data["name"]
                
                # Create sphere for keypoint
                sphere = self._create_keypoint_sphere(position, color, name)
                self.keypoint_objects[name] = sphere
                
                # Make sphere a child of parent axis
                sphere.parent = self.parent_axis
                sphere.scale = self.keypoint_scale
        
        # Create skeleton connections
        for start_id, end_id in self.MHR_SKELETON:
            if start_id in self.T_POSE_POSITIONS and end_id in self.T_POSE_POSITIONS:
                start_pos = self.T_POSE_POSITIONS[start_id]
                end_pos = self.T_POSE_POSITIONS[end_id]
                
                # Get color from the start keypoint
                start_color = self.MHR_KEYPOINTS[start_id]["color"]
                
                # Create line between spheres
                start_name = self.MHR_KEYPOINTS[start_id]['name']
                end_name = self.MHR_KEYPOINTS[end_id]['name']
                line_name = f"{start_name}_to_{end_name}"
                cylinder = self._draw_line(start_pos, end_pos, start_color, line_name)
                self.line_objects[line_name] = cylinder
                
                # Make cylinder a child of parent axis
                cylinder.parent = self.parent_axis
                cylinder.parent_type = 'OBJECT'
        
        print(f"Created MHR skeleton with {len(self.keypoint_objects)} keypoints and {len(self.line_objects)} connections")
        print("All skeleton components are children of 'MHR_Skeleton' at origin")

    def switch_axis(self, position):
        """Switch the axis of the position (can edit this arbitrarily as needed)"""
        return [position[0], position[2], -position[1]]

    def update_keypoint(self, keypoint_name, position):
        """Update the location of a keypoint by name
        
        Args:
            keypoint_name (str): Name of the keypoint to update
            position (list): New position [x, y, z]
        """
        position[0] *= self.x_scale
        position[1] *= self.y_scale
        position[2] *= self.z_scale

        position = self.switch_axis(position)

        if keypoint_name in self.keypoint_objects:
            keypoint_obj = self.keypoint_objects[keypoint_name]
            keypoint_obj.location = Vector(position)
            
            # Update all connected lines
            self._update_connected_lines(keypoint_name)
        else:
            print(f"Warning: Keypoint '{keypoint_name}' not found in skeleton")

    def _update_connected_lines(self, keypoint_name):
        """Update all lines connected to a keypoint"""
        # Find the keypoint ID from the name
        keypoint_id = None
        for kp_id, kp_data in self.MHR_KEYPOINTS.items():
            if kp_data["name"] == keypoint_name:
                keypoint_id = kp_id
                break
        
        if keypoint_id is None:
            return
        
        # Update all lines that connect to this keypoint
        for start_id, end_id in self.MHR_SKELETON:
            if start_id == keypoint_id or end_id == keypoint_id:
                start_name = self.MHR_KEYPOINTS[start_id]["name"]
                end_name = self.MHR_KEYPOINTS[end_id]["name"]
                line_name = f"{start_name}_to_{end_name}"
                
                if line_name in self.line_objects:
                    start_pos = self.keypoint_objects[start_name].location
                    end_pos = self.keypoint_objects[end_name].location
                    
                    start_color = self.MHR_KEYPOINTS[start_id]["color"]
                    self._update_line(line_name, start_pos, end_pos, start_color)

    def _update_line(self, line_name, start_pos, end_pos, color):
        """Update a line's position and rotation"""
        if line_name in self.line_objects:
            line_obj = self.line_objects[line_name]
            
            pos1 = Vector(start_pos)
            pos2 = Vector(end_pos)
            direction = pos2 - pos1
            distance = direction.length
            midpoint = (pos1 + pos2) / 2
            
            line_obj.location = midpoint
            
            if distance > 0:
                direction.normalize()
                z_axis = Vector((0, 0, 1))
                
                if abs(direction.dot(z_axis)) > 0.999:
                    if direction.dot(z_axis) < 0:
                        line_obj.rotation_euler = (math.pi, 0, 0)
                    else:
                        line_obj.rotation_euler = (0, 0, 0)
                else:
                    rotation_axis = z_axis.cross(direction)
                    rotation_axis.normalize()
                    angle = z_axis.angle(direction)
                    
                    from mathutils import Matrix
                    rotation_matrix = Matrix.Rotation(angle, 4, rotation_axis)
                    translation_matrix = Matrix.Translation(midpoint)
                    line_obj.matrix_world = translation_matrix @ rotation_matrix
            
            line_obj.scale = (self.line_scale_xy[0], self.line_scale_xy[1], distance)

    def snapshot(self, frame_id):
        """Create keyframes for all objects at the specified frame"""
        snapshot_data = {
            "frame_id": frame_id,
            "keypoints": {},
            "lines": {},
            "parent_axis": {}
        }
        
        # Keyframe parent axis
        if self.parent_axis:
            self.parent_axis.keyframe_insert(data_path="location", frame=frame_id)
            self.parent_axis.keyframe_insert(data_path="rotation_euler", frame=frame_id)
            self.parent_axis.keyframe_insert(data_path="scale", frame=frame_id)
            
            snapshot_data["parent_axis"] = {
                "location": list(self.parent_axis.location),
                "rotation_euler": list(self.parent_axis.rotation_euler),
                "scale": list(self.parent_axis.scale)
            }
        
        # Keyframe keypoints
        for name, obj in self.keypoint_objects.items():
            obj.keyframe_insert(data_path="location", frame=frame_id)
            obj.keyframe_insert(data_path="rotation_euler", frame=frame_id)
            obj.keyframe_insert(data_path="scale", frame=frame_id)
            
            snapshot_data["keypoints"][name] = {
                "location": list(obj.location),
                "rotation_euler": list(obj.rotation_euler),
                "scale": list(obj.scale)
            }
        
        # Keyframe lines
        for name, obj in self.line_objects.items():
            obj.keyframe_insert(data_path="location", frame=frame_id)
            obj.keyframe_insert(data_path="rotation_euler", frame=frame_id)
            obj.keyframe_insert(data_path="scale", frame=frame_id)
            
            snapshot_data["lines"][name] = {
                "location": list(obj.location),
                "rotation_euler": list(obj.rotation_euler),
                "scale": list(obj.scale)
            }
        
        print(f"Created keyframes for frame {frame_id} with {len(snapshot_data['keypoints'])} keypoints and {len(snapshot_data['lines'])} lines")
        return snapshot_data

    def export_skeleton_data(self):
        """Export the skeleton structure to JSON for use in other applications"""
        skeleton_data = {
            "keypoints": self.MHR_KEYPOINTS,
            "skeleton_connections": self.MHR_SKELETON,
            "t_pose_positions": self.T_POSE_POSITIONS
        }
        
        with open("mhr_skeleton.json", "w") as f:
            json.dump(skeleton_data, f, indent=2)
        
        print("Exported skeleton data to mhr_skeleton.json")


def apply_keypoints(visualizer, keypoints_3d):
    """Apply keypoints to skeleton"""
    for keypoint_id, keypoint_data in visualizer.MHR_KEYPOINTS.items():
        if keypoint_id < len(keypoints_3d):
            x, y, z = keypoints_3d[keypoint_id]
            scaled_position = [x * visualizer.x_scale, y * visualizer.y_scale, z * visualizer.z_scale]
            keypoint_name = keypoint_data["name"]
            visualizer.update_keypoint(keypoint_name, scaled_position)

def apply_track_data(visualizer, track_data):
    """Apply track data to visualizer"""
    keypoints_3d = track_data['keypoints_3d']
    apply_keypoints(visualizer, keypoints_3d)

def snapshot_frame(visualizer, frame_id):
    """Create a snapshot at the specified frame"""
    snapshot_data = visualizer.snapshot(frame_id)
    return snapshot_data
