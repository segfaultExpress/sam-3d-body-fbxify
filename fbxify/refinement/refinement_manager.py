from fbxify.refinement.refinement_config import RefinementConfig
import re
import math
import numpy as np
from typing import Dict, Any, Optional

# ============================================================================
# Vector and Math Utilities
# ============================================================================

def norm(v):
    """Compute L2 norm of a 3D vector."""
    return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])

def dot(v1, v2):
    """Dot product of two vectors (can be 3D or 4D for quaternions)."""
    return sum(a * b for a, b in zip(v1, v2))

def rad2deg(rad):
    """Convert radians to degrees."""
    return rad * 180.0 / math.pi

def deg2rad(deg):
    """Convert degrees to radians."""
    return deg * math.pi / 180.0

def dot4(q1, q2):
    """Dot product of two quaternions."""
    return q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3]

def neg4(q):
    """Negate a quaternion."""
    return [-q[0], -q[1], -q[2], -q[3]]

def quat_normalize(q):
    """Normalize a quaternion."""
    n = math.sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
    if n > 1e-10:
        return [q[0]/n, q[1]/n, q[2]/n, q[3]/n]
    return [1.0, 0.0, 0.0, 0.0]

# ============================================================================
# Quaternion Utilities
# ============================================================================

def quat_from_R(R):
    """
    Convert 3x3 rotation matrix to quaternion (w, x, y, z).
    R: [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]]
    Returns: [w, x, y, z]
    """
    # Trace-based method for numerical stability
    trace = R[0][0] + R[1][1] + R[2][2]
    
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2  # s = 4 * qw
        w = 0.25 * s
        x = (R[2][1] - R[1][2]) / s
        y = (R[0][2] - R[2][0]) / s
        z = (R[1][0] - R[0][1]) / s
    elif R[0][0] > R[1][1] and R[0][0] > R[2][2]:
        s = math.sqrt(1.0 + R[0][0] - R[1][1] - R[2][2]) * 2
        w = (R[2][1] - R[1][2]) / s
        x = 0.25 * s
        y = (R[0][1] + R[1][0]) / s
        z = (R[0][2] + R[2][0]) / s
    elif R[1][1] > R[2][2]:
        s = math.sqrt(1.0 + R[1][1] - R[0][0] - R[2][2]) * 2
        w = (R[0][2] - R[2][0]) / s
        x = (R[0][1] + R[1][0]) / s
        y = 0.25 * s
        z = (R[1][2] + R[2][1]) / s
    else:
        s = math.sqrt(1.0 + R[2][2] - R[0][0] - R[1][1]) * 2
        w = (R[1][0] - R[0][1]) / s
        x = (R[0][2] + R[2][0]) / s
        y = (R[1][2] + R[2][1]) / s
        z = 0.25 * s
    
    # Normalize
    n = math.sqrt(w*w + x*x + y*y + z*z)
    if n > 1e-10:
        return [w/n, x/n, y/n, z/n]
    return [1.0, 0.0, 0.0, 0.0]

def R_from_quat(q):
    """
    Convert quaternion (w, x, y, z) to 3x3 rotation matrix.
    q: [w, x, y, z]
    Returns: [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]]
    """
    w, x, y, z = q
    w2, x2, y2, z2 = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    
    return [
        [w2 + x2 - y2 - z2, 2*(xy - wz), 2*(wy + xz)],
        [2*(wz + xy), w2 - x2 + y2 - z2, 2*(yz - wx)],
        [2*(xz - wy), 2*(wx + yz), w2 - x2 - y2 + z2]
    ]

def quat_mul(q1, q2):
    """Multiply two quaternions: q1 * q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return [
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ]

def quat_inv(q):
    """Inverse of a quaternion (conjugate for unit quaternions)."""
    w, x, y, z = q
    n = w*w + x*x + y*y + z*z
    if n > 1e-10:
        return [w/n, -x/n, -y/n, -z/n]
    return [1.0, 0.0, 0.0, 0.0]

def quat_angle(q):
    """
    Get the rotation angle (in radians) from a quaternion.
    For a unit quaternion q = [cos(θ/2), sin(θ/2)*axis], returns θ.
    """
    w = q[0]
    # Clamp to [-1, 1] for numerical stability
    w = max(-1.0, min(1.0, w))
    return 2.0 * math.acos(abs(w))

def slerp(q1, q2, t):
    """
    Spherical linear interpolation between two quaternions.
    t: interpolation parameter [0, 1]
    """
    # Ensure shortest path
    dot_q = dot(q1, q2)
    if dot_q < 0:
        q2 = [-q2[0], -q2[1], -q2[2], -q2[3]]
        dot_q = -dot_q
    
    # Clamp for numerical stability
    dot_q = max(-1.0, min(1.0, dot_q))
    
    theta = math.acos(dot_q)
    if abs(theta) < 1e-6:
        # Quaternions are very close, use linear interpolation
        return [q1[i] + t * (q2[i] - q1[i]) for i in range(4)]
    
    sin_theta = math.sin(theta)
    w1 = math.sin((1 - t) * theta) / sin_theta
    w2 = math.sin(t * theta) / sin_theta
    
    return [w1 * q1[i] + w2 * q2[i] for i in range(4)]

def quat_log(q):
    """
    Logarithm map: quaternion -> tangent space (axis-angle representation).
    Returns: [x, y, z] (angular velocity vector)
    """
    w = q[0]
    x, y, z = q[1], q[2], q[3]
    
    # Clamp w for numerical stability
    w = max(-1.0, min(1.0, w))
    
    angle = math.acos(abs(w))
    if angle < 1e-6:
        return [0.0, 0.0, 0.0]
    
    sin_angle = math.sin(angle)
    if sin_angle < 1e-6:
        return [0.0, 0.0, 0.0]
    
    scale = 2.0 * angle / sin_angle
    if w < 0:
        scale = -scale
    
    return [scale * x, scale * y, scale * z]

def quat_exp(v):
    """
    Exponential map: tangent space -> quaternion (axis-angle -> quaternion).
    v: [x, y, z] (angular velocity vector)
    Returns: [w, x, y, z]
    """
    angle = norm(v)
    if angle < 1e-6:
        return [1.0, 0.0, 0.0, 0.0]
    
    half_angle = 0.5 * angle
    sin_half = math.sin(half_angle)
    cos_half = math.cos(half_angle)
    
    scale = sin_half / angle
    return [cos_half, scale * v[0], scale * v[1], scale * v[2]]

# ============================================================================
# Filter Functions
# ============================================================================

def butterworth_lowpass(v_series, cutoff_hz, dt, order=2):
    """
    Apply Butterworth lowpass filter to a vector series using biquad (2nd order IIR).
    v_series: [T][3] list of vectors
    cutoff_hz: cutoff frequency in Hz
    dt: time step in seconds
    order: filter order (default 2, only 2 is currently implemented)
    Returns: filtered [T][3] list
    
    This implements a true 2nd-order Butterworth filter with:
    - Maximally flat passband response
    - 40dB/decade rolloff (vs 20dB for first-order/EMA)
    - Proper biquad coefficients via bilinear transform
    """
    T = len(v_series)
    if T < 3:
        return v_series
    
    # Convert cutoff frequency to radians per second
    wc = 2.0 * math.pi * cutoff_hz
    
    # Pre-warp frequency for bilinear transform to compensate for frequency warping
    # This ensures the digital filter matches the analog cutoff frequency
    tan_wc_dt_2 = math.tan(wc * dt / 2.0)
    
    # For 2nd order Butterworth: H(s) = wc^2 / (s^2 + sqrt(2)*wc*s + wc^2)
    # Using bilinear transform: s = (2/dt) * (z-1)/(z+1)
    # After algebra, we get biquad coefficients:
    
    # Normalize by tan term
    k = tan_wc_dt_2
    k2 = k * k
    
    # Biquad coefficients for 2nd order Butterworth lowpass
    # Transfer function: H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)
    a0 = 1.0 + math.sqrt(2) * k + k2
    b0 = k2 / a0
    b1 = 2.0 * k2 / a0
    b2 = k2 / a0
    a1 = 2.0 * (k2 - 1.0) / a0
    a2 = (1.0 - math.sqrt(2) * k + k2) / a0
    
    # Initialize output and filter state (previous values for IIR)
    out = [list(v_series[0])]
    
    # Filter state: [x[n-1], x[n-2], y[n-1], y[n-2]] for each component
    # x = input, y = output
    state = [[[0.0, 0.0, 0.0, 0.0] for _ in range(3)]]
    
    for t in range(1, T):
        filtered = [0.0, 0.0, 0.0]
        for i in range(3):
            # Current input
            x = v_series[t][i]
            
            # Get previous state
            if t == 1:
                x_prev = v_series[0][i]
                x_prev2 = v_series[0][i]
                y_prev = out[0][i]
                y_prev2 = out[0][i]
            else:
                x_prev = v_series[t-1][i]
                x_prev2 = v_series[t-2][i]
                y_prev = out[t-1][i]
                y_prev2 = out[t-2][i]
            
            # Biquad filter: y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
            y = (b0 * x + b1 * x_prev + b2 * x_prev2 - 
                 a1 * y_prev - a2 * y_prev2)
            
            filtered[i] = y
        
        out.append(filtered)
    
    return out

def butterworth_quat(q_series, cutoff_hz, dt, order=2):
    """
    Apply Butterworth filter to quaternion series via tangent space.
    q_series: [T] list of quaternions [w, x, y, z]
    cutoff_hz: cutoff frequency in Hz
    dt: time step in seconds
    Returns: filtered [T] list of quaternions
    """
    T = len(q_series)
    if T < 3:
        return q_series
    
    # Convert to tangent space (log map)
    tangent_vectors = []
    for t in range(T):
        if t == 0:
            tangent_vectors.append([0.0, 0.0, 0.0])
        else:
            q_rel = quat_mul(quat_inv(q_series[t-1]), q_series[t])
            tangent_vectors.append(quat_log(q_rel))
    
    # Filter in tangent space
    filtered_tangent = butterworth_lowpass(tangent_vectors, cutoff_hz, dt, order)
    
    # Re-integrate back to quaternions
    out = [q_series[0]]
    for t in range(1, T):
        q_delta = quat_exp(filtered_tangent[t])
        q_new = quat_mul(out[t-1], q_delta)
        out.append(q_new)
    
    return out

class OneEuroFilter:
    """One Euro filter for scalar values."""
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0, dt=1.0/30.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.dt = dt
        self.x_prev = None
        self.dx_prev = 0.0
    
    def __call__(self, x):
        if self.x_prev is None:
            self.x_prev = x
            return x
        
        # Estimate derivative
        dx = (x - self.x_prev) / self.dt
        
        # Smooth derivative
        edx = self._smooth(dx, self.dx_prev, self.d_cutoff)
        self.dx_prev = edx
        
        # Adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(edx)
        
        # Smooth signal
        x_filtered = self._smooth(x, self.x_prev, cutoff)
        self.x_prev = x_filtered
        
        return x_filtered
    
    def _smooth(self, x, x_prev, cutoff):
        """Simple exponential smoothing."""
        te = 1.0 / (2.0 * math.pi * cutoff)
        alpha = self.dt / (te + self.dt)
        return alpha * x + (1.0 - alpha) * x_prev

def ema_filter_vec3(v_series, cutoff_hz, dt):
    """
    Apply Exponential Moving Average (EMA) lowpass filter to a vector series.
    v_series: [T][3] list of vectors
    cutoff_hz: cutoff frequency in Hz
    dt: time step in seconds
    Returns: filtered [T][3] list
    
    EMA is a first-order IIR filter with 20dB/decade rolloff.
    """
    T = len(v_series)
    if T == 0:
        return v_series
    
    # Calculate alpha from cutoff frequency
    # For a first-order lowpass: alpha = 1 - exp(-2*pi*fc*dt)
    alpha = 1.0 - math.exp(-2.0 * math.pi * cutoff_hz * dt)
    
    out = [list(v_series[0])]
    for t in range(1, T):
        filtered = [out[t-1][i] + alpha * (v_series[t][i] - out[t-1][i]) for i in range(3)]
        out.append(filtered)
    
    return out

def ema_filter_quat(q_series, cutoff_hz, dt):
    """
    Apply EMA filter to quaternion series via SLERP.
    q_series: [T] list of quaternions [w, x, y, z]
    cutoff_hz: cutoff frequency in Hz
    dt: time step in seconds
    Returns: filtered [T] list of quaternions
    
    EMA in SO(3) is performed via spherical linear interpolation (SLERP).
    """
    T = len(q_series)
    if T == 0:
        return q_series
    
    # Calculate alpha from cutoff frequency
    alpha = 1.0 - math.exp(-2.0 * math.pi * cutoff_hz * dt)
    
    out = [q_series[0]]
    for t in range(1, T):
        # EMA in SO(3) via SLERP toward new sample
        out.append(slerp(out[t-1], q_series[t], alpha))
    
    return out

def one_euro_filter_vec3(v_series, dt, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
    """
    Apply One Euro filter to a 3D vector series.
    v_series: [T][3] list of vectors
    dt: time step in seconds
    min_cutoff: minimum cutoff frequency
    beta: speed coefficient
    d_cutoff: derivative cutoff frequency
    Returns: filtered [T][3] list
    """
    T = len(v_series)
    if T == 0:
        return v_series
    
    # Create separate filters for each component
    filters = [OneEuroFilter(min_cutoff, beta, d_cutoff, dt) for _ in range(3)]
    
    out = []
    for t in range(T):
        filtered = [filters[i](v_series[t][i]) for i in range(3)]
        out.append(filtered)
    
    return out

def fix_quat_hemisphere(qs):
    out = [qs[0]]
    for t in range(1, len(qs)):
        out.append(qs[t] if dot4(out[t-1], qs[t]) >= 0 else neg4(qs[t]))
    return out

def one_euro_filter_quat(q_series, dt, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
    T = len(q_series)
    if T < 2:
        return q_series

    q_series = fix_quat_hemisphere(q_series)

    tangent = [[0.0,0.0,0.0]]
    for t in range(1, T):
        q_rel = quat_mul(quat_inv(q_series[t-1]), q_series[t])
        tangent.append(quat_log(q_rel))

    filtered = one_euro_filter_vec3(tangent, dt, min_cutoff, beta, d_cutoff)

    out = [quat_normalize(q_series[0])]
    for t in range(1, T):
        q_delta = quat_exp(filtered[t])
        q_new = quat_mul(out[t-1], q_delta)
        out.append(quat_normalize(q_new))

    return out



# ============================================================================
# Refinement Manager
# ============================================================================

class RefinementManager:
    """
    RefinementManager is a class that applies refinement and mocap-style smoothing to the animation.
    """
    def __init__(self, config: RefinementConfig = None, fps: float = 30.0):
        if config is None:
            # use default config
            config = RefinementConfig()
        self.configure(config, fps)

    def configure(self, config: RefinementConfig, fps: float = 30.0):
        self.config = config
        self.fps = fps
        self.dt = 1.0 / fps  # time step in seconds
    
    def _calculate_vector_change_percent(self, v_original, v_refined):
        """
        Calculate the percentage change between original and refined vector series.
        Returns the average percentage change in magnitude.
        """
        if len(v_original) == 0 or len(v_refined) == 0:
            return 0.0
        
        total_change = 0.0
        total_original_mag = 0.0
        count = 0
        
        for t in range(min(len(v_original), len(v_refined))):
            orig_mag = norm(v_original[t])
            refined_mag = norm(v_refined[t])
            
            if orig_mag > 1e-6:  # Avoid division by zero
                change = abs(refined_mag - orig_mag) / orig_mag * 100.0
                total_change += change
                total_original_mag += orig_mag
                count += 1
        
        if count == 0:
            return 0.0
        
        return total_change / count
    
    def _identity_matrix(self):
        """Return a 3x3 identity matrix."""
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    
    def _convert_to_list(self, value):
        """Convert numpy array or other types to list."""
        if hasattr(value, 'tolist'):
            return value.tolist()
        if hasattr(value, 'shape'):
            import numpy as np
            if isinstance(value, np.ndarray):
                return value.tolist()
        return value
    
    def _parse_3x3_nested_matrix(self, rot_list):
        """Parse a 3x3 nested list matrix: [[a,b,c], [d,e,f], [g,h,i]]."""
        if not isinstance(rot_list, list) or len(rot_list) != 3:
            return None
        if not isinstance(rot_list[0], (list, tuple)) or len(rot_list[0]) != 3:
            return None
        return [[float(rot_list[i][j]) for j in range(3)] for i in range(3)]
    
    def _parse_3x3_flattened_matrix(self, rot_list):
        """Parse a flattened 9-element list into 3x3 matrix: [a,b,c,d,e,f,g,h,i]."""
        if not isinstance(rot_list, list) or len(rot_list) != 9:
            return None
        return [[float(rot_list[i*3 + j]) for j in range(3)] for i in range(3)]
    
    def _parse_rotation_matrix(self, rot_t):
        """Parse a rotation matrix from various formats, return 3x3 list or None."""
        rot_t = self._convert_to_list(rot_t)
        
        if not isinstance(rot_t, list):
            return None
        
        # Try nested 3x3 format first
        nested = self._parse_3x3_nested_matrix(rot_t)
        if nested is not None:
            return nested
        
        # Try flattened 9-element format
        flattened = self._parse_3x3_flattened_matrix(rot_t)
        if flattened is not None:
            return flattened
        
        return None
    
    def _deep_copy_rotation_series(self, rot):
        """Create a deep copy of rotation series, handling various formats."""
        rot_original = []
        for t in range(len(rot)):
            try:
                parsed = self._parse_rotation_matrix(rot[t])
                rot_original.append(parsed if parsed is not None else self._identity_matrix())
            except (TypeError, IndexError) as e:
                print(f"Warning: Could not parse rotation matrix at frame {t}: {e}, using identity")
                rot_original.append(self._identity_matrix())
        return rot_original
    
    def _calculate_rotation_change_percent(self, R_original, R_refined):
        """
        Calculate the percentage change between original and refined rotation series.
        Returns the average angular change in degrees.
        """
        if len(R_original) == 0 or len(R_refined) == 0:
            return 0.0
        
        q_original = [quat_from_R(R) for R in R_original]
        q_refined = [quat_from_R(R) for R in R_refined]
        q_original = fix_quat_hemisphere(q_original)
        q_refined = fix_quat_hemisphere(q_refined)
        
        total_angle_change = 0.0
        count = 0
        
        for t in range(min(len(q_original), len(q_refined))):
            # Calculate relative rotation between original and refined
            q_rel = quat_mul(quat_inv(q_original[t]), q_refined[t])
            angle_deg = rad2deg(quat_angle(q_rel))
            total_angle_change += angle_deg
            count += 1
        
        if count == 0:
            return 0.0
        
        return total_angle_change / count

    def apply(self, estimation_results: Dict[str, Dict[str, Any]], 
             progress_callback: Optional[callable] = None) -> Dict[str, Dict[str, Any]]:
        """
        Apply refinement directly to estimation results (before joint mapping).
        
        This method refines:
        - pred_global_rots: Joint rotations (array of 3x3 matrices per frame)
        - global_rot: Root rotation (3x3 matrix per frame)
        - pred_cam_t: Root translation (3D vector per frame)
        
        Args:
            estimation_results: Dictionary in format {frame_X: {person_id: estimation_data}}
            progress_callback: Optional callback function(progress, description)
            
        Returns:
            Refined estimation results in the same format
        """
        if self.config is None:
            return estimation_results
        
        if progress_callback:
            progress_callback(0.0, "Applying refinement to estimation results...")
        
        # Get all person IDs across all frames
        all_person_ids = set()
        for frame_data in estimation_results.values():
            for person_id in frame_data.keys():
                all_person_ids.add(person_id)
        
        # Get all frame indices sorted
        frame_indices = sorted([int(k) for k in estimation_results.keys()])
        num_frames = len(frame_indices)
        
        if num_frames == 0:
            return estimation_results
        
        refined_results = {}
        
        # Process each person separately
        for person_index, person_id in enumerate(all_person_ids):
            if progress_callback:
                progress = person_index / len(all_person_ids)
                progress_callback(progress, f"Refining person {person_index + 1} of {len(all_person_ids)}")
            
            # Collect data for this person across all frames
            joint_rotations_series = []  # [T][num_joints][3][3]
            root_rotations_series = []    # [T][3][3]
            root_translations_series = [] # [T][3]
            joint_coords_series = []      # [T][num_joints][3] (for foot planting if needed)
            
            # First pass: collect all data
            for frame_idx in frame_indices:
                frame_key = str(frame_idx)
                frame_data = estimation_results.get(frame_key, {})
                person_data = frame_data.get(str(person_id))
                
                if person_data is None:
                    # Missing frame - add None placeholders
                    joint_rotations_series.append(None)
                    root_rotations_series.append(None)
                    root_translations_series.append(None)
                    joint_coords_series.append(None)
                    continue
                
                # Extract data
                pred_global_rots = person_data.get("pred_global_rots")
                global_rot = person_data.get("global_rot")
                pred_cam_t = person_data.get("pred_cam_t")
                pred_joint_coords = person_data.get("pred_joint_coords")
                
                # Convert to lists if numpy arrays
                if pred_global_rots is not None:
                    if hasattr(pred_global_rots, 'tolist'):
                        pred_global_rots = pred_global_rots.tolist()
                    joint_rotations_series.append(pred_global_rots)
                else:
                    joint_rotations_series.append(None)
                
                if global_rot is not None:
                    if hasattr(global_rot, 'tolist'):
                        global_rot = global_rot.tolist()
                    root_rotations_series.append(global_rot)
                else:
                    root_rotations_series.append(None)
                
                if pred_cam_t is not None:
                    if hasattr(pred_cam_t, 'tolist'):
                        pred_cam_t = pred_cam_t.tolist()
                    root_translations_series.append(pred_cam_t)
                else:
                    root_translations_series.append(None)
                
                if pred_joint_coords is not None:
                    if hasattr(pred_joint_coords, 'tolist'):
                        pred_joint_coords = pred_joint_coords.tolist()
                    joint_coords_series.append(pred_joint_coords)
                else:
                    joint_coords_series.append(None)
            
            # Refine joint rotations (each joint separately)
            if joint_rotations_series and any(x is not None for x in joint_rotations_series):
                # Get number of joints from first non-None frame
                num_joints = None
                for rot_frame in joint_rotations_series:
                    if rot_frame is not None:
                        num_joints = len(rot_frame)
                        break
                
                if num_joints is not None:
                    # Refine each joint separately
                    refined_joint_rots = []
                    for joint_idx in range(num_joints):
                        # Extract rotation series for this joint: [T][3][3]
                        joint_rot_series = []
                        for t in range(num_frames):
                            if joint_rotations_series[t] is not None and joint_idx < len(joint_rotations_series[t]):
                                joint_rot = joint_rotations_series[t][joint_idx]
                                # Parse rotation matrix to ensure it's in 3x3 format
                                parsed = self._parse_rotation_matrix(joint_rot)
                                joint_rot_series.append(parsed if parsed is not None else self._identity_matrix())
                            else:
                                joint_rot_series.append(None)
                        
                        # Refine this joint's rotation series
                        refined_joint_rot = self._process_rotation_series(
                            joint_rot_series, 
                            self.config.profiles.get("*", self.config.profiles.get("root")),
                            bone_name=f"joint_{joint_idx}"
                        )
                        refined_joint_rots.append(refined_joint_rot)
                    
                    # Reorganize: [num_joints][T][3][3] -> [T][num_joints][3][3]
                    joint_rotations_series = []
                    for t in range(num_frames):
                        frame_joint_rots = []
                        for joint_idx in range(num_joints):
                            frame_joint_rots.append(refined_joint_rots[joint_idx][t])
                        joint_rotations_series.append(frame_joint_rots)
            
            # Refine root rotation
            if root_rotations_series and any(x is not None for x in root_rotations_series):
                # Parse rotation matrices to ensure they're in 3x3 format
                parsed_root_rotations = []
                for rot in root_rotations_series:
                    if rot is None:
                        parsed_root_rotations.append(None)
                    else:
                        parsed = self._parse_rotation_matrix(rot)
                        parsed_root_rotations.append(parsed if parsed is not None else self._identity_matrix())
                
                prof = self.config.profiles.get("root", self.config.profiles.get("*"))
                root_rotations_series = self._process_rotation_series(
                    parsed_root_rotations,
                    prof,
                    bone_name="root_rotation"
                )
            
            # Refine root translation
            if root_translations_series and any(x is not None for x in root_translations_series):
                prof = self.config.profiles.get("root", self.config.profiles.get("*"))
                root_translations_series = self._process_vector_series(
                    root_translations_series,
                    prof,
                    bone_name="root_translation"
                )
            
            # Apply root motion stabilization (combines rotation and translation)
            if self.config.do_root_motion_fix and root_rotations_series and root_translations_series:
                root_motion_dict = {
                    "translation": root_translations_series,
                    "rotation": root_rotations_series
                }
                root_motion_dict = self._root_stabilization(root_motion_dict)
                root_translations_series = root_motion_dict["translation"]
                root_rotations_series = root_motion_dict["rotation"]
            
            # Store refined data back into results
            for frame_idx, frame_key in enumerate([str(f) for f in frame_indices]):
                if frame_key not in refined_results:
                    refined_results[frame_key] = {}
                
                # Get original person data to preserve other fields
                original_frame_data = estimation_results.get(frame_key, {})
                original_person_data = original_frame_data.get(str(person_id), {})
                
                # Create refined person data
                refined_person_data = original_person_data.copy()
                
                # Update with refined values
                if frame_idx < len(joint_rotations_series) and joint_rotations_series[frame_idx] is not None:
                    refined_person_data["pred_global_rots"] = joint_rotations_series[frame_idx]
                
                if frame_idx < len(root_rotations_series) and root_rotations_series[frame_idx] is not None:
                    refined_person_data["global_rot"] = root_rotations_series[frame_idx]
                
                if frame_idx < len(root_translations_series) and root_translations_series[frame_idx] is not None:
                    refined_person_data["pred_cam_t"] = root_translations_series[frame_idx]
                
                refined_results[frame_key][str(person_id)] = refined_person_data
        
        if progress_callback:
            progress_callback(1.0, "Refinement complete")
        
        return refined_results

    def _interpolate_missing_frames(self, series, is_rotation=False):
        """
        Interpolate None values in a series.
        - series: List of values (rotations [3][3] or vectors [3]) with possible None entries
        - is_rotation: If True, use slerp; if False, use linear interpolation
        - Returns: Series with None values replaced by interpolated values
        """
        # Check if there are any None values
        has_none = any(x is None for x in series)
        if not has_none:
            return series
        
        # Make a copy to avoid modifying the original
        result = series[:]
        T = len(result)
        
        # Find all valid indices
        valid_indices = [i for i in range(T) if result[i] is not None]
        
        # If all frames are None, return original unchanged
        if len(valid_indices) == 0:
            return result
        
        # If only one valid frame, use it for all None values
        if len(valid_indices) == 1:
            valid_idx = valid_indices[0]
            valid_value = result[valid_idx]
            for i in range(T):
                if result[i] is None:
                    if is_rotation:
                        # Deep copy rotation matrix
                        result[i] = [[valid_value[j][k] for k in range(3)] for j in range(3)]
                    else:
                        # Deep copy vector
                        result[i] = list(valid_value)
            return result
        
        # Find sequences of None values and interpolate
        i = 0
        while i < T:
            if result[i] is None:
                # Find the start of this None sequence
                start_none = i
                # Find the end of this None sequence
                while i < T and result[i] is None:
                    i += 1
                end_none = i
                
                # Find the last valid frame before this sequence
                prev_valid_idx = None
                for j in range(start_none - 1, -1, -1):
                    if result[j] is not None:
                        prev_valid_idx = j
                        break
                
                # Find the first valid frame after this sequence
                next_valid_idx = None
                for j in range(end_none, T):
                    if result[j] is not None:
                        next_valid_idx = j
                        break
                
                # Interpolate each None in the sequence
                if prev_valid_idx is not None and next_valid_idx is not None:
                    # We have both previous and next valid frames
                    prev_value = result[prev_valid_idx]
                    next_value = result[next_valid_idx]
                    
                    for j in range(start_none, end_none):
                        # Calculate interpolation parameter
                        # t = 0 at prev_valid_idx, t = 1 at next_valid_idx
                        t = (j - prev_valid_idx) / (next_valid_idx - prev_valid_idx)
                        
                        if is_rotation:
                            # Convert matrices to quaternions, slerp, convert back
                            q1 = quat_from_R(prev_value)
                            q2 = quat_from_R(next_value)
                            q_interp = slerp(q1, q2, t)
                            result[j] = R_from_quat(q_interp)
                        else:
                            # Linear interpolation for vectors
                            result[j] = [
                                prev_value[k] + t * (next_value[k] - prev_value[k])
                                for k in range(3)
                            ]
                elif prev_valid_idx is not None:
                    # Only previous valid frame (None at end)
                    prev_value = result[prev_valid_idx]
                    for j in range(start_none, end_none):
                        if is_rotation:
                            # Deep copy rotation matrix
                            result[j] = [[prev_value[k][l] for l in range(3)] for k in range(3)]
                        else:
                            # Deep copy vector
                            result[j] = list(prev_value)
                elif next_valid_idx is not None:
                    # Only next valid frame (None at start)
                    next_value = result[next_valid_idx]
                    for j in range(start_none, end_none):
                        if is_rotation:
                            # Deep copy rotation matrix
                            result[j] = [[next_value[k][l] for l in range(3)] for k in range(3)]
                        else:
                            # Deep copy vector
                            result[j] = list(next_value)
            else:
                i += 1
        
        return result

    def _process_vector_series(self, v_series, prof, bone_name=None):
        # CRITICAL: Interpolation must happen FIRST, before any other processing
        if self.config.do_interpolate_missing_keyframes:
            # Interpolate missing frames before processing
            v_series = self._interpolate_missing_frames(v_series, is_rotation=False)
        else:
            # Check for None values and return early if found
            if any(v is None for v in v_series):
                print(f"  Warning: Missing keyframes detected in {bone_name or 'vector series'}, skipping refinement (enable 'Interpolate Missing Keyframes' to interpolate)")
                return v_series
        
        # Defensive check: ensure no None values remain before processing
        if any(v is None for v in v_series):
            print(f"  Error: None values still present in {bone_name or 'vector series'} after interpolation check, skipping refinement")
            return v_series
        
        v_original = [list(v) for v in v_series]  # Deep copy for comparison
        v = v_series[:]  # [T][3]

        if self.config.do_spike_fix:
            v = self._despike_vector(v, prof)

        if self.config.do_vector_smoothing:
            v = self._smooth_vector(v, prof)
        
        # Calculate and report percentage change
        if bone_name:
            change_percent = self._calculate_vector_change_percent(v_original, v)
            if change_percent > 0.01:  # Only report if there's meaningful change
                print(f"  {bone_name}: adjusted vector by {change_percent:.2f}%")

        return v

    def _despike_vector(self, v, prof):
        # velocity/accel based outlier removal
        T = len(v)
        if T < 3:
            return v

        # precompute vel/acc
        vel = [[0.0, 0.0, 0.0] for _ in range(T)]
        acc = [[0.0, 0.0, 0.0] for _ in range(T)]
        for t in range(1, T):
            vel[t] = [(v[t][i] - v[t-1][i]) / self.dt for i in range(3)]
        for t in range(2, T):
            acc[t] = [(vel[t][i] - vel[t-1][i]) / self.dt for i in range(3)]

        for t in range(1, T-1):
            speed = norm(vel[t])
            a = norm(acc[t])

            is_spike = (speed > prof.max_pos_speed and a > prof.max_pos_accel)

            # classic "single frame pop": neighbors are consistent but middle isn't
            if is_spike:
                v[t] = [0.5 * (v[t-1][i] + v[t+1][i]) for i in range(3)]

        return v

    def _smooth_vector(self, v, prof):
        if prof.method == "ema":
            return ema_filter_vec3(v, cutoff_hz=prof.cutoff_hz, dt=self.dt)

        if prof.method == "butterworth":
            # placeholder: you’d design biquad coefficients for cutoff_hz
            return butterworth_lowpass(v, cutoff_hz=prof.cutoff_hz, dt=self.dt)

        if prof.method == "one_euro":
            return one_euro_filter_vec3(
                v, dt=self.dt,
                min_cutoff=prof.one_euro_min_cutoff,
                beta=prof.one_euro_beta,
                d_cutoff=prof.one_euro_d_cutoff
            )

        return v

    def _process_rotation_series(self, R_series, prof, bone_name=None):
        # R_series: [T][3][3]
        
        # CRITICAL: Interpolation must happen FIRST, before any other processing
        if self.config.do_interpolate_missing_keyframes:
            # Interpolate missing frames before processing
            R_series = self._interpolate_missing_frames(R_series, is_rotation=True)
        else:
            # Check for None values and return early if found
            if any(R is None for R in R_series):
                print(f"  Warning: Missing keyframes detected in {bone_name or 'rotation series'}, skipping refinement (enable 'Interpolate Missing Keyframes' to interpolate)")
                return R_series
        
        R_original = [[[R[i][j] for j in range(3)] for i in range(3)] for R in R_series]  # Deep copy
        q = [quat_from_R(R) for R in R_series]   # [T] quats
        q = fix_quat_hemisphere(q)

        if self.config.do_spike_fix:
            q = self._despike_rotation(q, prof)

        if self.config.do_rotation_smoothing:
            q = self._smooth_rotation(q, prof)

        # back to matrices
        R_refined = [R_from_quat(qt) for qt in q]
        
        # Calculate and report percentage change
        if bone_name:
            change_deg = self._calculate_rotation_change_percent(R_original, R_refined)
            if change_deg > 0.1:  # Only report if there's meaningful change (>0.1 degrees)
                print(f"  {bone_name}: adjusted rotation by {change_deg:.2f}° (avg)")
        
        return R_refined

    def _despike_rotation(self, q, prof):
        T = len(q)
        if T < 3:
            return q

        ang_vel = [0.0] * T
        ang_acc = [0.0] * T

        for t in range(1, T):
            dq = quat_mul(quat_inv(q[t-1]), q[t])
            angle_deg = rad2deg(quat_angle(dq))          # shortest angle
            ang_vel[t] = angle_deg / self.dt

        for t in range(2, T):
            ang_acc[t] = (ang_vel[t] - ang_vel[t-1]) / self.dt

        for t in range(1, T-1):
            is_spike = (ang_vel[t] > prof.max_ang_speed_deg and
                        ang_acc[t] > prof.max_ang_accel_deg)

            if is_spike:
                # replace with slerp neighbor midpoint
                q[t] = slerp(q[t-1], q[t+1], 0.5)

        return q

    def _smooth_rotation(self, q, prof):
        if prof.method == "ema":
            return ema_filter_quat(q, cutoff_hz=prof.cutoff_hz, dt=self.dt)

        if prof.method == "one_euro":
            # OneEuro on rotation *vector* in tangent space:
            # r_t = log( inv(q_prev) * q_t ) / dt  (angular velocity in local frame)
            return one_euro_filter_quat(
                q, dt=self.dt,
                min_cutoff=prof.one_euro_min_cutoff,
                beta=prof.one_euro_beta,
                d_cutoff=prof.one_euro_d_cutoff
            )

        if prof.method == "butterworth":
            # Butterworth on tangent vectors (log map), then re-integrate.
            return butterworth_quat(q, cutoff_hz=prof.cutoff_hz, dt=self.dt)

        return q

    def _profile_for(self, bone_name):
        # very rough wildcard matching
        for pattern, prof in self.config.profiles.items():
            if self._match(pattern, bone_name):
                return prof
        return self.config.profiles["*"]

    def _match(self, pattern, bone_name):
        bone_name = bone_name.lower()
        regex = re.compile(pattern.replace("*", ".*").lower())
        return regex.match(bone_name) is not None

    def _root_stabilization(self, root_motion):
        """
        Stabilize root motion to reduce jitter and unwanted movement.
        root_motion: dict with keys like "translation" [T][3] and "rotation" [T][3][3]
        Returns: stabilized root_motion dict
        """
        if root_motion is None:
            return root_motion
        
        prof = self.config.profiles.get("root", self.config.profiles["*"])
        stabilized = {}
        
        # Stabilize translation (position)
        if "translation" in root_motion:
            trans = root_motion["translation"]  # [T][3]
            trans_original = [[t[i] for i in range(3)] for t in trans]  # Deep copy for comparison
            T = len(trans)
            
            if T > 0:
                # Apply different cutoffs for XY (horizontal) vs Z (vertical)
                # For root motion, we want different smoothing for horizontal vs vertical
                if prof.method == "one_euro":
                    # Filter X and Y with horizontal cutoff
                    filter_x = OneEuroFilter(
                        min_cutoff=prof.root_cutoff_xy_hz,
                        beta=prof.one_euro_beta,
                        d_cutoff=prof.one_euro_d_cutoff,
                        dt=self.dt
                    )
                    filter_y = OneEuroFilter(
                        min_cutoff=prof.root_cutoff_xy_hz,
                        beta=prof.one_euro_beta,
                        d_cutoff=prof.one_euro_d_cutoff,
                        dt=self.dt
                    )
                    # Filter Z with vertical cutoff (typically lower for less jitter)
                    filter_z = OneEuroFilter(
                        min_cutoff=prof.root_cutoff_z_hz,
                        beta=prof.one_euro_beta,
                        d_cutoff=prof.one_euro_d_cutoff,
                        dt=self.dt
                    )
                    
                    filtered_trans = []
                    for t in range(T):
                        filtered_trans.append([
                            filter_x(trans[t][0]),
                            filter_y(trans[t][1]),
                            filter_z(trans[t][2])
                        ])
                    stabilized["translation"] = filtered_trans
                elif prof.method == "ema":
                    # Apply EMA with separate cutoffs for XY vs Z
                    # Extract components
                    x_series = [trans[t][0] for t in range(T)]
                    y_series = [trans[t][1] for t in range(T)]
                    z_series = [trans[t][2] for t in range(T)]
                    
                    # Calculate alpha values for each cutoff
                    alpha_xy = 1.0 - math.exp(-2.0 * math.pi * prof.root_cutoff_xy_hz * self.dt)
                    alpha_z = 1.0 - math.exp(-2.0 * math.pi * prof.root_cutoff_z_hz * self.dt)
                    
                    # Apply EMA filter to each component separately
                    filtered_x = [x_series[0]]
                    filtered_y = [y_series[0]]
                    filtered_z = [z_series[0]]
                    
                    for t in range(1, T):
                        filtered_x.append(filtered_x[t-1] + alpha_xy * (x_series[t] - filtered_x[t-1]))
                        filtered_y.append(filtered_y[t-1] + alpha_xy * (y_series[t] - filtered_y[t-1]))
                        filtered_z.append(filtered_z[t-1] + alpha_z * (z_series[t] - filtered_z[t-1]))
                    
                    # Recombine
                    filtered_trans = []
                    for t in range(T):
                        filtered_trans.append([
                            filtered_x[t],
                            filtered_y[t],
                            filtered_z[t]
                        ])
                    stabilized["translation"] = filtered_trans
                else:
                    # For other methods, use profile cutoffs (standard processing)
                    filtered = self._process_vector_series(trans, prof)
                    stabilized["translation"] = filtered
                
                # Calculate and report percentage change for root translation
                change_percent = self._calculate_vector_change_percent(trans_original, stabilized["translation"])
                if change_percent > 0.01:  # Only report if there's meaningful change
                    print(f"root_stabilization: adjusted root translation by {change_percent:.2f}%")
        
        # Stabilize rotation
        if "rotation" in root_motion:
            rot = root_motion["rotation"]  # [T][3][3]
            rot_original = self._deep_copy_rotation_series(rot)
            
            # Process without bone_name to avoid duplicate message
            stabilized["rotation"] = self._process_rotation_series(rot, prof, bone_name=None)
            
            # Calculate and report percentage change for root rotation
            change_deg = self._calculate_rotation_change_percent(rot_original, stabilized["rotation"])
            if change_deg > 0.1:  # Only report if there's meaningful change
                print(f"root_stabilization: adjusted root rotation by {change_deg:.2f}° (avg)")
        
        # Copy any other fields
        for key in root_motion:
            if key not in stabilized:
                stabilized[key] = root_motion[key]
        
        return stabilized
    
    def _foot_planting_adjustment(self, root_motion, joint_to_bone_mapping):
        """
        Adjust root motion based on foot contact to reduce jitter.
        
        Args:
            root_motion: dict with "translation" [T][3] and "rotation" [T][3][3]
            joint_to_bone_mapping: joint-to-bone mapping structure (for accessing bone data)
        
        Returns:
            Adjusted root_motion dict
        """
        
        if not self.config.do_foot_planting:
            return root_motion
        
        

        return root_motion

    def _foot_planting_ai_slopcode(self, root_motion, joint_to_bone_mapping):
        """
        Adjust root motion based on foot contact to reduce jitter.
        
        Args:
            root_motion: dict with "translation" [T][3] and "rotation" [T][3][3]
            joint_to_bone_mapping: joint-to-bone mapping structure (for accessing bone data)
        
        Returns:
            Adjusted root_motion dict
        """
        if not self.config.do_foot_planting:
            return root_motion
        
        fp_config = self.config.foot_planting_config
        
        # Debug: Check if we have the necessary data
        print(f"\n=== FOOT PLANTING DEBUG ===")
        print(f"Foot planting enabled: {self.config.do_foot_planting}")
        print(f"Config: velocity_threshold={fp_config.foot_contact_velocity_threshold}, "
              f"min_height={fp_config.foot_contact_min_height}, "
              f"blend_factor={fp_config.blend_factor}")
        
        if root_motion is None:
            print("ERROR: root_motion is None")
            return root_motion
        
        if "translation" not in root_motion:
            print("ERROR: root_motion missing 'translation' key")
            return root_motion
        
        T = len(root_motion["translation"])
        print(f"Number of frames: {T}")
        print(f"joint_to_bone_mapping is None: {joint_to_bone_mapping is None}")
        
        if joint_to_bone_mapping is None:
            print("WARNING: joint_to_bone_mapping is None - cannot compute foot positions")
            print("Foot planting will be skipped.")
            return root_motion
        
        # Try to find foot bones in the hierarchy
        def find_bone_by_name(bone_dict, name):
            """Recursively find a bone by name in the hierarchy."""
            if not isinstance(bone_dict, dict):
                return None
            if bone_dict.get("name") == name:
                return bone_dict
            for child in bone_dict.get("children", []):
                result = find_bone_by_name(child, name)
                if result is not None:
                    return result
            return None
        
        left_foot_bone = find_bone_by_name(joint_to_bone_mapping, "l_foot")
        right_foot_bone = find_bone_by_name(joint_to_bone_mapping, "r_foot")
        
        print(f"Left foot bone found: {left_foot_bone is not None}")
        print(f"Right foot bone found: {right_foot_bone is not None}")
        
        if left_foot_bone is not None:
            print(f"Left foot bone structure: {list(left_foot_bone.keys())}")
            if "data" in left_foot_bone:
                print(f"Left foot data keys: {list(left_foot_bone['data'].keys()) if isinstance(left_foot_bone.get('data'), dict) else 'N/A'}")
                if "rotation" in left_foot_bone.get("data", {}):
                    rot_data = left_foot_bone["data"]["rotation"]
                    print(f"Left foot rotation data: {len(rot_data) if isinstance(rot_data, list) else 'N/A'} frames")
        
        if right_foot_bone is not None:
            print(f"Right foot bone structure: {list(right_foot_bone.keys())}")
            if "data" in right_foot_bone:
                print(f"Right foot data keys: {list(right_foot_bone['data'].keys()) if isinstance(right_foot_bone.get('data'), dict) else 'N/A'}")
        
        # Try to find bones and check what data they have
        def find_bone_data(bone_dict, name):
            """Find bone and return all available data."""
            bone = find_bone_by_name(bone_dict, name)
            if bone is None:
                return None, {}
            data = bone.get("data", {})
            return bone, data
        
        left_ankle_bone, left_ankle_data = find_bone_data(joint_to_bone_mapping, "l_lowleg")
        right_ankle_bone, right_ankle_data = find_bone_data(joint_to_bone_mapping, "r_lowleg")
        left_foot_bone_data = left_foot_bone.get("data", {}) if left_foot_bone else {}
        right_foot_bone_data = right_foot_bone.get("data", {}) if right_foot_bone else {}
        
        print(f"Left ankle (l_lowleg) data keys: {list(left_ankle_data.keys())}")
        print(f"Right ankle (r_lowleg) data keys: {list(right_ankle_data.keys())}")
        print(f"Left foot data keys: {list(left_foot_bone_data.keys())}")
        print(f"Right foot data keys: {list(right_foot_bone_data.keys())}")
        
        # Check for coords, joint_coords, or rotation data
        left_ankle_coords = left_ankle_data.get("coords") or left_ankle_data.get("joint_coords")
        right_ankle_coords = right_ankle_data.get("coords") or right_ankle_data.get("joint_coords")
        left_foot_coords = left_foot_bone_data.get("coords") or left_foot_bone_data.get("joint_coords")
        right_foot_coords = right_foot_bone_data.get("coords") or right_foot_bone_data.get("joint_coords")
        
        print(f"Left ankle coords available: {left_ankle_coords is not None and len(left_ankle_coords) > 0 if left_ankle_coords else False}")
        print(f"Right ankle coords available: {right_ankle_coords is not None and len(right_ankle_coords) > 0 if right_ankle_coords else False}")
        print(f"Left foot coords available: {left_foot_coords is not None and len(left_foot_coords) > 0 if left_foot_coords else False}")
        print(f"Right foot coords available: {right_foot_coords is not None and len(right_foot_coords) > 0 if right_foot_coords else False}")
        
        # Validate we have the necessary data
        if not left_ankle_coords or not right_ankle_coords:
            print("WARNING: Missing ankle coordinate data - foot planting skipped")
            return root_motion
        
        # Choose which keypoint to use for contact detection
        if fp_config.use_mid_foot and left_foot_coords and right_foot_coords:
            left_contact_coords = left_foot_coords
            right_contact_coords = right_foot_coords
            contact_name = "mid-foot"
        else:
            left_contact_coords = left_ankle_coords
            right_contact_coords = right_ankle_coords
            contact_name = "ankle"
        
        print(f"Using {contact_name} keypoints for contact detection")
        
        # Convert to numpy arrays for easier manipulation
        root_trans = np.array(root_motion["translation"])  # [T, 3]
        
        # Build world-space foot positions: root_trans + relative_foot_pos
        left_foot_world = np.zeros((T, 3))
        right_foot_world = np.zeros((T, 3))
        
        for t in range(T):
            root_t = root_trans[t]
            
            # Get relative foot positions
            if t < len(left_contact_coords) and left_contact_coords[t] is not None:
                left_rel = np.array(left_contact_coords[t][:3])
                left_foot_world[t] = root_t + left_rel
            else:
                left_foot_world[t] = root_t  # Fallback to root if missing
            
            if t < len(right_contact_coords) and right_contact_coords[t] is not None:
                right_rel = np.array(right_contact_coords[t][:3])
                right_foot_world[t] = root_t + right_rel
            else:
                right_foot_world[t] = root_t  # Fallback to root if missing
        
        # Calculate foot velocities
        # For contact detection, we want RELATIVE velocity (foot movement relative to root)
        # because if the root moves forward, the foot's world position changes even if it's planted
        left_foot_relative_velocity = np.zeros((T, 3))
        right_foot_relative_velocity = np.zeros((T, 3))
        
        # Get relative foot positions (already in local/root space)
        left_foot_relative = np.zeros((T, 3))
        right_foot_relative = np.zeros((T, 3))
        
        for t in range(T):
            if t < len(left_contact_coords) and left_contact_coords[t] is not None:
                left_foot_relative[t] = np.array(left_contact_coords[t][:3])
            if t < len(right_contact_coords) and right_contact_coords[t] is not None:
                right_foot_relative[t] = np.array(right_contact_coords[t][:3])
        
        # Calculate relative velocity (change in relative position per frame)
        for t in range(1, T):
            left_foot_relative_velocity[t] = left_foot_relative[t] - left_foot_relative[t-1]
            right_foot_relative_velocity[t] = right_foot_relative[t] - right_foot_relative[t-1]
        
        # Calculate speed (magnitude of relative velocity)
        left_foot_speed = np.linalg.norm(left_foot_relative_velocity, axis=1)  # [T]
        right_foot_speed = np.linalg.norm(right_foot_relative_velocity, axis=1)  # [T]
        
        # Also calculate world velocity for debugging
        left_foot_world_velocity = np.zeros((T, 3))
        right_foot_world_velocity = np.zeros((T, 3))
        for t in range(1, T):
            left_foot_world_velocity[t] = left_foot_world[t] - left_foot_world[t-1]
            right_foot_world_velocity[t] = right_foot_world[t] - right_foot_world[t-1]
        left_foot_world_speed = np.linalg.norm(left_foot_world_velocity, axis=1)
        right_foot_world_speed = np.linalg.norm(right_foot_world_velocity, axis=1)
        
        # Find ground level (minimum Z across all frames for both feet)
        # We'll use the lowest point as ground reference
        all_foot_z = np.concatenate([left_foot_world[:, 2], right_foot_world[:, 2]])
        ground_level = np.min(all_foot_z)
        foot_height_above_ground = np.minimum(
            left_foot_world[:, 2] - ground_level,
            right_foot_world[:, 2] - ground_level
        )
        
        print(f"Ground level (min Z): {ground_level:.4f}")
        print(f"Foot height range: [{np.min(foot_height_above_ground):.4f}, {np.max(foot_height_above_ground):.4f}]")
        
        # Detect foot contact: low RELATIVE velocity AND low height
        # Convert velocity threshold from m/s to m/frame
        fps = self.fps
        velocity_threshold_per_frame = fp_config.foot_contact_velocity_threshold / fps
        print(f"Velocity threshold: {fp_config.foot_contact_velocity_threshold:.4f} m/s = {velocity_threshold_per_frame*1000:.2f} mm/frame")
        print(f"Height threshold: {fp_config.foot_contact_min_height:.4f} m")
        print(f"Min relative speed (left): {np.min(left_foot_speed[1:])*1000:.2f} mm/frame, Max: {np.max(left_foot_speed[1:])*1000:.2f} mm/frame")
        print(f"Min relative speed (right): {np.min(right_foot_speed[1:])*1000:.2f} mm/frame, Max: {np.max(right_foot_speed[1:])*1000:.2f} mm/frame")
        
        left_in_contact = (left_foot_speed < velocity_threshold_per_frame) & \
                          (left_foot_world[:, 2] - ground_level < fp_config.foot_contact_min_height)
        right_in_contact = (right_foot_speed < velocity_threshold_per_frame) & \
                           (right_foot_world[:, 2] - ground_level < fp_config.foot_contact_min_height)
        
        # Smooth contact detection using a moving window
        window_size = fp_config.contact_smoothing_window
        if window_size > 1:
            # Use convolution to smooth (1 = contact, 0 = no contact)
            kernel = np.ones(window_size) / window_size
            left_contact_smooth = np.convolve(left_in_contact.astype(float), kernel, mode='same') > 0.5
            right_contact_smooth = np.convolve(right_in_contact.astype(float), kernel, mode='same') > 0.5
            left_in_contact = left_contact_smooth
            right_in_contact = right_contact_smooth
        
        # Debug: print contact stats
        left_contact_frames = np.sum(left_in_contact)
        right_contact_frames = np.sum(right_in_contact)
        print(f"Left foot in contact: {left_contact_frames}/{T} frames ({100*left_contact_frames/T:.1f}%)")
        print(f"Right foot in contact: {right_contact_frames}/{T} frames ({100*right_contact_frames/T:.1f}%)")
        
        # Print per-frame debug for first few frames
        num_debug_frames = min(10, T)
        print(f"\n--- Per-frame debug (first {num_debug_frames} frames) ---")
        for t in range(num_debug_frames):
            root_t = root_trans[t]
            left_world = left_foot_world[t]
            right_world = right_foot_world[t]
            left_vel = left_foot_speed[t]
            right_vel = right_foot_speed[t]
            left_contact = left_in_contact[t]
            right_contact = right_in_contact[t]
            
            left_height = left_world[2] - ground_level
            right_height = right_world[2] - ground_level
            left_world_vel = left_foot_world_speed[t] if t < len(left_foot_world_speed) else 0
            right_world_vel = right_foot_world_speed[t] if t < len(right_foot_world_speed) else 0
            
            print(f"Frame {t}:")
            print(f"  Root: [{root_t[0]:.4f}, {root_t[1]:.4f}, {root_t[2]:.4f}]")
            print(f"  Left foot: world=[{left_world[0]:.4f}, {left_world[1]:.4f}, {left_world[2]:.4f}], "
                  f"rel_vel={left_vel*1000:.2f}mm/frame, world_vel={left_world_vel*1000:.2f}mm/frame, "
                  f"height={left_height*1000:.2f}mm, contact={left_contact}")
            print(f"  Right foot: world=[{right_world[0]:.4f}, {right_world[1]:.4f}, {right_world[2]:.4f}], "
                  f"rel_vel={right_vel*1000:.2f}mm/frame, world_vel={right_world_vel*1000:.2f}mm/frame, "
                  f"height={right_height*1000:.2f}mm, contact={right_contact}")
        
        # Adjust root motion based on foot contact
        # When a foot is planted, we want to keep its world position stable
        # Strategy: compute desired root position from planted foot's movement
        adjusted_root_trans = root_trans.copy()
        
        for t in range(1, T):
            # Check if either foot is in contact
            left_contact = left_in_contact[t]
            right_contact = right_in_contact[t]
            
            if not (left_contact or right_contact):
                continue  # No contact, use original root motion
            
            # Compute desired root adjustment
            # If a foot is planted, its world position should change minimally
            # So if the foot moved in world space, adjust root to compensate
            root_adjustment = np.zeros(3)
            
            # When a foot is planted, we want to lock its world position
            # Strategy: if foot relative position changed, adjust root to compensate
            # so the foot's world position stays stable
            
            if left_contact:
                # Left foot relative position changed - adjust root to compensate
                # The relative change should be minimal when planted, but if it changed,
                # we adjust root in the opposite direction to keep world position stable
                relative_change = left_foot_relative[t] - left_foot_relative[t-1]
                root_adjustment -= relative_change * 0.5  # Partial compensation
            
            if right_contact:
                # Right foot relative position changed - adjust root to compensate
                relative_change = right_foot_relative[t] - right_foot_relative[t-1]
                root_adjustment -= relative_change * 0.5  # Partial compensation
            
            if left_contact and right_contact:
                # Both feet planted - average the adjustment
                root_adjustment *= 0.5
            
            # Apply adjustment with blend factor
            adjusted_root_trans[t] = root_trans[t] + root_adjustment * fp_config.blend_factor
        
        # Smooth the adjusted root motion
        if fp_config.root_smoothing_window > 1:
            window_size = fp_config.root_smoothing_window
            kernel = np.ones(window_size) / window_size
            for dim in range(3):
                smoothed = np.convolve(adjusted_root_trans[:, dim], kernel, mode='same')
                adjusted_root_trans[:, dim] = smoothed
        
        # Blend original and adjusted root motion
        final_root_trans = root_trans * (1 - fp_config.blend_factor) + adjusted_root_trans * fp_config.blend_factor
        
        print(f"\nRoot motion adjustment stats:")
        print(f"  Max adjustment: {np.max(np.abs(final_root_trans - root_trans)):.4f}m")
        print(f"  Avg adjustment: {np.mean(np.abs(final_root_trans - root_trans)):.4f}m")
        print(f"=== END FOOT PLANTING DEBUG ===\n")
        
        # Return adjusted root motion
        return {
            "translation": final_root_trans.tolist(),
            "rotation": root_motion["rotation"]
        }



