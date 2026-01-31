from fbxify.refinement.refinement_config import RefinementConfig
import re
import functools
import math
import numpy as np
import copy
from typing import Dict, Any, Optional
import os
from datetime import datetime

from fbxify.metadata import JOINT_NAMES
from fbxify.i18n import Translator, DEFAULT_LANGUAGE

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
    # Threshold for high change warning (in degrees)
    HIGH_CHANGE_THRESHOLD_DEG = 30.0
    
    def __init__(self, config: RefinementConfig = None, fps: float = 30.0, lang: str = DEFAULT_LANGUAGE):
        if config is None:
            # use default config
            config = RefinementConfig()
        self.lang = lang
        self.configure(config, fps)

    def configure(self, config: RefinementConfig, fps: float = 30.0):
        self.config = config
        self.fps = fps
        self.dt = 1.0 / fps  # time step in seconds
        self.last_refinement_logs = None
    
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

    def _matrix_to_euler_deg(self, rot_t):
        """
        Convert a 3x3 rotation matrix to Euler angles (degrees).
        Uses a standard XYZ convention with a singularity fallback.
        """
        rot = self._parse_rotation_matrix(rot_t)
        if rot is None:
            return None
        r00, r01, r02 = rot[0]
        r10, r11, r12 = rot[1]
        r20, r21, r22 = rot[2]
        sy = math.sqrt(r00 * r00 + r10 * r10)
        singular = sy < 1e-6
        if not singular:
            x = math.atan2(r21, r22)
            y = math.atan2(-r20, sy)
            z = math.atan2(r10, r00)
        else:
            x = math.atan2(-r12, r11)
            y = math.atan2(-r20, sy)
            z = 0.0
        return [rad2deg(x), rad2deg(y), rad2deg(z)]
    
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
    
    def _calculate_rotation_change_per_frame(self, R_original, R_refined):
        """
        Calculate frame-by-frame rotation changes.
        Returns a list of angular changes in degrees for each frame.
        """
        if len(R_original) == 0 or len(R_refined) == 0:
            return []
        
        q_original = [quat_from_R(R) for R in R_original]
        q_refined = [quat_from_R(R) for R in R_refined]
        q_original = fix_quat_hemisphere(q_original)
        q_refined = fix_quat_hemisphere(q_refined)
        
        frame_changes = []
        for t in range(min(len(q_original), len(q_refined))):
            # Calculate relative rotation between original and refined
            q_rel = quat_mul(quat_inv(q_original[t]), q_refined[t])
            angle_deg = rad2deg(quat_angle(q_rel))
            frame_changes.append(angle_deg)
        
        return frame_changes

    def _log_print(self, log_file, msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    def _nan_vector(self, length):
        return [float("nan")] * length

    def _first_non_none(self, series):
        for item in series:
            if item is not None:
                return item
        return None

    def _first_non_none_length(self, series):
        first_item = self._first_non_none(series)
        return len(first_item) if first_item is not None else 0

    def _rebuild_series_from_items(self, refined_items, item_count):
        series_length = len(refined_items[0]) if refined_items and refined_items[0] else 0
        rebuilt_series = []
        for t in range(series_length):
            frame_items = []
            for item_idx in range(item_count):
                frame_items.append(refined_items[item_idx][t])
            rebuilt_series.append(frame_items)
        return rebuilt_series

    def _copy_value(self, value, is_rotation):
        if is_rotation:
            return [[value[k][l] for l in range(3)] for k in range(3)]
        return list(value)

    def _interpolate_value(self, prev_value, next_value, t, is_rotation):
        if is_rotation:
            q1 = quat_from_R(prev_value)
            q2 = quat_from_R(next_value)
            q_interp = slerp(q1, q2, t)
            return R_from_quat(q_interp)
        return [
            prev_value[k] + t * (next_value[k] - prev_value[k])
            for k in range(3)
        ]

    def _fill_range_with_value(self, result, start_idx, end_idx, value, is_rotation):
        for j in range(start_idx, end_idx):
            result[j] = self._copy_value(value, is_rotation)

    def _find_prev_valid_index(self, series, start_idx):
        for j in range(start_idx, -1, -1):
            if series[j] is not None:
                return j
        return None

    def _find_next_valid_index(self, series, start_idx):
        for j in range(start_idx, len(series)):
            if series[j] is not None:
                return j
        return None

    def _extract_series_per_index(self, series, item_count, transform):
        extracted = []
        for item_idx in range(item_count):
            item_series = []
            for t in range(len(series)):
                frame = series[t]
                if frame is not None and item_idx < len(frame):
                    item_series.append(transform(frame[item_idx]))
                else:
                    item_series.append(None)
            extracted.append(item_series)
        return extracted

    def _ensure_3d_vector(self, vector):
        if isinstance(vector, (list, tuple)) and len(vector) >= 3:
            return list(vector[:3])
        return [0.0, 0.0, 0.0]

    def _delta_vector_or_nan(self, orig_value, refined_value):
        if orig_value is None or refined_value is None:
            return self._nan_vector(3)
        return [
            refined_value[0] - orig_value[0],
            refined_value[1] - orig_value[1],
            refined_value[2] - orig_value[2],
        ]

    def _delta_euler_or_nan(self, orig_rot, refined_rot):
        if orig_rot is None or refined_rot is None:
            return self._nan_vector(3)
        orig_euler = self._matrix_to_euler_deg(orig_rot)
        refined_euler = self._matrix_to_euler_deg(refined_rot)
        if orig_euler is None or refined_euler is None:
            return self._nan_vector(3)
        return [
            refined_euler[0] - orig_euler[0],
            refined_euler[1] - orig_euler[1],
            refined_euler[2] - orig_euler[2],
        ]

    def _build_per_joint_deltas(self, series_to_frame_map, orig_series, refined_series, item_count, delta_fn):
        deltas = []
        for t in range(len(series_to_frame_map)):
            orig_frame = orig_series[t] if t < len(orig_series) else None
            refined_frame = refined_series[t] if t < len(refined_series) else None
            if orig_frame is None or refined_frame is None:
                deltas.append([self._nan_vector(3) for _ in range(item_count)])
                continue
            frame_deltas = []
            for j in range(item_count):
                try:
                    orig_item = orig_frame[j]
                    refined_item = refined_frame[j]
                    frame_deltas.append(delta_fn(orig_item, refined_item))
                except Exception:
                    frame_deltas.append(self._nan_vector(3))
            deltas.append(frame_deltas)
        return deltas

    def _build_per_frame_deltas(self, series_to_frame_map, orig_series, refined_series, delta_fn):
        deltas = []
        for t in range(len(series_to_frame_map)):
            orig_value = orig_series[t] if t < len(orig_series) else None
            refined_value = refined_series[t] if t < len(refined_series) else None
            deltas.append(delta_fn(orig_value, refined_value))
        return deltas

    def _has_enabled_features(self) -> bool:
        return (
            self.config.do_spike_fix or
            self.config.do_rotation_smoothing or
            self.config.do_vector_smoothing or
            self.config.do_root_motion_fix or
            self.config.do_foot_planting or
            self.config.do_interpolate_missing_keyframes
        )

    def _log_refinement_start(self, log_print, log_path, estimation_results):
        log_print("\n" + "=" * 80)
        log_print("REFINEMENT PROCESS")
        log_print("=" * 80)
        log_print(f"Log file: {log_path}")
        log_print(f"Frames: {len(sorted([int(k) for k in estimation_results.keys()]))}")
        log_print("Enabled features:")
        log_print(f"  - Spike fix: {self.config.do_spike_fix}")
        log_print(f"  - Rotation smoothing: {self.config.do_rotation_smoothing}")
        log_print(f"  - Vector smoothing: {self.config.do_vector_smoothing}")
        log_print(f"  - Root motion fix: {self.config.do_root_motion_fix}")
        log_print(f"  - Foot planting: {self.config.do_foot_planting}")
        log_print(f"  - Interpolate missing: {self.config.do_interpolate_missing_keyframes}")
        log_print("=" * 80 + "\n")

    def _init_refinement_tracking(self):
        return {
            "profile_changes": {},
            "bone_changes": {},
            "profile_vector_changes": {},
            "bone_vector_changes": {},
            "profile_spike_counts": {},
        }

    def _collect_person_ids_and_frames(self, estimation_results):
        all_person_ids = set()
        for frame_data in estimation_results.values():
            for person_id in frame_data.keys():
                all_person_ids.add(person_id)
        frame_indices = sorted([int(k) for k in estimation_results.keys()])
        return all_person_ids, frame_indices

    def _init_refined_results(self, estimation_results):
        refined_results = {}
        for frame_key, frame_data in estimation_results.items():
            refined_results[frame_key] = {}
            for person_id_str, person_data in frame_data.items():
                refined_results[frame_key][person_id_str] = person_data.copy()
        return refined_results

    def _log_refinement_summary(
        self,
        log_print,
        profile_changes,
        profile_vector_changes,
        profile_spike_counts,
        bone_changes,
    ):
        log_print("\n" + "=" * 80)
        log_print("REFINEMENT SUMMARY")
        log_print("=" * 80)

        if profile_changes or profile_vector_changes or profile_spike_counts:
            log_print("\nGeneral refinement:")

            profile_rotation_averages = {}
            for profile_name, changes in profile_changes.items():
                if changes:
                    profile_rotation_averages[profile_name] = sum(changes) / len(changes)

            profile_vector_averages = {}
            for profile_name, changes in profile_vector_changes.items():
                if changes:
                    profile_vector_averages[profile_name] = sum(changes) / len(changes)

            profile_order = ["ARMS", "LEGS", "HANDS", "FINGERS", "HEAD", "ROOT", "DEFAULT"]
            for profile_name in profile_order:
                rotation_avg = profile_rotation_averages.get(profile_name)
                vector_avg = profile_vector_averages.get(profile_name)
                spike_info = profile_spike_counts.get(profile_name)

                if rotation_avg is not None or vector_avg is not None or spike_info is not None:
                    parts = []
                    if rotation_avg is not None:
                        parts.append(f"Rotation {rotation_avg:.1f}°")
                    if vector_avg is not None:
                        parts.append(f"Vector {vector_avg:.1f}% (avg adjusted)")
                    if spike_info is not None:
                        spike_count, total_frames = spike_info
                        spike_pct = (spike_count / total_frames * 100.0) if total_frames > 0 else 0.0
                        parts.append(f"{spike_count} Spikes ({spike_pct:.1f}%)")
                    log_print(f"{profile_name}: {' | '.join(parts)}")

            high_change_bones = []
            increasing_bones = []

            for bone_name, change_deg in bone_changes.items():
                if change_deg > self.HIGH_CHANGE_THRESHOLD_DEG:
                    high_change_bones.append((bone_name, change_deg))

                bone_profile = self._profile_name_for(bone_name)
                if bone_profile in profile_rotation_averages:
                    profile_avg = profile_rotation_averages[bone_profile]
                    if change_deg > profile_avg * 2.0 and change_deg > 10.0:
                        increasing_bones.append((bone_name, change_deg))

            high_change_bones.sort(key=lambda x: x[1], reverse=True)
            increasing_bones.sort(key=lambda x: x[1], reverse=True)

            if increasing_bones:
                bone_names = [bone[0] for bone in increasing_bones]
                log_print(
                    "\nWarning - The following bones had an increasing average degree modification. "
                    f"Refinement may be too aggressive: {bone_names}"
                )

            if high_change_bones:
                bone_names = [bone[0] for bone in high_change_bones]
                log_print(
                    f"Warning - The following bones had a very high (>{self.HIGH_CHANGE_THRESHOLD_DEG:.0f}°) "
                    f"average modification. Refinement may be too aggressive: {bone_names}"
                )
        else:
            log_print("\nNo rotation changes tracked.")

        log_print("\n" + "=" * 80)
        log_print("REFINEMENT PROCESS - COMPLETE")
        log_print("=" * 80 + "\n")

    def _find_template_person_data(self, estimation_results, frame_indices, person_id):
        for frame_idx in frame_indices:
            frame_key = str(frame_idx)
            frame_data = estimation_results.get(frame_key, {})
            person_data = frame_data.get(str(person_id))
            if person_data is not None:
                return person_data
        return None

    def _append_person_series_entry(self, series, person_data):
        series["joint_rotations"].append(self._convert_to_list(person_data.get("pred_global_rots")))
        series["root_rotations"].append(self._convert_to_list(person_data.get("global_rot")))
        series["root_translations"].append(self._convert_to_list(person_data.get("pred_cam_t")))
        series["joint_coords"].append(self._convert_to_list(person_data.get("pred_joint_coords")))
        series["keypoints_3d"].append(self._convert_to_list(person_data.get("pred_keypoints_3d")))

    def _collect_person_series(self, estimation_results, frame_indices, person_id):
        series = {
            "joint_rotations": [],
            "root_rotations": [],
            "root_translations": [],
            "joint_coords": [],
            "keypoints_3d": [],
        }
        original_frame_indices = set()
        series_to_frame_map = []

        for frame_idx in frame_indices:
            frame_key = str(frame_idx)
            frame_data = estimation_results.get(frame_key, {})
            person_data = frame_data.get(str(person_id))

            if person_data is None:
                if self.config.do_interpolate_missing_keyframes:
                    for key in series:
                        series[key].append(None)
                    series_to_frame_map.append(frame_idx)
                continue

            original_frame_indices.add(frame_idx)
            series_to_frame_map.append(frame_idx)
            self._append_person_series_entry(series, person_data)

        return series, original_frame_indices, series_to_frame_map

    def _clone_original_series(self, series):
        return {
            "joint_rotations": copy.deepcopy(series["joint_rotations"]),
            "root_rotations": copy.deepcopy(series["root_rotations"]),
            "root_translations": copy.deepcopy(series["root_translations"]),
            "joint_coords": copy.deepcopy(series["joint_coords"]),
        }

    def _refine_person_rotations(self, joint_rotations_series, profile_changes, bone_changes, profile_spike_counts):
        if not joint_rotations_series or not any(x is not None for x in joint_rotations_series):
            return joint_rotations_series

        num_joints = self._first_non_none_length(joint_rotations_series)
        if num_joints == 0:
            return joint_rotations_series

        def _parse_or_identity(rot):
            parsed = self._parse_rotation_matrix(rot)
            return parsed if parsed is not None else self._identity_matrix()

        refined_joint_rots = []
        per_joint_series = self._extract_series_per_index(
            joint_rotations_series, num_joints, _parse_or_identity
        )
        for joint_idx, joint_rot_series in enumerate(per_joint_series):
            bone_name = JOINT_NAMES[joint_idx]
            prof = self._profile_for(bone_name)
            profile_name = self._profile_name_for(bone_name)
            refined_joint_rot = self._process_rotation_series(
                joint_rot_series,
                prof,
                bone_name=bone_name,
                profile_name=profile_name,
                profile_changes=profile_changes,
                bone_changes=bone_changes,
                profile_spike_counts=profile_spike_counts,
            )
            refined_joint_rots.append(refined_joint_rot)

        return self._rebuild_series_from_items(refined_joint_rots, num_joints)

    def _refine_base_rotations(self, base_rotations_series, profile_changes, bone_changes, profile_spike_counts):
        if not base_rotations_series or not any(x is not None for x in base_rotations_series):
            return base_rotations_series

        parsed_base_rotations = []
        for rot in base_rotations_series:
            if rot is None:
                parsed_base_rotations.append(None)
            else:
                parsed = self._parse_rotation_matrix(rot)
                parsed_base_rotations.append(parsed if parsed is not None else self._identity_matrix())

        prof = self.config.profiles.get("root", self.config.profiles.get("*"))
        return self._process_rotation_series(
            parsed_base_rotations,
            prof,
            bone_name="root_rotation",
            profile_name="ROOT",
            profile_changes=profile_changes,
            bone_changes=bone_changes,
            profile_spike_counts=profile_spike_counts,
        )

    def _refine_base_translations(self, base_translations_series, profile_vector_changes, bone_vector_changes):
        if not base_translations_series or not any(x is not None for x in base_translations_series):
            return base_translations_series

        prof = self.config.profiles.get("root", self.config.profiles.get("*"))
        return self._process_vector_series(
            base_translations_series,
            prof,
            bone_name="root_translation",
            profile_name="ROOT",
            profile_vector_changes=profile_vector_changes,
            bone_vector_changes=bone_vector_changes,
        )

    def _refine_joint_coords(self, joint_coords_series, profile_vector_changes, bone_vector_changes):
        if not joint_coords_series or not any(x is not None for x in joint_coords_series):
            return joint_coords_series

        num_joints = self._first_non_none_length(joint_coords_series)
        if num_joints == 0:
            return joint_coords_series

        refined_joint_coords = []
        per_joint_series = self._extract_series_per_index(
            joint_coords_series, num_joints, self._ensure_3d_vector
        )
        for joint_idx, joint_coord_series in enumerate(per_joint_series):
            bone_name = JOINT_NAMES[joint_idx] if joint_idx < len(JOINT_NAMES) else f"joint_{joint_idx}"
            prof = self._profile_for(bone_name)
            profile_name = self._profile_name_for(bone_name)
            refined_joint_coord = self._process_vector_series(
                joint_coord_series,
                prof,
                bone_name=bone_name,
                profile_name=profile_name,
                profile_vector_changes=profile_vector_changes,
                bone_vector_changes=bone_vector_changes,
            )
            refined_joint_coords.append(refined_joint_coord)

        return self._rebuild_series_from_items(refined_joint_coords, num_joints)

    def _refine_keypoints(self, keypoints_3d_series, profile_vector_changes, bone_vector_changes):
        if not keypoints_3d_series or not any(x is not None for x in keypoints_3d_series):
            return keypoints_3d_series

        num_keypoints = self._first_non_none_length(keypoints_3d_series)
        if num_keypoints == 0:
            return keypoints_3d_series

        refined_keypoints_3d = []
        per_keypoint_series = self._extract_series_per_index(
            keypoints_3d_series, num_keypoints, self._ensure_3d_vector
        )
        for kp_idx, kp_series in enumerate(per_keypoint_series):
            bone_name = f"keypoint_{kp_idx}"
            prof = self.config.profiles.get("*", self.config.profiles.get("root"))
            profile_name = "DEFAULT"
            refined_kp = self._process_vector_series(
                kp_series,
                prof,
                bone_name=bone_name,
                profile_name=profile_name,
                profile_vector_changes=profile_vector_changes,
                bone_vector_changes=bone_vector_changes,
            )
            refined_keypoints_3d.append(refined_kp)

        return self._rebuild_series_from_items(refined_keypoints_3d, num_keypoints)

    def _apply_base_motion_adjustments(self, base_rotations_series, base_translations_series, joint_coords_series, log_print):
        # FYI - "root" is for user's reference, since MHR has a "root" joint, you'll see "base" refer to the armature obj, while "root" refer to the root joint here
        if self.config.do_root_motion_fix and base_rotations_series and base_translations_series:
            log_print("\n" + "=" * 80)
            log_print("ROOT MOTION STABILIZATION")
            log_print("=" * 80)
            base_motion_dict = {
                "translation": base_translations_series,
                "rotation": base_rotations_series,
            }
            base_motion_dict = self._base_stabilization(base_motion_dict, log_print)
            base_translations_series = base_motion_dict["translation"]
            base_rotations_series = base_motion_dict["rotation"]
            log_print("=" * 80 + "\n")

        if self.config.do_foot_planting and base_rotations_series and base_translations_series and joint_coords_series:
            base_motion_dict = {
                "translation": base_translations_series,
                "rotation": base_rotations_series,
            }
            base_motion_dict, should_plant = self._prepare_base_motion_for_foot_planting(base_motion_dict, log_print)
            if should_plant:
                base_motion_dict = self._foot_planting_adjustment(base_motion_dict, joint_coords_series, log_print)
                base_translations_series = base_motion_dict["translation"]
                base_rotations_series = base_motion_dict["rotation"]

        return base_translations_series, base_rotations_series

    def _build_person_refinement_logs(
        self,
        series_to_frame_map,
        orig_series,
        joint_rotations_series,
        base_rotations_series,
        base_translations_series,
        joint_coords_series,
    ):
        person_logs = {
            "frames": series_to_frame_map,
        }

        num_joints_coords = self._first_non_none_length(orig_series["joint_coords"])
        if num_joints_coords == 0:
            num_joints_coords = self._first_non_none_length(joint_coords_series)
        if num_joints_coords > 0:
            coords_deltas = self._build_per_joint_deltas(
                series_to_frame_map,
                orig_series["joint_coords"],
                joint_coords_series,
                num_joints_coords,
                self._delta_vector_or_nan,
            )
            person_logs["pred_joint_coords"] = coords_deltas

        num_joints_rots = self._first_non_none_length(orig_series["joint_rotations"])
        if num_joints_rots == 0:
            num_joints_rots = self._first_non_none_length(joint_rotations_series)
        if num_joints_rots > 0:
            rot_deltas = self._build_per_joint_deltas(
                series_to_frame_map,
                orig_series["joint_rotations"],
                joint_rotations_series,
                num_joints_rots,
                self._delta_euler_or_nan,
            )
            person_logs["pred_global_rots"] = rot_deltas

        cam_deltas = self._build_per_frame_deltas(
            series_to_frame_map,
            orig_series["root_translations"],
            base_translations_series,
            self._delta_vector_or_nan,
        )
        if cam_deltas:
            person_logs["pred_cam_t"] = cam_deltas

        global_rot_deltas = self._build_per_frame_deltas(
            series_to_frame_map,
            orig_series["root_rotations"],
            base_rotations_series,
            self._delta_euler_or_nan,
        )
        if global_rot_deltas:
            person_logs["global_rot"] = global_rot_deltas

        return person_logs

    def _should_keep_interpolated_frame(self, series_idx, joint_rotations_series, keypoints_3d_series):
        if (
            series_idx < len(joint_rotations_series) and joint_rotations_series[series_idx] is not None and
            series_idx < len(keypoints_3d_series) and keypoints_3d_series[series_idx] is not None
        ):
            return True
        return False

    def _ensure_refined_person_entry(
        self,
        refined_results,
        estimation_results,
        frame_key,
        person_id,
        was_original,
        has_interpolated_data,
        template_person_data,
    ):
        if frame_key not in refined_results:
            refined_results[frame_key] = {}

        if str(person_id) not in refined_results[frame_key]:
            if was_original:
                original_data = estimation_results.get(frame_key, {}).get(str(person_id))
                if original_data:
                    refined_results[frame_key][str(person_id)] = original_data.copy()
                else:
                    return None
            elif has_interpolated_data and template_person_data is not None:
                refined_person_data = template_person_data.copy()
                refined_results[frame_key][str(person_id)] = refined_person_data
            else:
                return None

        return refined_results[frame_key][str(person_id)]

    def _write_refined_person_frames(
        self,
        person_id,
        estimation_results,
        refined_results,
        series_to_frame_map,
        original_frame_indices,
        template_person_data,
        joint_rotations_series,
        base_rotations_series,
        base_translations_series,
        joint_coords_series,
        keypoints_3d_series,
    ):
        for series_idx, frame_idx in enumerate(series_to_frame_map):
            frame_key = str(frame_idx)
            was_original = frame_idx in original_frame_indices

            has_interpolated_data = False
            if self.config.do_interpolate_missing_keyframes and not was_original:
                has_interpolated_data = self._should_keep_interpolated_frame(
                    series_idx, joint_rotations_series, keypoints_3d_series
                )

            if not was_original and not has_interpolated_data:
                continue

            refined_person_data = self._ensure_refined_person_entry(
                refined_results,
                estimation_results,
                frame_key,
                person_id,
                was_original,
                has_interpolated_data,
                template_person_data,
            )
            if refined_person_data is None:
                continue

            if series_idx < len(joint_rotations_series) and joint_rotations_series[series_idx] is not None:
                refined_person_data["pred_global_rots"] = joint_rotations_series[series_idx]

            if series_idx < len(base_rotations_series) and base_rotations_series[series_idx] is not None:
                refined_person_data["global_rot"] = base_rotations_series[series_idx]

            if series_idx < len(base_translations_series) and base_translations_series[series_idx] is not None:
                refined_person_data["pred_cam_t"] = base_translations_series[series_idx]

            if series_idx < len(joint_coords_series) and joint_coords_series[series_idx] is not None:
                refined_person_data["pred_joint_coords"] = joint_coords_series[series_idx]

            if series_idx < len(keypoints_3d_series) and keypoints_3d_series[series_idx] is not None:
                refined_person_data["pred_keypoints_3d"] = keypoints_3d_series[series_idx]

            if (
                refined_person_data.get("pred_joint_coords") is None or
                refined_person_data.get("pred_global_rots") is None or
                refined_person_data.get("pred_keypoints_3d") is None
            ):
                if not was_original:
                    del refined_results[frame_key][str(person_id)]
                    if not refined_results[frame_key]:
                        del refined_results[frame_key]

    def _refine_person(
        self,
        person_id,
        estimation_results,
        frame_indices,
        refined_results,
        profile_changes,
        bone_changes,
        profile_vector_changes,
        bone_vector_changes,
        profile_spike_counts,
        collect_refinement_logs,
        log_print,
    ):
        template_person_data = self._find_template_person_data(estimation_results, frame_indices, person_id)
        series, original_frame_indices, series_to_frame_map = self._collect_person_series(
            estimation_results, frame_indices, person_id
        )
        orig_series = self._clone_original_series(series)

        joint_rotations_series = self._refine_person_rotations(
            series["joint_rotations"], profile_changes, bone_changes, profile_spike_counts
        )
        base_rotations_series = self._refine_base_rotations(
            series["root_rotations"], profile_changes, bone_changes, profile_spike_counts
        )
        base_translations_series = self._refine_base_translations(
            series["root_translations"], profile_vector_changes, bone_vector_changes
        )
        joint_coords_series = self._refine_joint_coords(
            series["joint_coords"], profile_vector_changes, bone_vector_changes
        )
        keypoints_3d_series = self._refine_keypoints(
            series["keypoints_3d"], profile_vector_changes, bone_vector_changes
        )

        base_translations_series, base_rotations_series = self._apply_base_motion_adjustments(
            base_rotations_series,
            base_translations_series,
            joint_coords_series,
            log_print,
        )

        person_logs = None
        if collect_refinement_logs:
            person_logs = self._build_person_refinement_logs(
                series_to_frame_map,
                orig_series,
                joint_rotations_series,
                base_rotations_series,
                base_translations_series,
                joint_coords_series,
            )

        self._write_refined_person_frames(
            person_id,
            estimation_results,
            refined_results,
            series_to_frame_map,
            original_frame_indices,
            template_person_data,
            joint_rotations_series,
            base_rotations_series,
            base_translations_series,
            joint_coords_series,
            keypoints_3d_series,
        )

        return person_logs

    def apply(self, estimation_results: Dict[str, Dict[str, Any]],
             progress_callback: Optional[callable] = None,
             collect_refinement_logs: bool = False) -> Dict[str, Dict[str, Any]]:
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
        self.last_refinement_logs = None
        if self.config is None:
            print("RefinementManager.apply(): config is None, skipping refinement")
            return estimation_results

        has_enabled_features = self._has_enabled_features()

        print("RefinementManager.apply(): config exists, checking enabled features...")
        print(f"  do_spike_fix: {self.config.do_spike_fix}")
        print(f"  do_rotation_smoothing: {self.config.do_rotation_smoothing}")
        print(f"  do_vector_smoothing: {self.config.do_vector_smoothing}")
        print(f"  do_root_motion_fix: {self.config.do_root_motion_fix}")
        print(f"  do_foot_planting: {self.config.do_foot_planting}")
        print(f"  do_interpolate_missing_keyframes: {self.config.do_interpolate_missing_keyframes}")
        print(f"  has_enabled_features: {has_enabled_features}")

        if not has_enabled_features:
            print("RefinementManager.apply(): All refinement features are disabled, skipping refinement")
            return estimation_results

        translator = Translator(self.lang)
        print("RefinementManager.apply(): Proceeding with refinement...")
        if progress_callback:
            progress_callback(0.0, translator.t("progress.applying_refinement"))

        log_file, log_path = self._setup_refinement_log()
        log_print = functools.partial(self._log_print, log_file)
        refinement_logs = {"persons": {}} if collect_refinement_logs else None

        try:
            self._log_refinement_start(log_print, log_path, estimation_results)

            tracking = self._init_refinement_tracking()
            profile_changes = tracking["profile_changes"]
            bone_changes = tracking["bone_changes"]
            profile_vector_changes = tracking["profile_vector_changes"]
            bone_vector_changes = tracking["bone_vector_changes"]
            profile_spike_counts = tracking["profile_spike_counts"]

            all_person_ids, frame_indices = self._collect_person_ids_and_frames(estimation_results)
            if not frame_indices:
                return estimation_results

            refined_results = self._init_refined_results(estimation_results)

            for person_index, person_id in enumerate(all_person_ids):
                if progress_callback:
                    progress = person_index / len(all_person_ids)
                    progress_callback(
                        progress,
                        translator.t(
                            "progress.refining_person",
                            person_index=person_index + 1,
                            total_people=len(all_person_ids),
                        ),
                    )

                person_logs = self._refine_person(
                    person_id,
                    estimation_results,
                    frame_indices,
                    refined_results,
                    profile_changes,
                    bone_changes,
                    profile_vector_changes,
                    bone_vector_changes,
                    profile_spike_counts,
                    collect_refinement_logs,
                    log_print,
                )

                if collect_refinement_logs and person_logs is not None:
                    refinement_logs["persons"][str(person_id)] = person_logs

            if progress_callback:
                progress_callback(1.0, translator.t("progress.refinement_complete"))

            self._log_refinement_summary(
                log_print,
                profile_changes,
                profile_vector_changes,
                profile_spike_counts,
                bone_changes,
            )

            if collect_refinement_logs:
                self.last_refinement_logs = refinement_logs

        finally:
            log_file.close()

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
            if result[i] is not None:
                i += 1
                continue

            start_none = i
            while i < T and result[i] is None:
                i += 1
            end_none = i

            prev_valid_idx = self._find_prev_valid_index(result, start_none - 1)
            next_valid_idx = self._find_next_valid_index(result, end_none)

            if prev_valid_idx is None and next_valid_idx is None:
                continue

            if prev_valid_idx is not None and next_valid_idx is not None:
                prev_value = result[prev_valid_idx]
                next_value = result[next_valid_idx]
                denom = (next_valid_idx - prev_valid_idx)
                for j in range(start_none, end_none):
                    t = (j - prev_valid_idx) / denom
                    result[j] = self._interpolate_value(prev_value, next_value, t, is_rotation)
                continue

            if prev_valid_idx is not None:
                prev_value = result[prev_valid_idx]
                self._fill_range_with_value(result, start_none, end_none, prev_value, is_rotation)
                continue

            next_value = result[next_valid_idx]
            self._fill_range_with_value(result, start_none, end_none, next_value, is_rotation)
        
        return result

    def _find_data_islands(self, series, min_island_size=2):
        """
        Find continuous islands of non-None data in a series.
        Returns list of (start_idx, end_idx, island_data) tuples.
        Islands smaller than min_island_size are excluded.
        """
        islands = []
        T = len(series)
        i = 0
        
        while i < T:
            # Find start of next island
            if series[i] is not None:
                start = i
                # Find end of island
                while i < T and series[i] is not None:
                    i += 1
                end = i - 1
                island_size = end - start + 1
                
                # Only include islands that meet minimum size
                if island_size >= min_island_size:
                    island_data = series[start:end+1]
                    islands.append((start, end, island_data))
            else:
                i += 1
        
        return islands

    def _fill_none_values(self, series, is_rotation=False):
        """
        Fill None values in a series by forward-filling from previous valid frame.
        Returns None if no previous valid frame exists (meaning we can't process).
        - series: List of values (rotations [3][3] or vectors [3]) with possible None entries
        - is_rotation: If True, handles rotation matrices; if False, handles vectors
        - Returns: Series with None values replaced by previous valid values, or None if no valid values exist
        """
        # Make a copy to avoid modifying the original
        result = series[:]
        T = len(result)
        
        # Find first valid value
        first_valid_idx = None
        for i in range(T):
            if result[i] is not None:
                first_valid_idx = i
                break
        
        # If no valid values exist, return None to signal we can't process
        if first_valid_idx is None:
            return None
        
        # Forward-fill: use previous valid value
        last_valid = result[first_valid_idx]
        for i in range(T):
            if result[i] is not None:
                last_valid = result[i]
            else:
                result[i] = self._copy_value(last_valid, is_rotation)
        
        return result

    def _process_vector_series(self, v_series, prof, bone_name=None, profile_name=None, profile_vector_changes=None, bone_vector_changes=None, processing_island=False):
        # CRITICAL: Interpolation must happen FIRST, before any other processing
        if self.config.do_interpolate_missing_keyframes:
            # Interpolate missing frames before processing
            v_series = self._interpolate_missing_frames(v_series, is_rotation=False)
        else:
            # Process islands of continuous data separately
            if any(v is None for v in v_series) and not processing_island:
                # Find islands of continuous data
                islands = self._find_data_islands(v_series, min_island_size=2)
                
                if not islands:
                    # No valid islands found, return original
                    return v_series
                
                # Create result array, preserving None values
                v_refined = [None] * len(v_series)
                
                # Process each island independently
                for start_idx, end_idx, island_data in islands:
                    # Process this island as if it were a complete series
                    island_refined = self._process_vector_series(
                        island_data, prof, bone_name=bone_name, profile_name=profile_name,
                        profile_vector_changes=profile_vector_changes, 
                        bone_vector_changes=bone_vector_changes,
                        processing_island=True  # Prevent recursive island splitting
                    )
                    
                    # Place refined island back into result
                    for i, refined_val in enumerate(island_refined):
                        v_refined[start_idx + i] = refined_val
                
                return v_refined
        
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
        
        # Calculate and track vector adjustment percentage
        if bone_name:
            change_percent = self._calculate_vector_change_percent(v_original, v)
            if change_percent > 0.01:  # Only track if there's meaningful change
                # Track per bone
                if bone_vector_changes is not None:
                    bone_vector_changes[bone_name] = change_percent
                
                # Track per profile
                if profile_vector_changes is not None and profile_name:
                    if profile_name not in profile_vector_changes:
                        profile_vector_changes[profile_name] = []
                    profile_vector_changes[profile_name].append(change_percent)

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

    def _process_rotation_series(self, R_series, prof, bone_name=None, profile_name=None, profile_changes=None, bone_changes=None, profile_spike_counts=None, processing_island=False):
        # R_series: [T][3][3]
        
        # CRITICAL: Interpolation must happen FIRST, before any other processing
        if self.config.do_interpolate_missing_keyframes:
            # Interpolate missing frames before processing
            R_series = self._interpolate_missing_frames(R_series, is_rotation=True)
        else:
            # Process islands of continuous data separately
            if any(R is None for R in R_series) and not processing_island:
                # Find islands of continuous data
                islands = self._find_data_islands(R_series, min_island_size=2)
                
                if not islands:
                    # No valid islands found, return original
                    return R_series
                
                # Create result array, preserving None values
                R_refined = [None] * len(R_series)
                
                # Process each island independently
                for start_idx, end_idx, island_data in islands:
                    # Process this island as if it were a complete series
                    island_refined = self._process_rotation_series(
                        island_data, prof, bone_name=bone_name, profile_name=profile_name,
                        profile_changes=profile_changes, bone_changes=bone_changes,
                        profile_spike_counts=profile_spike_counts,
                        processing_island=True  # Prevent recursive island splitting
                    )
                    
                    # Place refined island back into result
                    for i, refined_val in enumerate(island_refined):
                        R_refined[start_idx + i] = refined_val
                
                return R_refined
        
        R_original = [[[R[i][j] for j in range(3)] for i in range(3)] for R in R_series]  # Deep copy
        q = [quat_from_R(R) for R in R_series]   # [T] quats
        q = fix_quat_hemisphere(q)

        spike_count = 0
        total_frames = len(q)
        if self.config.do_spike_fix:
            q, spike_count, total_frames = self._despike_rotation(q, prof)
        else:
            # Still count spikes even if spike_fix is disabled, for statistics
            _, spike_count, total_frames = self._despike_rotation(q, prof)
        
        # Track spike statistics per profile
        if profile_spike_counts is not None and profile_name:
            if profile_name not in profile_spike_counts:
                profile_spike_counts[profile_name] = [0, 0]  # [spike_count, total_frames]
            profile_spike_counts[profile_name][0] += spike_count
            profile_spike_counts[profile_name][1] += total_frames

        if self.config.do_rotation_smoothing:
            q = self._smooth_rotation(q, prof)

        # back to matrices
        R_refined = [R_from_quat(qt) for qt in q]
        
        # Calculate and track changes
        if bone_name:
            change_deg = self._calculate_rotation_change_percent(R_original, R_refined)
            if change_deg > 0.1:  # Only track if there's meaningful change (>0.1 degrees)
                # Track per bone
                if bone_changes is not None:
                    bone_changes[bone_name] = change_deg
                
                # Track per profile
                if profile_changes is not None and profile_name:
                    if profile_name not in profile_changes:
                        profile_changes[profile_name] = []
                    profile_changes[profile_name].append(change_deg)
        
        return R_refined

    def _despike_rotation(self, q, prof):
        T = len(q)
        if T < 3:
            return q, 0, T

        ang_vel = [0.0] * T
        ang_acc = [0.0] * T

        for t in range(1, T):
            dq = quat_mul(quat_inv(q[t-1]), q[t])
            angle_deg = rad2deg(quat_angle(dq))          # shortest angle
            ang_vel[t] = angle_deg / self.dt

        for t in range(2, T):
            ang_acc[t] = (ang_vel[t] - ang_vel[t-1]) / self.dt

        spike_count = 0
        for t in range(1, T-1):
            is_spike = (ang_vel[t] > prof.max_ang_speed_deg and
                        ang_acc[t] > prof.max_ang_accel_deg)

            if is_spike:
                spike_count += 1
                # replace with slerp neighbor midpoint
                q[t] = slerp(q[t-1], q[t+1], 0.5)

        return q, spike_count, T

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
            matches = self._match(pattern, bone_name)
            if matches:
                return prof
        return self.config.profiles["*"]
    
    def _profile_name_for(self, bone_name):
        """Get the profile name (pattern) that matches a bone name."""
        for pattern, prof in self.config.profiles.items():
            matches = self._match(pattern, bone_name)
            if matches:
                # Map pattern to readable name
                if pattern == "*arm*" or pattern == "*clavicle*":
                    return "ARMS"
                elif pattern == "*leg*":
                    return "LEGS"
                elif pattern == "*hand*" or pattern == "*wrist*":
                    return "HANDS"
                elif pattern == "*finger*":
                    return "FINGERS"
                elif pattern == "*head*":
                    return "HEAD"
                elif pattern == "root":
                    return "ROOT"
                else:
                    return "DEFAULT"
        return "DEFAULT"

    def _match(self, pattern, bone_name):
        bone_name = bone_name.lower()
        regex = re.compile(pattern.replace("*", ".*").lower())
        # Use search() instead of match() to allow matching anywhere in the string
        return regex.search(bone_name) is not None

    def _prepare_base_translation(self, trans, log_print):
        if trans is None:
            log_print("WARNING: root_motion['translation'] is None, skipping stabilization")
            return None
        if self.config.do_interpolate_missing_keyframes:
            trans = self._interpolate_missing_frames(trans, is_rotation=False)
        elif any(t is None for t in trans):
            log_print(
                "WARNING: Missing keyframes detected in root translation, skipping stabilization "
                "(enable 'Interpolate Missing Keyframes' to interpolate)"
            )
            return None
        if any(t is None for t in trans):
            log_print(
                "WARNING: None values still present in root translation after interpolation check, "
                "skipping stabilization"
            )
            return None
        return trans

    def _filter_base_translation(self, trans, prof):
        T = len(trans)
        if T == 0:
            return trans

        if prof.method == "one_euro":
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
            return filtered_trans

        if prof.method == "ema":
            x_series = [trans[t][0] for t in range(T)]
            y_series = [trans[t][1] for t in range(T)]
            z_series = [trans[t][2] for t in range(T)]

            alpha_xy = 1.0 - math.exp(-2.0 * math.pi * prof.root_cutoff_xy_hz * self.dt)
            alpha_z = 1.0 - math.exp(-2.0 * math.pi * prof.root_cutoff_z_hz * self.dt)

            filtered_x = [x_series[0]]
            filtered_y = [y_series[0]]
            filtered_z = [z_series[0]]

            for t in range(1, T):
                filtered_x.append(filtered_x[t-1] + alpha_xy * (x_series[t] - filtered_x[t-1]))
                filtered_y.append(filtered_y[t-1] + alpha_xy * (y_series[t] - filtered_y[t-1]))
                filtered_z.append(filtered_z[t-1] + alpha_z * (z_series[t] - filtered_z[t-1]))

            filtered_trans = []
            for t in range(T):
                filtered_trans.append([
                    filtered_x[t],
                    filtered_y[t],
                    filtered_z[t]
                ])
            return filtered_trans

        return self._process_vector_series(trans, prof, processing_island=False)

    def _stabilize_base_translation(self, trans, prof, log_print):
        prepared = self._prepare_base_translation(trans, log_print)
        if prepared is None:
            return None, None
        trans_original = [[t[i] for i in range(3)] for t in prepared]
        filtered = self._filter_base_translation(prepared, prof)
        return filtered, trans_original

    def _stabilize_base_rotation(self, rot, prof, log_print):
        if rot is None:
            log_print("WARNING: root_motion['rotation'] is None, skipping stabilization")
            return None, None
        rot_original = self._deep_copy_rotation_series(rot)
        stabilized = self._process_rotation_series(rot, prof, bone_name=None)
        return stabilized, rot_original

    def _prepare_base_motion_for_foot_planting(self, root_motion, log_print):
        # Handle None values: interpolate if enabled, otherwise forward-fill.
        if self.config.do_interpolate_missing_keyframes:
            if any(t is None for t in root_motion["translation"]):
                root_motion["translation"] = self._interpolate_missing_frames(
                    root_motion["translation"],
                    is_rotation=False,
                )
            if "rotation" in root_motion and root_motion["rotation"] is not None:
                if any(r is None for r in root_motion["rotation"]):
                    root_motion["rotation"] = self._interpolate_missing_frames(
                        root_motion["rotation"],
                        is_rotation=True,
                    )
            if any(t is None for t in root_motion["translation"]):
                return root_motion, False
            if "rotation" in root_motion and root_motion["rotation"] is not None:
                if any(r is None for r in root_motion["rotation"]):
                    return root_motion, False
            return root_motion, True

        filled_trans = self._fill_none_values(root_motion["translation"], is_rotation=False)
        if filled_trans is None:
            log_print("WARNING: No valid translation frames found, skipping foot planting")
            return root_motion, False
        root_motion["translation"] = filled_trans

        if "rotation" in root_motion and root_motion["rotation"] is not None:
            filled_rot = self._fill_none_values(root_motion["rotation"], is_rotation=True)
            if filled_rot is None:
                log_print("WARNING: No valid rotation frames found, skipping foot planting")
                return root_motion, False
            root_motion["rotation"] = filled_rot

        return root_motion, True

    def _base_stabilization(self, root_motion, log_print):
        """
        Stabilize root motion to reduce jitter and unwanted movement.
        root_motion: dict with keys like "translation" [T][3] and "rotation" [T][3][3]
        Returns: stabilized root_motion dict
        """
        if root_motion is None:
            return root_motion
        
        prof = self.config.profiles.get("root", self.config.profiles["*"])
        stabilized = {}
        
        if "translation" in root_motion:
            filtered_trans, trans_original = self._stabilize_base_translation(
                root_motion["translation"],
                prof,
                log_print
            )
            if filtered_trans is not None:
                stabilized["translation"] = filtered_trans
                if not any(t is None for t in stabilized["translation"]):
                    change_percent = self._calculate_vector_change_percent(trans_original, stabilized["translation"])
                    if change_percent > 0.01:
                        log_print(f"Root translation adjusted by {change_percent:.2f}%")
    

        if "rotation" in root_motion:
            stabilized_rot, rot_original = self._stabilize_base_rotation(
                root_motion["rotation"],
                prof,
                log_print
            )
            if stabilized_rot is not None:
                stabilized["rotation"] = stabilized_rot
                if not any(r is None for r in stabilized["rotation"]):
                    change_deg = self._calculate_rotation_change_percent(rot_original, stabilized["rotation"])
                    if change_deg > 0.1:
                        log_print(f"Root rotation adjusted by {change_deg:.2f}° (avg)")
        
        # Copy any other fields
        for key in root_motion:
            if key not in stabilized:
                stabilized[key] = root_motion[key]
        
        return stabilized
    
    def _setup_refinement_log(self):
        """Setup logging for refinement debug output."""
        logs_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_filename = f"refinement_{timestamp}.log"
        log_path = os.path.join(logs_dir, log_filename)
        log_file = open(log_path, 'w', encoding='utf-8')
        return log_file, log_path
    
    def _validate_foot_planting_inputs(self, root_motion, joint_coords_series, log_print):
        """Validate inputs for foot planting."""
        if root_motion is None or "translation" not in root_motion:
            log_print("ERROR: root_motion is None or missing 'translation'")
            return False
        if joint_coords_series is None or len(joint_coords_series) == 0:
            log_print("ERROR: joint_coords_series is None or empty")
            return False
        T = len(root_motion["translation"])
        if T == 0:
            log_print("ERROR: T == 0")
            return False
        return True
    
    def _get_foot_contact_indices(self, fp_config, log_print):
        """Get joint indices for root and foot contact points."""
        if fp_config.use_mid_foot:
            l_contact_idx = JOINT_NAMES.index("l_ball")
            r_contact_idx = JOINT_NAMES.index("r_ball")
            log_print(f"Using MID_FOOT: l_ball={l_contact_idx}, r_ball={r_contact_idx}")
        else:
            l_contact_idx = JOINT_NAMES.index("l_foot")
            r_contact_idx = JOINT_NAMES.index("r_foot")
            log_print(f"Using FOOT: l_foot={l_contact_idx}, r_foot={r_contact_idx}")
        return l_contact_idx, r_contact_idx
    
    def _extract_joint_positions(self, joint_coords_series, T, l_contact_idx, r_contact_idx):
        """Extract root and foot positions from joint coordinates series."""
        left_foot_joint_space = []
        right_foot_joint_space = []
                
        for t in range(T):
            if t < len(joint_coords_series) and joint_coords_series[t] is not None:
                # Check and extract left foot position
                if l_contact_idx < len(joint_coords_series[t]) and joint_coords_series[t][l_contact_idx] is not None:
                    left_pos = list(joint_coords_series[t][l_contact_idx][:3])
                else:
                    left_pos = [0.0, 0.0, 0.0]
                
                # Check and extract right foot position
                if r_contact_idx < len(joint_coords_series[t]) and joint_coords_series[t][r_contact_idx] is not None:
                    right_pos = list(joint_coords_series[t][r_contact_idx][:3])
                else:
                    right_pos = [0.0, 0.0, 0.0]
                
                left_foot_joint_space.append(left_pos)
                right_foot_joint_space.append(right_pos)
            else:
                left_foot_joint_space.append([0.0, 0.0, 0.0])
                right_foot_joint_space.append([0.0, 0.0, 0.0])
        
        return (
            np.array(left_foot_joint_space),
            np.array(right_foot_joint_space)
        )
    
    def _calculate_foot_velocities(self, foot_offset, T, dt, log_print):
        """Calculate foot offset velocities relative to base."""
        foot_offset_velocity = np.zeros((T, 3))
        for t in range(1, T):
            foot_offset_velocity[t] = (foot_offset[t] - foot_offset[t-1]) / dt
        foot_offset_speed = np.linalg.norm(foot_offset_velocity, axis=1)
        return foot_offset_velocity, foot_offset_speed
    
    def _calculate_foot_height(self, foot_position, height_direction):
        """Calculate foot height by projecting position onto height direction vector.
        
        Args:
            foot_position: [3] or [T, 3] array of foot positions
            height_direction: [3] normalized direction vector for "up"
        
        Returns:
            Height value(s) - positive means higher, negative means lower
        """
        height_dir = np.array(height_direction)
        if foot_position.ndim == 1:
            return np.dot(foot_position, height_dir)
        return np.dot(foot_position, height_dir)

    def _calculate_ground_height_reference(self, base_trans, height_direction, log_print):
        """Compute per-frame ground height reference using root motion auto-flooring."""
        height_dir = np.array(height_direction)
        base_height = np.dot(base_trans, height_dir)
        valid_mask = np.isfinite(base_height)
        if np.any(valid_mask):
            avg_base_height = float(np.mean(base_height[valid_mask]))
        else:
            avg_base_height = 0.0
        auto_floor_offset = -avg_base_height
        ground_height = base_height + auto_floor_offset
        log_print(
            f"Auto-floor (refinement): avg_base_h={avg_base_height:.4f}, "
            f"auto_floor_offset={auto_floor_offset:.4f}"
        )
        return ground_height

    def _log_foot_height_debug(self, foot_label, foot_height, foot_offset_speed,
                               ground_height, height_threshold, velocity_threshold, log_print):
        """Log detailed foot height vs ground diagnostics for debugging."""
        height_delta = foot_height - (ground_height + height_threshold)
        height_check = height_delta < 0.0
        velocity_check = foot_offset_speed < velocity_threshold
        in_contact = height_check & velocity_check

        # Highlight when the foot was below the ground threshold, but maybe was too fast
        below_indices = np.where(height_check)[0]
        total_frames = len(foot_height)

        log_print(f"{foot_label} foot height debug:")

        above_deltas = height_delta[below_indices]
        if above_deltas.size:
            log_print(
                f"  Above-ground delta (ground - foot): min={np.min(above_deltas):.4f}m, "
                f"max={np.max(above_deltas):.4f}m, avg={np.mean(above_deltas):.4f}m"
            )
        else:
            log_print("  Above-ground delta (ground - foot): no below-threshold frames")

        for t in range(total_frames):
            log_print(
                f"  t={t:04d} foot_h={foot_height[t]:.4f} "
                f"ground_h={ground_height[t]:.4f} "
                f"delta={height_delta[t]:.4f} "
                f"speed={foot_offset_speed[t]:.4f} h_thr={height_threshold:.4f} v_thr={velocity_threshold:.4f} "
                f"height_ok={height_check[t]} vel_ok={velocity_check[t]} "
                f"in_contact={in_contact[t]}"
            )
    
    def _detect_foot_contact(self, foot_height, foot_offset_speed, ground_height, height_threshold, velocity_threshold):
        """Detect foot contact based on height and velocity thresholds.
        
        Args:
            foot_height: Height value(s) from _calculate_foot_height (positive = higher)
            foot_offset_speed: Foot velocity relative to root
            ground_height: Per-frame ground height reference
            height_threshold: Maximum height above ground to count as contact
            velocity_threshold: Maximum velocity for contact
        
        Returns:
            Boolean array indicating contact
        """
        # Find whether the foot_height is above or below the ground (plus some foot threshold to solidify 'lifted' foot)
        height_delta = foot_height - (ground_height + height_threshold)
        height_check = height_delta < 0.0
        velocity_check = foot_offset_speed < velocity_threshold
        return height_check & velocity_check
    
    def _smooth_contact_detection(self, contact_array, window_size):
        """Smooth contact detection using moving average."""
        if window_size <= 1:
            return contact_array
        kernel = np.ones(window_size) / window_size
        # Use edge padding instead of zero padding to prevent boundary artifacts
        pad_width = window_size // 2
        padded = np.pad(contact_array.astype(float), pad_width, mode='edge')
        convolved = np.convolve(padded, kernel, mode='valid')
        smoothed = convolved > 0.5
        return smoothed
    
    def _calculate_contact_confidence(self, in_contact, foot_offset_speed, velocity_threshold, T):
        """Calculate contact confidence based on velocity."""
        confidence = np.zeros(T)
        for t in range(T):
            if in_contact[t]:
                vel_score = 1.0 - min(foot_offset_speed[t] / max(velocity_threshold * 2, 1e-6), 1.0)
                confidence[t] = vel_score
        return confidence
    
    def _lock_foot_on_contact_start(self, foot_contact, was_in_contact, curr_root, root_rot, foot_offset_world, t, log_print, foot_name):
        """Lock foot position when contact starts."""
        if not (foot_contact and not was_in_contact):
            return None
        foot_world = curr_root + root_rot[t] @ foot_offset_world[t]
        # Per-frame logging removed - summary will be printed at the end
        return foot_world.copy()
    
    def _unlock_foot_on_contact_end(self, foot_contact, locked_pos, t, log_print, foot_name):
        """Unlock foot position when contact ends."""
        if foot_contact:
            return locked_pos
        # Per-frame logging removed - summary will be printed at the end
        return None
    
    def _select_support_foot(self, left_locked, right_locked, left_conf, right_conf, t, log_print):
        """Select which foot to use as support when both are locked."""
        if not (left_locked is not None and right_locked is not None):
            return (left_locked is not None, right_locked is not None)
        # Per-frame logging removed - summary will be printed at the end
        if left_conf > right_conf:
            return (True, False)
        return (False, True)
    
    def _adjust_base_for_locked_foot(self, curr_root, locked_foot_pos, root_rot, foot_offset_world, blend_factor, t, log_print, foot_name):
        """Adjust root position to maintain locked foot position."""
        desired_root = locked_foot_pos - root_rot[t] @ foot_offset_world[t]
        root_adjustment = desired_root - curr_root
        adjusted_root = curr_root + root_adjustment * blend_factor
        # Per-frame logging removed - summary will be printed at the end
        return adjusted_root
    
    def _smooth_base_motion(self, base_trans, window_size):
        """Smooth root motion using moving average."""
        if window_size <= 1:
            return base_trans
        smoothed = base_trans.copy()
        kernel = np.ones(window_size) / window_size
        for dim in range(3):
            # Use edge padding instead of zero padding to prevent drift to 0,0,0 at boundaries
            # This ensures the last frames maintain their position instead of being averaged with zeros
            pad_width = window_size // 2
            padded = np.pad(base_trans[:, dim], pad_width, mode='edge')
            convolved = np.convolve(padded, kernel, mode='valid')
            smoothed[:, dim] = convolved
        return smoothed
    
    def _foot_planting_adjustment(self, base_motion, joint_coords_series, log_print):
        """
        Adjust root motion based on foot contact to reduce jitter.
        
        Coordinate systems:
        - Joint space: negative Y is "up", coordinates are absolute relative to body_world at [0,0,0]
        - Camera space (base_motion): positive Y is "up"
        - Conversion: camera = [joint[0], -joint[1], -joint[2]]
        
        Args:
            base_motion: dict with "translation" [T][3] (camera space) and "rotation" [T][3][3]
            joint_coords_series: [T][num_joints][3] - joint coordinates per frame (absolute in joint space, relative to body_world)
        
        Returns:
            Adjusted base_motion dict with corrected translation
        """
        if not self.config.do_foot_planting:
            return base_motion
        
        log_print("\n" + "="*80)
        log_print("FOOT PLANTING")
        log_print("="*80)
        
        if not self._validate_foot_planting_inputs(base_motion, joint_coords_series, log_print):
            return base_motion
        
        fp_config = self.config.foot_planting_config
        T = len(base_motion["translation"])

        print(f"T: {T}")
        
        try:
            l_contact_idx, r_contact_idx = self._get_foot_contact_indices(fp_config, log_print)
        except ValueError as e:
            log_print(f"ERROR: Joint names not found: {e}")
            return base_motion
        
        base_trans = np.array(base_motion["translation"])
        base_rot = np.array(base_motion["rotation"])
        
        left_foot_joint_space, right_foot_joint_space = self._extract_joint_positions(
            joint_coords_series, T, l_contact_idx, r_contact_idx
        )

        left_foot_world_space = left_foot_joint_space + base_trans
        right_foot_world_space = right_foot_joint_space + base_trans
        
        # Calculate velocity from joint space (pred_coords delta) to avoid jitter from pred_cam_t
        # since we're tryping to assist base jitter, we should use the joint space velocity instead of the camera space velocity
        _, left_foot_offset_speed = self._calculate_foot_velocities(left_foot_joint_space, T, self.dt, log_print)
        _, right_foot_offset_speed = self._calculate_foot_velocities(right_foot_joint_space, T, self.dt, log_print)
                
        velocity_threshold = fp_config.foot_contact_velocity_threshold
        height_threshold = fp_config.foot_contact_min_height
        height_direction = np.array(fp_config.height_direction)
        ground_height = self._calculate_ground_height_reference(base_trans, height_direction, log_print)
        
        
        left_foot_height = self._calculate_foot_height(left_foot_joint_space, height_direction)
        right_foot_height = self._calculate_foot_height(right_foot_joint_space, height_direction)

        # Extensive debug logs for foot height vs ground checks
        self._log_foot_height_debug(
            "LEFT",
            left_foot_height,
            left_foot_offset_speed,
            ground_height,
            height_threshold,
            velocity_threshold,
            log_print
        )
        self._log_foot_height_debug(
            "RIGHT",
            right_foot_height,
            right_foot_offset_speed,
            ground_height,
            height_threshold,
            velocity_threshold,
            log_print
        )
        
        left_in_contact = self._detect_foot_contact(
            left_foot_height,
            left_foot_offset_speed,
            ground_height,
            height_threshold,
            velocity_threshold
        )
        right_in_contact = self._detect_foot_contact(
            right_foot_height,
            right_foot_offset_speed,
            ground_height,
            height_threshold,
            velocity_threshold
        )
        
        left_in_contact = self._smooth_contact_detection(left_in_contact, fp_config.contact_smoothing_window)
        right_in_contact = self._smooth_contact_detection(right_in_contact, fp_config.contact_smoothing_window)
        
        left_contact_confidence = self._calculate_contact_confidence(left_in_contact, left_foot_offset_speed, velocity_threshold, T)
        right_contact_confidence = self._calculate_contact_confidence(right_in_contact, right_foot_offset_speed, velocity_threshold, T)
        
        # Log contact summary
        log_print(f"Contact summary: Left {np.sum(left_in_contact)}/{T} frames ({100.0*np.sum(left_in_contact)/T:.1f}%), "
                  f"Right {np.sum(right_in_contact)}/{T} frames ({100.0*np.sum(right_in_contact)/T:.1f}%)")
        
        adjusted_base_trans, foot_planting_stats = self._adjust_base_for_foot_planting(
            base_trans, base_rot, left_in_contact, right_in_contact,
            left_foot_world_space, right_foot_world_space,
            left_contact_confidence, right_contact_confidence,
            fp_config.blend_factor, T, log_print
        )
        
        # Print foot planting summary
        frames_with_contact = foot_planting_stats['frames_with_contact']
        frames_without_contact = foot_planting_stats['frames_without_contact']
        total_frames = frames_with_contact + frames_without_contact
        contact_percentage = (frames_with_contact / total_frames * 100.0) if total_frames > 0 else 0.0
        no_contact_percentage = (frames_without_contact / total_frames * 100.0) if total_frames > 0 else 0.0
        
        log_print(f"\nFoot planting summary:")
        log_print(f"  Frames with feet planted: {frames_with_contact}/{total_frames} ({contact_percentage:.1f}%)")
        log_print(f"  Frames without feet planted: {frames_without_contact}/{total_frames} ({no_contact_percentage:.1f}%)")
        
        adjusted_base_trans = self._smooth_base_motion(adjusted_base_trans, fp_config.root_smoothing_window)
        
        max_adjustment = np.max(np.abs(adjusted_base_trans - base_trans))
        avg_adjustment = np.mean(np.abs(adjusted_base_trans - base_trans))
        log_print(f"\nRoot adjustment: max={max_adjustment:.4f}m, avg={avg_adjustment:.4f}m")
        log_print("="*80 + "\n")
        
        return {
            "translation": adjusted_base_trans.tolist(),
            "rotation": base_motion["rotation"]
        }
    
    def _adjust_base_for_foot_planting(self, base_trans, base_rot, left_in_contact, right_in_contact,
                                     left_foot_offset_world, right_foot_offset_world,
                                     left_contact_confidence, right_contact_confidence,
                                     blend_factor, T, log_print):
        """Adjust root motion based on locked foot positions."""
        adjusted_base_trans = base_trans.copy()
        left_foot_locked_pos = None
        right_foot_locked_pos = None
        
        # Track statistics
        frames_with_contact = 0
        frames_without_contact = 0
        
        for t in range(T):
            curr_root = self._get_current_base_position(adjusted_base_trans, base_trans, t)
            left_contact = left_in_contact[t]
            right_contact = right_in_contact[t]
            left_was_in_contact = left_in_contact[t-1] if t > 0 else False
            right_was_in_contact = right_in_contact[t-1] if t > 0 else False
            
            new_left_locked = self._lock_foot_on_contact_start(
                left_contact, left_was_in_contact, curr_root, base_rot, left_foot_offset_world, t, log_print, "LEFT"
            )
            if new_left_locked is not None:
                left_foot_locked_pos = new_left_locked
            
            new_right_locked = self._lock_foot_on_contact_start(
                right_contact, right_was_in_contact, curr_root, base_rot, right_foot_offset_world, t, log_print, "RIGHT"
            )
            if new_right_locked is not None:
                right_foot_locked_pos = new_right_locked
            
            left_foot_locked_pos = self._unlock_foot_on_contact_end(left_contact, left_foot_locked_pos, t, log_print, "LEFT")
            right_foot_locked_pos = self._unlock_foot_on_contact_end(right_contact, right_foot_locked_pos, t, log_print, "RIGHT")
            
            # Track if at least one foot is planted
            has_contact = (left_foot_locked_pos is not None) or (right_foot_locked_pos is not None)
            if has_contact:
                frames_with_contact += 1
            else:
                frames_without_contact += 1
            
            adjusted_base_trans[t] = self._apply_base_adjustment_for_frame(
                curr_root, left_foot_locked_pos, right_foot_locked_pos,
                base_rot, left_foot_offset_world, right_foot_offset_world,
                left_contact_confidence, right_contact_confidence,
                blend_factor, t, log_print
            )
        
        stats = {
            'frames_with_contact': frames_with_contact,
            'frames_without_contact': frames_without_contact
        }
        return adjusted_base_trans, stats
    
    def _get_current_base_position(self, adjusted_base_trans, root_trans, t):
        """Get current root position for frame t."""
        if t == 0:
            return root_trans[t].copy()
        curr_root = adjusted_base_trans[t-1].copy()
        return curr_root + (root_trans[t] - root_trans[t-1])
    
    def _apply_base_adjustment_for_frame(self, curr_root, left_foot_locked_pos, right_foot_locked_pos,
                                        root_rot, left_foot_offset_world, right_foot_offset_world,
                                        left_contact_confidence, right_contact_confidence,
                                        blend_factor, t, log_print):
        """Apply root adjustment for a single frame."""
        if left_foot_locked_pos is None and right_foot_locked_pos is None:
            # Per-frame logging removed - summary will be printed at the end
            return curr_root
        
        use_left, use_right = self._select_support_foot(
            left_foot_locked_pos, right_foot_locked_pos,
            left_contact_confidence[t], right_contact_confidence[t], t, log_print
        )
        
        if use_left:
            adjusted = self._adjust_base_for_locked_foot(
                curr_root, left_foot_locked_pos, root_rot, left_foot_offset_world, blend_factor, t, log_print, "LEFT"
            )
            return adjusted
        
        if use_right:
            adjusted = self._adjust_base_for_locked_foot(
                curr_root, right_foot_locked_pos, root_rot, right_foot_offset_world, blend_factor, t, log_print, "RIGHT"
            )
            return adjusted
        
        return curr_root
    
