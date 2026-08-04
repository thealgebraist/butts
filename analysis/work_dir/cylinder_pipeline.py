import cv2
import sys
import subprocess
import numpy as np
import struct
import math

def generate_cylinder():
    points = []
    R = 1.0
    H = 4.0
    segments = 36
    for i in range(segments):
        theta = i * 2.0 * math.pi / segments
        points.append([R * math.cos(theta), -H/2.0, R * math.sin(theta)])
        points.append([R * math.cos(theta), H/2.0, R * math.sin(theta)])
    for i in range(4):
        theta = i * math.pi / 2.0
        for j in range(1, 10):
            y = -H/2.0 + j * (H / 10.0)
            points.append([R * math.cos(theta), y, R * math.sin(theta)])
    return np.array(points, dtype=np.float32)

def project_points(points, pitch, yaw, roll, scale, W, H):
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)
    
    R = np.array([
        [cy*cr, cy*sr, -sy],
        [sp*sy*cr - cp*sr, sp*sy*sr + cp*cr, sp*cy],
        [cp*sy*cr + sp*sr, cp*sy*sr - sp*cr, cp*cy]
    ])
    
    f = 500.0 * scale
    Z_offset = 10.0
    
    rotated = points.dot(R.T)
    rotated[:, 2] += Z_offset
    
    # Filter points behind camera
    valid = rotated[:, 2] > 0.1
    rotated = rotated[valid]
    
    px = (f * rotated[:, 0] / rotated[:, 2] + W / 2).astype(np.int32)
    py = (f * rotated[:, 1] / rotated[:, 2] + H / 2).astype(np.int32)
    
    pts2d = np.column_stack((px, py))
    return pts2d

def process_video(input_video, output_video):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error opening video")
        sys.exit(1)
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0
    
    W, H = 512, 910
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (W, H))
    
    cpp_proc = subprocess.Popen(['./cylinder_optimizer'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=False)
    
    # Initial state tracking parameters
    pitch, yaw, roll, scale = 0.0, 0.0, 0.0, 2.0
    base_cylinder = generate_cylinder()
    
    count = 0
    header_format = 'ffff'
    header_size = struct.calcsize(header_format)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        small = cv2.resize(frame, (W, H))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        
        # Send header (Visual IMU initial state)
        cpp_proc.stdin.write(struct.pack(header_format, pitch, yaw, roll, scale))
        # Send frame
        cpp_proc.stdin.write(gray.tobytes())
        cpp_proc.stdin.flush()
        
        # Receive optimized parameters
        data = cpp_proc.stdout.read(header_size)
        if len(data) < header_size:
            break
            
        pitch, yaw, roll, scale = struct.unpack(header_format, data)
        
        # Project optimized cylinder points
        pts2d = project_points(base_cylinder, pitch, yaw, roll, scale, W, H)
        
        # Mask intersection rendering
        if len(pts2d) > 0:
            hull = cv2.convexHull(pts2d)
            mask = np.zeros((H, W), dtype=np.uint8)
            cv2.fillConvexPoly(mask, hull, 255)
            
            # Intersection with original frame
            masked_frame = cv2.bitwise_and(small, small, mask=mask)
        else:
            masked_frame = np.zeros_like(small)
        
        out.write(masked_frame)
        count += 1
        if count % 100 == 0:
            print(f"Processed {count} frames (pitch={pitch:.2f}, yaw={yaw:.2f}, roll={roll:.2f}, scale={scale:.2f})")
            
    cpp_proc.stdin.close()
    cpp_proc.wait()
    cap.release()
    out.release()
    print(f"Saved {count} frames to {output_video}")

if __name__ == "__main__":
    process_video(sys.argv[1], sys.argv[2])
