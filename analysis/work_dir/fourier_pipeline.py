import cv2
import sys
import subprocess
import numpy as np
import pywt

def process_video(input_video, output_video):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error opening video")
        sys.exit(1)
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0
    
    # Strictly aligned to Radix-2 sizes (512, 1024)
    W, H = 512, 1024
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (W, H))
    
    cpp_proc = subprocess.Popen(['./fourier_optimizer'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=False)
    
    count = 0
    frame_size = W * H
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        small = cv2.resize(frame, (W, H))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        
        cpp_proc.stdin.write(gray.tobytes())
        cpp_proc.stdin.flush()
        
        data = cpp_proc.stdout.read(frame_size)
        if len(data) < frame_size:
            break
            
        edge_map = np.frombuffer(data, dtype=np.uint8).reshape((H, W))
        
        # Extract the binary structure of the can
        _, binary_mask = cv2.threshold(edge_map, 30, 255, cv2.THRESH_BINARY)
        points = cv2.findNonZero(binary_mask)
        solid_mask = np.zeros((H, W), dtype=np.uint8)
        if points is not None:
            hull = cv2.convexHull(points)
            cv2.fillConvexPoly(solid_mask, hull, 255)
            
        masked_frame = np.zeros_like(small)
        
        # Apply Haar Wavelet DWT to each color channel
        for c in range(3):
            channel = small[:, :, c]
            coeffs = pywt.dwt2(channel, 'haar')
            LL, (LH, HL, HH) = coeffs
            
            # Subbands are half the size of the original frame (W/2, H/2)
            subband_H, subband_W = LL.shape
            
            # Downsample the solid mask to match the subband dimensions
            mask_subband = cv2.resize(solid_mask, (subband_W, subband_H), interpolation=cv2.INTER_NEAREST)
            
            # Invert mask: 0 inside the can, 1 outside the can
            inv_mask = (mask_subband == 0).astype(np.float32)
            
            # Annihilate ALL wavelet coefficients inside the mask
            LL *= inv_mask
            LH *= inv_mask
            HL *= inv_mask
            HH *= inv_mask
            
            # Inverse 2D DWT
            rec_channel = pywt.idwt2((LL, (LH, HL, HH)), 'haar')
            
            # IDWT may produce float values that slightly overflow 0-255 bounds
            rec_channel = np.clip(rec_channel, 0, 255)
            masked_frame[:, :, c] = rec_channel.astype(np.uint8)
        
        out.write(masked_frame)
        count += 1
        if count % 100 == 0:
            print(f"Processed {count} frames...")
            
    cpp_proc.stdin.close()
    cpp_proc.wait()
    cap.release()
    out.release()
    print(f"Saved {count} frames to {output_video}")

if __name__ == "__main__":
    process_video(sys.argv[1], sys.argv[2])
