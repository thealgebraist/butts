import cv2
import sys
import struct
import numpy as np

def extract_tensor(input_video, output_bin):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error opening video")
        sys.exit(1)
        
    frames = []
    count = 0
    # Process only a fraction of frames or heavily downscale to save disk space
    # 32x32 x N frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (32, 32))
        frames.append(small)
        count += 1
        if count >= 300: # Limit to 300 frames to save memory and space
            break
            
    tensor = np.stack(frames, axis=0).astype(np.float32) / 255.0
    T, H, W = tensor.shape
    
    with open(output_bin, 'wb') as f:
        f.write(struct.pack('iii', T, H, W))
        f.write(tensor.tobytes())
        
    print(f"Tensor {T}x{H}x{W} written to {output_bin}")

if __name__ == "__main__":
    extract_tensor(sys.argv[1], sys.argv[2])
