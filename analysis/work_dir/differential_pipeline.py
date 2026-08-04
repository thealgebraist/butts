import cv2
import sys
import subprocess
import numpy as np

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
    
    cpp_proc = subprocess.Popen(['./fast_vectorizer'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=False)
    
    count = 0
    frame_size = W * H
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        orig_w, orig_h = frame.shape[1], frame.shape[0]
        small = cv2.resize(frame, (W, H))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        
        cpp_proc.stdin.write(gray.tobytes())
        cpp_proc.stdin.flush()
        
        data = cpp_proc.stdout.read(frame_size)
        if len(data) < frame_size:
            break
            
        differential_field = np.frombuffer(data, dtype=np.uint8).reshape((H, W))
        
        # Write grayscale magnitude field directly as 3-channel
        diff_color = cv2.cvtColor(differential_field, cv2.COLOR_GRAY2BGR)
        
        out.write(diff_color)
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
