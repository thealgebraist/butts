import cv2
import sys
import subprocess
import os

def process_video(input_video, output_video):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error opening video")
        sys.exit(1)
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))
    
    # Start C++ bounding box estimator
    cpp_proc = subprocess.Popen(['./hessian_bbox'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=False)
    
    scale_w = w / 512.0
    scale_h = h / 512.0
    
    count = 0
    # Process only 300 frames to save execution time, or process all?
    # The prompt says "in each frame". We will process all frames.
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (512, 512))
        
        cpp_proc.stdin.write(small.tobytes())
        cpp_proc.stdin.flush()
        
        line = cpp_proc.stdout.readline().decode('utf-8').strip()
        if not line:
            break
            
        x_min, y_min, x_max, y_max = map(int, line.split())
        
        if x_min != 0 or x_max != 0:
            # Scale back to original resolution
            orig_x_min = int(x_min * scale_w)
            orig_y_min = int(y_min * scale_h)
            orig_x_max = int(x_max * scale_w)
            orig_y_max = int(y_max * scale_h)
            
            # Draw formal mathematically derived bounding box
            cv2.rectangle(frame, (orig_x_min, orig_y_min), (orig_x_max, orig_y_max), (0, 0, 255), 4)
            cv2.putText(frame, "Hessian Trick Saliency BBox", (orig_x_min, orig_y_min - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        out.write(frame)
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
