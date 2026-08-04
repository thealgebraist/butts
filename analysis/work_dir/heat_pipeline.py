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
        scale = 512.0 / orig_w
        out_w = int(orig_w * scale)
        out_h = int(orig_h * scale)
        if out_w % 2 != 0: out_w += 1
        if out_h % 2 != 0: out_h += 1
        
        # We enforce exactly W=512, H=910 (which is 1000x1776 downscaled)
        # to match the C++ code exactly.
        small = cv2.resize(frame, (W, H))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        
        cpp_proc.stdin.write(gray.tobytes())
        cpp_proc.stdin.flush()
        
        data = cpp_proc.stdout.read(frame_size)
        if len(data) < frame_size:
            break
            
        edges = np.frombuffer(data, dtype=np.uint8).reshape((H, W))
        
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
        filtered_contours = []
        for cnt in contours:
            if cv2.arcLength(cnt, False) > 30:
                approx = cv2.approxPolyDP(cnt, 2.0, False)
                filtered_contours.append(approx)
        
        vector_frame = np.ones((H, W, 3), dtype=np.uint8) * 255
        cv2.polylines(vector_frame, filtered_contours, False, (0, 0, 0), 2)
        
        for cnt in filtered_contours:
            if len(cnt) >= 5:
                ellipse = cv2.fitEllipse(cnt)
                if ellipse[1][0] < W and ellipse[1][1] < H:
                    cv2.ellipse(vector_frame, ellipse, (0, 0, 255), 1, cv2.LINE_AA)
        
        out.write(vector_frame)
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
