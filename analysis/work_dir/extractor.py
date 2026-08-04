import cv2
import sys
import struct

def extract(input_video, output_bin):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error: Could not open video.")
        sys.exit(1)
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0
    
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame.")
        sys.exit(1)
        
    h, w = frame.shape[:2]
    scale = 512.0 / w if w > 512 else 1.0
    out_w = int(w * scale)
    out_h = int(h * scale)
    
    if out_w % 2 != 0: out_w += 1
    if out_h % 2 != 0: out_h += 1
    
    with open(output_bin, 'wb') as f:
        # write header: w, h, fps
        f.write(struct.pack('ii f', out_w, out_h, fps))
        
        count = 0
        while ret:
            if scale != 1.0:
                frame = cv2.resize(frame, (out_w, out_h))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            f.write(gray.tobytes())
            ret, frame = cap.read()
            count += 1
            if count % 100 == 0:
                print(f"Extracted {count} frames...")
                
    print(f"Extraction complete: {count} frames.")
    cap.release()

if __name__ == "__main__":
    extract(sys.argv[1], sys.argv[2])
