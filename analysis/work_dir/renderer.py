import cv2
import numpy as np
import sys
import struct

def render(input_bin, output_video):
    with open(input_bin, 'rb') as f:
        # read header
        header = f.read(12)
        if len(header) < 12:
            print("Error: empty input.")
            sys.exit(1)
        w, h, fps = struct.unpack('ii f', header)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))
        
        count = 0
        frame_size = w * h
        
        while True:
            data = f.read(frame_size)
            if len(data) < frame_size:
                break
            
            # Reconstruct binary edge image
            edges = np.frombuffer(data, dtype=np.uint8).reshape((h, w))
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
            filtered_contours = []
            for cnt in contours:
                if cv2.arcLength(cnt, False) > 30:
                    approx = cv2.approxPolyDP(cnt, 2.0, False)
                    filtered_contours.append(approx)
            
            vector_frame = np.ones((h, w, 3), dtype=np.uint8) * 255
            cv2.polylines(vector_frame, filtered_contours, False, (0, 0, 0), 2)
            
            for cnt in filtered_contours:
                if len(cnt) >= 5:
                    ellipse = cv2.fitEllipse(cnt)
                    if ellipse[1][0] < w and ellipse[1][1] < h:
                        cv2.ellipse(vector_frame, ellipse, (0, 0, 255), 1, cv2.LINE_AA)
            
            out.write(vector_frame)
            count += 1
            if count % 100 == 0:
                print(f"Rendered {count} frames...")
                
    out.release()
    print(f"Rendering complete: {count} frames saved to {output_video}")

if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2])
