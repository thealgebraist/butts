import cv2
import numpy as np
import sys
import gc
from scipy.linalg import svd

def process_eigen_video(input_video, table_image, output_video):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error opening video")
        sys.exit(1)
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0
    
    # Target resolution for SVD to remain within memory constraints (< 4GB)
    W_target, H_target = 256, 512
    W_orig, H_orig = 512, 1024
    
    # Load Table Reference Vector
    table_img = cv2.imread(table_image, cv2.IMREAD_GRAYSCALE)
    table_small = cv2.resize(table_img, (W_target, H_target))
    t_vec = table_small.astype(np.float32).flatten()
    # Normalize reference vector for cosine similarity
    t_norm = np.linalg.norm(t_vec)
    if t_norm > 0:
        t_vec /= t_norm
    
    frames = []
    print("Extracting frames and building spatio-temporal matrix X...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert to grayscale and downsample
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (W_target, H_target))
        frames.append(small.astype(np.float32).flatten())
        
    cap.release()
    
    # Build Matrix X (Pixels x Frames)
    # Shape: (131072, N)
    X = np.column_stack(frames)
    N = X.shape[1]
    del frames
    gc.collect()
    
    print(f"Matrix X built. Shape: {X.shape}. Memory: {X.nbytes / (1024**3):.2f} GB")
    print("Computing Global Singular Value Decomposition (SVD)... this may take a moment.")
    
    # Compute Economy SVD
    # X = U * S * V^T
    # U is (131072, N), S is (N,), Vh is (N, N)
    U, S, Vh = svd(X, full_matrices=False)
    
    print("SVD completed. Performing Target Vector Correlation Ablation...")
    
    # Calculate Cosine Similarities against the table vector
    # U[:, i] is the i-th Eigenframe
    ablated_count = 0
    for i in range(N):
        u_i = U[:, i]
        # U columns are already unit vectors, so dot product is cosine similarity
        similarity = np.abs(np.dot(u_i, t_vec))
        
        # 10% Correlation Threshold
        if similarity > 0.1:
            S[i] = 0.0
            ablated_count += 1
            
    print(f"Ablated {ablated_count} table-correlated eigenframes out of {N}.")
    print("Reconstructing spatiotemporal matrix from surviving eigenframes...")
    
    # Reconstruct X_hat
    # X_hat = U * S * V^T
    # We can multiply S into Vh first to save memory
    Vh = S[:, np.newaxis] * Vh
    X_hat = np.dot(U, Vh)
    
    # Memory cleanup
    del U, S, Vh, X
    gc.collect()
    
    print("Reassembly complete. Streaming upscaled frames to FFmpeg...")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (W_orig, H_orig))
    
    for i in range(N):
        # Extract reconstructed flattened frame
        rec_flat = X_hat[:, i]
        # Reshape to downsampled spatial dimensions
        rec_frame = rec_flat.reshape((H_target, W_target))
        # Clip to valid range and cast
        rec_frame = np.clip(rec_frame, 0, 255).astype(np.uint8)
        # Upscale back to HD
        hd_frame = cv2.resize(rec_frame, (W_orig, H_orig), interpolation=cv2.INTER_CUBIC)
        # Convert to BGR for standard video writing
        hd_color = cv2.cvtColor(hd_frame, cv2.COLOR_GRAY2BGR)
        out.write(hd_color)
        
        if (i + 1) % 100 == 0:
            print(f"Encoded {i + 1} frames...")
            
    out.release()
    print("Global Eigenframe Ablation sequence generation complete.")

if __name__ == "__main__":
    process_eigen_video(sys.argv[1], sys.argv[2], sys.argv[3])
