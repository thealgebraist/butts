import cv2
import numpy as np

def generate_empirical_mask(can_path, table_path, output_bin):
    W, H = 512, 1024
    
    can_img = cv2.imread(can_path, cv2.IMREAD_GRAYSCALE)
    table_img = cv2.imread(table_path, cv2.IMREAD_GRAYSCALE)
    
    can_img = cv2.resize(can_img, (W, H))
    table_img = cv2.resize(table_img, (W, H))
    
    # Forward 2D FFT
    F_can = np.fft.fft2(can_img)
    F_table = np.fft.fft2(table_img)
    
    # Power spectral densities
    P_can = np.abs(F_can)
    P_table = np.abs(F_table)
    
    # Normalize powers to avoid illumination bias
    P_can_norm = P_can / np.max(P_can)
    P_table_norm = P_table / np.max(P_table)
    
    # Create mask: keep frequency if it's strictly dominant in the can's structure
    # meaning its normalized power in the can is greater than its normalized power in the table
    # We add a small scalar threshold to avoid keeping zero-energy background noise.
    # The rule: P_can > P_table * 1.5 AND P_can > some_noise_floor
    noise_floor = 1e-4
    
    mask = (P_can_norm > (P_table_norm * 1.5)) & (P_can_norm > noise_floor)
    
    # The DC component (0,0) dominates everything, but we don't want to kill the average intensity
    # Actually, we should probably keep the DC component to maintain average brightness.
    mask[0, 0] = True
    
    # Convert mask to float32 exactly matching the C++ memory layout (1.0f or 0.0f)
    mask_float = np.zeros((H, W), dtype=np.float32)
    mask_float[mask] = 1.0
    
    # Save directly to binary
    mask_float.tofile(output_bin)
    print(f"Saved custom frequency mask to {output_bin} ({np.sum(mask)} / {W*H} frequencies active)")

if __name__ == "__main__":
    generate_empirical_mask('/Users/anders/projects/thrash/center1/can.png', 
                            '/Users/anders/projects/thrash/center1/table.png', 
                            '/Users/anders/projects/thrash/work_dir/custom_freq_mask.bin')
