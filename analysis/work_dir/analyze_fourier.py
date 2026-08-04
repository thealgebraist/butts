import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def analyze_fourier(image_path, output_dir):
    # Load image and convert to grayscale
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Total Frame FFT
    F = np.fft.fft2(gray)
    F_shift = np.fft.fftshift(F)
    mag_spectrum_total = 20 * np.log(1 + np.abs(F_shift))
    
    # 2. Isolate Soda Can Edges (using high threshold Sobel to match previous logic)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag_spatial = np.sqrt(gx**2 + gy**2)
    
    # Assuming center isolation logic from earlier
    H, W = gray.shape
    center_x = W // 2
    can_mask = np.zeros_like(gray, dtype=np.uint8)
    
    threshold = 120
    for y in range(H):
        left_edge = center_x
        right_edge = center_x
        for x in range(center_x, -1, -1):
            if mag_spatial[y, x] > threshold:
                left_edge = x
                break
        for x in range(center_x, W):
            if mag_spatial[y, x] > threshold:
                right_edge = x
                break
        can_mask[y, left_edge:right_edge] = 255
        
    # Extract can and table
    can_only = cv2.bitwise_and(gray, gray, mask=can_mask)
    table_mask = cv2.bitwise_not(can_mask)
    table_only = cv2.bitwise_and(gray, gray, mask=table_mask)
    
    # 3. FFT of Can
    F_can = np.fft.fft2(can_only)
    F_can_shift = np.fft.fftshift(F_can)
    mag_spectrum_can = 20 * np.log(1 + np.abs(F_can_shift))
    
    # 4. FFT of Table
    F_table = np.fft.fft2(table_only)
    F_table_shift = np.fft.fftshift(F_table)
    mag_spectrum_table = 20 * np.log(1 + np.abs(F_table_shift))
    
    # 5. Energy Analysis
    # Define Low Frequency as D < 30 px, High Frequency as D >= 30 px
    D0 = 30
    Y, X = np.ogrid[:H, :W]
    center_u, center_v = H//2, W//2
    dist_sq = (Y - center_u)**2 + (X - center_v)**2
    
    low_freq_mask = dist_sq < D0**2
    high_freq_mask = dist_sq >= D0**2
    
    def calculate_energy(F_cplx):
        energy_matrix = np.abs(F_cplx)**2
        total_e = np.sum(energy_matrix)
        low_e = np.sum(energy_matrix[low_freq_mask])
        high_e = np.sum(energy_matrix[high_freq_mask])
        return (low_e / total_e) * 100, (high_e / total_e) * 100

    can_lf, can_hf = calculate_energy(F_can_shift)
    table_lf, table_hf = calculate_energy(F_table_shift)
    
    # Write analysis results to a text file for reference in LaTeX
    with open('fourier_analysis_results.txt', 'w') as f:
        f.write(f"Can Energy: Low Freq (<30px) = {can_lf:.2f}%, High Freq (>=30px) = {can_hf:.2f}%\n")
        f.write(f"Table Energy: Low Freq (<30px) = {table_lf:.2f}%, High Freq (>=30px) = {table_hf:.2f}%\n")

    # Generate Visualization Plot
    plt.figure(figsize=(15, 10))
    
    plt.subplot(231), plt.imshow(gray, cmap='gray')
    plt.title('Original Frame'), plt.xticks([]), plt.yticks([])
    
    plt.subplot(232), plt.imshow(table_only, cmap='gray')
    plt.title('Isolated Table'), plt.xticks([]), plt.yticks([])
    
    plt.subplot(233), plt.imshow(can_only, cmap='gray')
    plt.title('Isolated Can'), plt.xticks([]), plt.yticks([])
    
    plt.subplot(234), plt.imshow(mag_spectrum_total, cmap='gray', vmin=0, vmax=255)
    plt.title('Total Spectrum'), plt.xticks([]), plt.yticks([])
    
    plt.subplot(235), plt.imshow(mag_spectrum_table, cmap='gray', vmin=0, vmax=255)
    plt.title('Table Spectrum'), plt.xticks([]), plt.yticks([])
    
    plt.subplot(236), plt.imshow(mag_spectrum_can, cmap='gray', vmin=0, vmax=255)
    plt.title('Can Spectrum'), plt.xticks([]), plt.yticks([])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fourier_coefficient_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    analyze_fourier('/Users/anders/projects/thrash/center1/IMG_2559.jpeg', '/Users/anders/projects/pdf/')
