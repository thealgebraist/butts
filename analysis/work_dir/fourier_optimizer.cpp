#include <Accelerate/Accelerate.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

// Heat equation
void heat_equation(vector<uint8_t>& frame, int W, int H, int iterations) {
    vector<float> u(W * H);
    vector<float> unew(W * H);
    for (int i = 0; i < W * H; ++i) u[i] = frame[i] / 255.0f;
    float alpha = 0.2f;
    for (int iter = 0; iter < iterations; ++iter) {
        for (int y = 1; y < H - 1; ++y) {
            for (int x = 1; x < W - 1; ++x) {
                int idx = y * W + x;
                unew[idx] = u[idx] + alpha * (
                    u[y * W + x + 1] + u[y * W + x - 1] + 
                    u[(y + 1) * W + x] + u[(y - 1) * W + x] - 
                    4.0f * u[idx]
                );
            }
        }
        u = unew;
    }
    for (int i = 0; i < W * H; ++i) {
        frame[i] = static_cast<uint8_t>(min(max(u[i] * 255.0f, 0.0f), 255.0f));
    }
}

int main() {
    int log2W = 9;
    int log2H = 10;
    int W = 1 << log2W; // 512
    int H = 1 << log2H; // 1024
    int N = W * H;
    
    // Create FFT setup structure
    FFTSetup setup = vDSP_create_fftsetup(max(log2W, log2H), FFT_RADIX2);
    if (setup == nullptr) {
        cerr << "Failed to allocate vDSP FFT setup." << endl;
        return 1;
    }
    
    vector<uint8_t> in_frame(N);
    vector<uint8_t> out_frame(N);
    
    vector<float> realP(N);
    vector<float> imagP(N);
    DSPSplitComplex complexData = {realP.data(), imagP.data()};
    
    // Load the Empirical Custom Frequency Mask directly from memory
    vector<float> custom_mask(N);
    FILE* fp = fopen("custom_freq_mask.bin", "rb");
    if (fp) {
        fread(custom_mask.data(), sizeof(float), N, fp);
        fclose(fp);
        custom_mask[0] = 0.0f; // Force DC component to 0 to eliminate average brightness bleed
    } else {
        cerr << "Failed to load custom_freq_mask.bin" << endl;
        return 1;
    }
    
    float scale = 1.0f / N;
    
    while (cin.read(reinterpret_cast<char*>(in_frame.data()), N)) {
        // Load real spatial data, zero imaginary part
        for (int i = 0; i < N; ++i) {
            realP[i] = static_cast<float>(in_frame[i]);
            imagP[i] = 0.0f;
        }
        
        // 2D Forward FFT
        vDSP_fft2d_zip(setup, &complexData, 1, 0, log2W, log2H, FFT_FORWARD);
        
        // Apply Empirical Frequency Mask in Frequency Domain
        for (int i = 0; i < N; ++i) {
            realP[i] *= custom_mask[i];
            imagP[i] *= custom_mask[i];
        }
        
        // 2D Inverse FFT
        vDSP_fft2d_zip(setup, &complexData, 1, 0, log2W, log2H, FFT_INVERSE);
        
        // Extract magnitude, scale back, and threshold/clamp to 8-bit
        for (int i = 0; i < N; ++i) {
            // After IFFT, spatial result is scaled by N
            float mag = sqrt(realP[i] * realP[i] + imagP[i] * imagP[i]) * scale;
            
            // To make edges highly visible, we can amplify the signal slightly
            mag *= 2.0f; 
            
            out_frame[i] = static_cast<uint8_t>(min(max(mag, 0.0f), 255.0f));
        }
        
        // Stream out 
        cout.write(reinterpret_cast<char*>(out_frame.data()), N);
        cout.flush();
    }
    
    vDSP_destroy_fftsetup(setup);
    return 0;
}
