#include <iostream>
#include <vector>
#include <cmath>
#include <random>

using namespace std;

int main() {
    int W = 512, H = 512;
    int N = W * H;
    vector<uint8_t> frame(N);
    vector<float> saliency(N);
    
    mt19937 rng(42);
    uniform_int_distribution<int> dist(0, 1);
    
    while (cin.read(reinterpret_cast<char*>(frame.data()), N)) {
        fill(saliency.begin(), saliency.end(), 0.0f);
        
        // Compute stochastic trace of the local Hessian at each pixel
        int M = 8;
        
        for (int y = 1; y < H - 1; ++y) {
            for (int x = 1; x < W - 1; ++x) {
                int idx = y * W + x;
                
                // Finite differences for second derivatives
                float I_xx = frame[y * W + x + 1] - 2 * frame[idx] + frame[y * W + x - 1];
                float I_yy = frame[(y + 1) * W + x] - 2 * frame[idx] + frame[(y - 1) * W + x];
                float I_xy = (frame[(y + 1) * W + x + 1] - frame[(y + 1) * W + x - 1] 
                            - frame[(y - 1) * W + x + 1] + frame[(y - 1) * W + x - 1]) / 4.0f;
                            
                float trace_est = 0;
                for (int m = 0; m < M; ++m) {
                    float v1 = dist(rng) ? 1.0f : -1.0f;
                    float v2 = dist(rng) ? 1.0f : -1.0f;
                    
                    // v^T H v = I_xx v1^2 + 2 I_xy v1 v2 + I_yy v2^2
                    // Since v1^2 = v2^2 = 1, it's I_xx + I_yy + 2 I_xy v1 v2
                    trace_est += I_xx + I_yy + 2.0f * I_xy * v1 * v2;
                }
                trace_est /= M;
                
                // Energy is trace squared
                saliency[idx] = trace_est * trace_est;
            }
        }
        
        // Bounding box extraction
        int x_min = W, y_min = H, x_max = 0, y_max = 0;
        float threshold = 500.0f; // Saliency threshold
        
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                if (saliency[y * W + x] > threshold) {
                    if (x < x_min) x_min = x;
                    if (y < y_min) y_min = y;
                    if (x > x_max) x_max = x;
                    if (y > y_max) y_max = y;
                }
            }
        }
        
        if (x_min > x_max) { // No salient pixels found
            x_min = 0; y_min = 0; x_max = 0; y_max = 0;
        }
        
        // Output bounding box coordinates to stdout
        cout << x_min << " " << y_min << " " << x_max << " " << y_max << endl;
    }
    
    return 0;
}
