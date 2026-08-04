#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

void heat_equation(vector<uint8_t>& frame, int W, int H, int iterations) {
    vector<float> u(W * H);
    vector<float> unew(W * H);
    for (int i = 0; i < W * H; ++i) u[i] = frame[i] / 255.0f;
    
    float alpha = 0.2f; // Stability limit for 2D is 0.25
    
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

void process_frame_simd_l1_blocked(const uint8_t* in, uint8_t* out, int width, int height) {
    const int BLOCK_SIZE = 32;
    for (int by = 1; by < height - 1; by += BLOCK_SIZE) {
        for (int bx = 1; bx < width - 1; bx += BLOCK_SIZE) {
            int y_end = min(by + BLOCK_SIZE, height - 1);
            int x_end = min(bx + BLOCK_SIZE, width - 1);
            
            for (int y = by; y < y_end; ++y) {
                for (int x = bx; x < x_end; ++x) {
                    int t0 = in[(y-1)*width + x - 1], t1 = in[(y-1)*width + x], t2 = in[(y-1)*width + x + 1];
                    int m0 = in[y*width + x - 1],                       m2 = in[y*width + x + 1];
                    int b0 = in[(y+1)*width + x - 1], b1 = in[(y+1)*width + x], b2 = in[(y+1)*width + x + 1];
                    
                    int gx = -t0 + t2 - 2*m0 + 2*m2 - b0 + b2;
                    int gy = t0 + 2*t1 + t2 - b0 - 2*b1 - b2;
                    int mag = abs(gx) + abs(gy);
                    
                    out[y*width + x] = min(mag, 255); // First differential magnitude
                }
            }
        }
    }
}

void filter_outside_can(const vector<uint8_t>& mag_buffer, vector<uint8_t>& out_frame, int W, int H) {
    int center_x = W / 2;
    int threshold = 120; // Increased threshold to ensure we hit the intense outer boundary, ignoring inner textures
    
    for (int y = 0; y < H; ++y) {
        int left_edge = center_x;
        int right_edge = center_x;
        
        // Walk left from center to find left edge
        for (int x = center_x; x >= 0; --x) {
            if (mag_buffer[y * W + x] > threshold) {
                left_edge = x;
                break;
            }
        }
        
        // Walk right from center to find right edge
        for (int x = center_x; x < W; ++x) {
            if (mag_buffer[y * W + x] > threshold) {
                right_edge = x;
                break;
            }
        }
        
        // Mask everything outside the edges
        for (int x = 0; x < W; ++x) {
            if (x >= left_edge && x <= right_edge) {
                out_frame[y * W + x] = mag_buffer[y * W + x]; // Keep can edges
            } else {
                out_frame[y * W + x] = 0; // Filter away background
            }
        }
    }
}

int main() {
    int W = 512, H = 910;
    int N = W * H;
    vector<uint8_t> in_frame(N);
    vector<uint8_t> mag_buffer(N);
    vector<uint8_t> out_frame(N);

    while (cin.read(reinterpret_cast<char*>(in_frame.data()), N)) {
        // 1. Heat equation (24 iterations)
        heat_equation(in_frame, W, H, 24);
        
        // 2. L1 Blocked Sobel Vectorization (First Differential)
        fill(mag_buffer.begin(), mag_buffer.end(), 0);
        process_frame_simd_l1_blocked(in_frame.data(), mag_buffer.data(), W, H);
        
        // 3. Filter everything outside the center can
        filter_outside_can(mag_buffer, out_frame, W, H);
        
        // Output to stdout
        cout.write(reinterpret_cast<char*>(out_frame.data()), N);
    }
    return 0;
}
