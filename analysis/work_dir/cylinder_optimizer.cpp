#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

// 3D Point
struct Point3D { float x, y, z; };
struct Point2D { float x, y; };

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

// L1 Blocked Sobel Magnitude
void compute_differential(const uint8_t* in, uint8_t* out, int width, int height) {
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
                    out[y*width + x] = min(abs(gx) + abs(gy), 255);
                }
            }
        }
    }
}

// Generate base cylinder point cloud (skeleton)
vector<Point3D> generate_cylinder() {
    vector<Point3D> points;
    float R = 1.0f;
    float H = 4.0f;
    int segments = 36;
    // Top circle
    for (int i = 0; i < segments; ++i) {
        float theta = i * 2.0f * M_PI / segments;
        points.push_back({R * cos(theta), -H/2.0f, R * sin(theta)});
    }
    // Bottom circle
    for (int i = 0; i < segments; ++i) {
        float theta = i * 2.0f * M_PI / segments;
        points.push_back({R * cos(theta), H/2.0f, R * sin(theta)});
    }
    // Vertical lines
    for (int i = 0; i < 4; ++i) {
        float theta = i * M_PI / 2.0f;
        for (int j = 1; j < 10; ++j) {
            float y = -H/2.0f + j * (H / 10.0f);
            points.push_back({R * cos(theta), y, R * sin(theta)});
        }
    }
    return points;
}

// Project and compute cost
float evaluate_cylinder(const vector<Point3D>& base_points, const vector<uint8_t>& mag, int W, int H, 
                        float pitch, float yaw, float roll, float scale) {
    float cp = cos(pitch), sp = sin(pitch);
    float cy = cos(yaw), sy = sin(yaw);
    float cr = cos(roll), sr = sin(roll);
    
    // Rotation matrix Rx * Ry * Rz
    float r00 = cy * cr;
    float r01 = cy * sr;
    float r02 = -sy;
    float r10 = sp * sy * cr - cp * sr;
    float r11 = sp * sy * sr + cp * cr;
    float r12 = sp * cy;
    float r20 = cp * sy * cr + sp * sr;
    float r21 = cp * sy * sr - sp * cr;
    float r22 = cp * cy;

    float total_mag = 0;
    int valid_points = 0;
    
    float f = 500.0f * scale; // Focal length / Scale combined
    float Z_offset = 10.0f;

    for (const auto& p : base_points) {
        // Rotate
        float rx = r00 * p.x + r01 * p.y + r02 * p.z;
        float ry = r10 * p.x + r11 * p.y + r12 * p.z;
        float rz = r20 * p.x + r21 * p.y + r22 * p.z + Z_offset;
        
        if (rz <= 0.1f) continue;
        
        // Project
        int px = static_cast<int>(f * rx / rz + W / 2);
        int py = static_cast<int>(f * ry / rz + H / 2);
        
        if (px >= 0 && px < W && py >= 0 && py < H) {
            total_mag += mag[py * W + px];
            valid_points++;
        }
    }
    
    if (valid_points == 0) return 0.0f;
    return total_mag / valid_points; // Average magnitude
}

// Coordinate Descent Binary Search
void optimize_cylinder(const vector<Point3D>& base_points, const vector<uint8_t>& mag, int W, int H, 
                       float& pitch, float& yaw, float& roll, float& scale) {
    float params[4] = {pitch, yaw, roll, scale};
    float ranges[4] = {0.5f, 0.5f, 0.5f, 0.5f}; // Search ranges for each
    
    for (int iter = 0; iter < 5; ++iter) {
        for (int p = 0; p < 4; ++p) {
            float low = params[p] - ranges[p];
            float high = params[p] + ranges[p];
            
            // Binary search for maximum of concave function
            for (int b_iter = 0; b_iter < 8; ++b_iter) {
                float mid1 = low + (high - low) / 3.0f;
                float mid2 = high - (high - low) / 3.0f;
                
                float temp1[4] = {params[0], params[1], params[2], params[3]};
                float temp2[4] = {params[0], params[1], params[2], params[3]};
                temp1[p] = mid1;
                temp2[p] = mid2;
                
                float cost1 = evaluate_cylinder(base_points, mag, W, H, temp1[0], temp1[1], temp1[2], temp1[3]);
                float cost2 = evaluate_cylinder(base_points, mag, W, H, temp2[0], temp2[1], temp2[2], temp2[3]);
                
                if (cost1 > cost2) {
                    high = mid2;
                } else {
                    low = mid1;
                }
            }
            params[p] = (low + high) / 2.0f;
            ranges[p] *= 0.5f; // Shrink search range for next outer iteration
        }
    }
    pitch = params[0]; yaw = params[1]; roll = params[2]; scale = params[3];
}

struct FrameHeader {
    float pitch, yaw, roll, scale;
};

int main() {
    int W = 512, H = 910;
    int N = W * H;
    vector<uint8_t> in_frame(N);
    vector<uint8_t> mag_buffer(N);
    
    vector<Point3D> base_cylinder = generate_cylinder();

    FrameHeader header;
    while (cin.read(reinterpret_cast<char*>(&header), sizeof(FrameHeader))) {
        if (!cin.read(reinterpret_cast<char*>(in_frame.data()), N)) break;
        
        // 1. Heat equation
        heat_equation(in_frame, W, H, 24);
        
        // 2. Differential field
        fill(mag_buffer.begin(), mag_buffer.end(), 0);
        compute_differential(in_frame.data(), mag_buffer.data(), W, H);
        
        // 3. Binary Search / Optimization
        optimize_cylinder(base_cylinder, mag_buffer, W, H, header.pitch, header.yaw, header.roll, header.scale);
        
        // Output optimized parameters
        cout.write(reinterpret_cast<char*>(&header), sizeof(FrameHeader));
        cout.flush();
    }
    return 0;
}
