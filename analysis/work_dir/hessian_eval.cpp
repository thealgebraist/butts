#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <numeric>

using namespace std;

// Hessian Vector Product: H = -Laplacian (3D)
void HVP(const vector<float>& v, vector<float>& Hv, int T, int H, int W) {
    fill(Hv.begin(), Hv.end(), 0.0f);
    
    for (int t = 0; t < T; ++t) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                int idx = t * H * W + y * W + x;
                float val = v[idx];
                float sum_neighbors = 0.0f;
                int neighbors = 0;
                
                if (t > 0) { sum_neighbors += v[(t - 1) * H * W + y * W + x]; neighbors++; }
                if (t < T - 1) { sum_neighbors += v[(t + 1) * H * W + y * W + x]; neighbors++; }
                if (y > 0) { sum_neighbors += v[t * H * W + (y - 1) * W + x]; neighbors++; }
                if (y < H - 1) { sum_neighbors += v[t * H * W + (y + 1) * W + x]; neighbors++; }
                if (x > 0) { sum_neighbors += v[t * H * W + y * W + (x - 1)]; neighbors++; }
                if (x < W - 1) { sum_neighbors += v[t * H * W + y * W + (x + 1)]; neighbors++; }
                
                // -Laplacian
                Hv[idx] = -(sum_neighbors - neighbors * val);
            }
        }
    }
}

float dot_product(const vector<float>& a, const vector<float>& b) {
    float sum = 0;
    for (size_t i = 0; i < a.size(); ++i) sum += a[i] * b[i];
    return sum;
}

float norm2(const vector<float>& a) {
    return sqrt(dot_product(a, a));
}

int main(int argc, char** argv) {
    if (argc < 2) return 1;
    
    ifstream fin(argv[1], ios::binary);
    if (!fin) return 1;
    
    int T, H, W;
    fin.read((char*)&T, sizeof(int));
    fin.read((char*)&H, sizeof(int));
    fin.read((char*)&W, sizeof(int));
    
    int N = T * H * W;
    vector<float> tensor(N);
    fin.read((char*)tensor.data(), N * sizeof(float));
    
    mt19937 rng(42);
    uniform_int_distribution<int> dist(0, 1);
    
    vector<float> v(N);
    for (int i = 0; i < N; ++i) {
        v[i] = dist(rng) ? 1.0f : -1.0f;
    }
    
    vector<float> Hv(N);
    HVP(v, Hv, T, H, W);
    
    // 1. Trace Estimation
    float trace_H = dot_product(v, Hv);
    
    // 2. Frobenius Norm (Trace of H^2)
    float frobenius_sq = dot_product(Hv, Hv);
    
    // 3. Leading Eigenvalue via Power Iteration
    vector<float> pi_v = v;
    vector<float> pi_Hv(N);
    float lambda_max = 0.0f;
    for (int iter = 0; iter < 20; ++iter) {
        HVP(pi_v, pi_Hv, T, H, W);
        lambda_max = dot_product(pi_v, pi_Hv) / dot_product(pi_v, pi_v);
        float n = norm2(pi_Hv);
        for (int i = 0; i < N; ++i) pi_v[i] = pi_Hv[i] / n;
    }
    
    // 4. Spectral Gap (Min Eigenvalue via shifted Power Iteration)
    // H has max eigenvalue bounded by ~12 for 3D Laplacian.
    // Use (12*I - H) to find smallest eigenvalue.
    vector<float> min_v = v;
    vector<float> min_Hv(N);
    float shift = 12.0f;
    float lambda_shifted = 0.0f;
    for (int iter = 0; iter < 20; ++iter) {
        HVP(min_v, min_Hv, T, H, W);
        for (int i = 0; i < N; ++i) min_Hv[i] = shift * min_v[i] - min_Hv[i]; // (12I - H)v
        lambda_shifted = dot_product(min_v, min_Hv) / dot_product(min_v, min_v);
        float n = norm2(min_Hv);
        for (int i = 0; i < N; ++i) min_v[i] = min_Hv[i] / n;
    }
    float lambda_min = shift - lambda_shifted;
    
    cout << "--- Hessian Trick Analysis ---\n";
    cout << "Spatiotemporal Elements (N): " << N << "\n";
    cout << "1. Trace(H): " << trace_H << "\n";
    cout << "2. Frobenius Norm ||H||_F: " << sqrt(frobenius_sq) << "\n";
    cout << "3. Leading Eigenvalue (Max curvature): " << lambda_max << "\n";
    cout << "4. Spectral Gap (Min Eigenvalue): " << lambda_min << "\n";
    
    return 0;
}
