190709
This run was done modifying the dilation kernel.
It tests for the following:
params = {
            'gaussWindow': range(1, 8, 2),
            'residualConnections': range(1, 8, 2),
            'sigma': np.linspace(0.1, 0.9, 5),
            'dilationIterations': range(1, 8, 2),
            'kernel': range(1, 8, 2)
        }
Overall, minor to no improvement by adding the kernel param. Will be removed.

# 190622_201853: [L] Good S->N landing. Very stable.
[INFO] Starting 3 iterations
[INFO] Using bbox frames 97 to 812
[INFO] Cached 1124 frames with shape x-1920 y-1080
[INFO] FPS: 91.07
gaussWindow: 1, residualConnections: 7, sigma: 0.7, dilationIterations: 1, kernel: 7, precision: 0.702, recall: 0.580, f1_Score: 0.635
[INFO] FPS: 85.83
gaussWindow: 1, residualConnections: 7, sigma: 0.1, dilationIterations: 1, kernel: 5, precision: 0.374, recall: 0.601, f1_Score: 0.461
[INFO] FPS: 63.13
gaussWindow: 7, residualConnections: 5, sigma: 0.9, dilationIterations: 5, kernel: 3, precision: 0.751, recall: 0.480, f1_Score: 0.585
# 190622_202211: [T]
[INFO] Starting 3 iterations
[INFO] Using bbox frames 1 to 883
[INFO] Cached 1124 frames with shape x-1920 y-1080
[INFO] FPS: 63.58
gaussWindow: 5, residualConnections: 7, sigma: 0.7, dilationIterations: 5, kernel: 3, precision: 0.921, recall: 0.760, f1_Score: 0.833
[INFO] FPS: 76.06
gaussWindow: 1, residualConnections: 7, sigma: 0.3, dilationIterations: 1, kernel: 7, precision: 0.376, recall: 0.778, f1_Score: 0.507
[INFO] FPS: 65.50
gaussWindow: 5, residualConnections: 7, sigma: 0.9, dilationIterations: 1, kernel: 7, precision: 0.927, recall: 0.750, f1_Score: 0.829
# 190624_200747: [FB]
[INFO] Starting 3 iterations
[INFO] Using bbox frames 1 to 1122
[INFO] Cached 1124 frames with shape x-1920 y-1080
[INFO] FPS: 75.54
gaussWindow: 1, residualConnections: 3, sigma: 0.1, dilationIterations: 1, kernel: 5, precision: 0.761, recall: 0.263, f1_Score: 0.391
[INFO] FPS: 81.68
gaussWindow: 1, residualConnections: 3, sigma: 0.1, dilationIterations: 1, kernel: 5, precision: 0.761, recall: 0.263, f1_Score: 0.391
[INFO] FPS: 67.71
gaussWindow: 7, residualConnections: 5, sigma: 0.3, dilationIterations: 1, kernel: 5, precision: 0.855, recall: 0.116, f1_Score: 0.204
# 190703_230610: [T]
[INFO] Starting 3 iterations
[INFO] Using bbox frames 26 to 299
[INFO] Cached 351 frames with shape x-1920 y-1080
[INFO] FPS: 68.81
gaussWindow: 5, residualConnections: 1, sigma: 0.3, dilationIterations: 3, kernel: 5, precision: 0.907, recall: 0.692, f1_Score: 0.785
[INFO] FPS: 70.21
gaussWindow: 3, residualConnections: 1, sigma: 0.1, dilationIterations: 3, kernel: 5, precision: 0.893, recall: 0.692, f1_Score: 0.780
[INFO] FPS: 66.41
gaussWindow: 7, residualConnections: 5, sigma: 0.9, dilationIterations: 1, kernel: 3, precision: 0.923, recall: 0.652, f1_Score: 0.765
# 190707_210056: [L]
[INFO] Starting 3 iterations
[INFO] Using bbox frames 1 to 243
[INFO] Cached 339 frames with shape x-1920 y-1080
[INFO] FPS: 56.06
gaussWindow: 7, residualConnections: 3, sigma: 0.9, dilationIterations: 7, kernel: 3, precision: 1.000, recall: 0.705, f1_Score: 0.827
[INFO] FPS: 57.32
gaussWindow: 1, residualConnections: 1, sigma: 0.1, dilationIterations: 1, kernel: 5, precision: 0.028, recall: 0.711, f1_Score: 0.054
[INFO] FPS: 66.47
gaussWindow: 7, residualConnections: 5, sigma: 0.7, dilationIterations: 1, kernel: 7, precision: 1.000, recall: 0.699, f1_Score: 0.823
