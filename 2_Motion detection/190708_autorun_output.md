190708
This run was done without modifying the dilation kernel.
It tests for the following:
params = {
            'gaussWindow': range(1, 8, 2),
            'residualConnections': range(1, 8, 2),
            'sigma': np.linspace(0.1, 0.9, 5),
            'dilationIterations': range(1, 8, 2),
        }
Overall, the results are excellent and very steady accross the different runs.

# 190622_201853: [L] Good S->N landing. Very stable.
[INFO] Starting 3 iterations
[INFO] Using bbox frames 97 to 812
[INFO] Cached 1124 frames with shape x-1920 y-1080
[INFO] FPS: 92.07
gaussWindow: 1, residualConnections: 7, sigma: 0.7, dilationIterations: 3, precision: 0.702, recall: 0.580, f1_Score: 0.635
[INFO] FPS: 96.83
gaussWindow: 1, residualConnections: 7, sigma: 0.1, dilationIterations: 3, precision: 0.404, recall: 0.596, f1_Score: 0.481
[INFO] FPS: 69.77
gaussWindow: 7, residualConnections: 5, sigma: 0.9, dilationIterations: 5, precision: 0.751, recall: 0.480, f1_Score: 0.585
# 190622_202211: [T]
[INFO] Starting 3 iterations
[INFO] Using bbox frames 1 to 883
[INFO] Cached 1124 frames with shape x-1920 y-1080
[INFO] FPS: 76.61
gaussWindow: 5, residualConnections: 7, sigma: 0.7, dilationIterations: 5, precision: 0.921, recall: 0.760, f1_Score: 0.833
[INFO] FPS: 99.57
gaussWindow: 1, residualConnections: 7, sigma: 0.3, dilationIterations: 3, precision: 0.376, recall: 0.778, f1_Score: 0.507
[INFO] FPS: 83.67
gaussWindow: 5, residualConnections: 7, sigma: 0.9, dilationIterations: 3, precision: 0.927, recall: 0.750, f1_Score: 0.829
# 190624_200747: [FB]
[INFO] Starting 3 iterations
[INFO] Using bbox frames 1 to 1122
[INFO] Cached 1124 frames with shape x-1920 y-1080
[INFO] FPS: 73.61
gaussWindow: 1, residualConnections: 3, sigma: 0.1, dilationIterations: 3, precision: 0.698, recall: 0.238, f1_Score: 0.355
[INFO] FPS: 76.15
gaussWindow: 1, residualConnections: 5, sigma: 0.1, dilationIterations: 1, precision: 0.561, recall: 0.254, f1_Score: 0.349
[INFO] FPS: 71.00
gaussWindow: 7, residualConnections: 3, sigma: 0.3, dilationIterations: 3, precision: 0.829, recall: 0.082, f1_Score: 0.149
# **190703_230610**: [T]
[INFO] Starting 3 iterations
[INFO] Using bbox frames 26 to 299
[INFO] Cached 351 frames with shape x-1920 y-1080
[INFO] FPS: 72.39
gaussWindow: 5, residualConnections: 1, sigma: 0.1, dilationIterations: 5, precision: 0.900, recall: 0.692, f1_Score: 0.783
[INFO] FPS: 81.69
gaussWindow: 1, residualConnections: 1, sigma: 0.1, dilationIterations: 5, precision: 0.838, recall: 0.692, f1_Score: 0.758
[INFO] FPS: 71.90
gaussWindow: 7, residualConnections: 5, sigma: 0.9, dilationIterations: 1, precision: 0.923, recall: 0.652, f1_Score: 0.765
# **190707_210056**: [L]
[INFO] Starting 3 iterations
[INFO] Using bbox frames 1 to 243
[INFO] Cached 339 frames with shape x-1920 y-1080
[INFO] FPS: 61.16
gaussWindow: 7, residualConnections: 3, sigma: 0.9, dilationIterations: 7, precision: 1.000, recall: 0.705, f1_Score: 0.827
[INFO] FPS: 64.34
gaussWindow: 1, residualConnections: 1, sigma: 0.1, dilationIterations: 3, precision: 0.034, recall: 0.711, f1_Score: 0.066
[INFO] FPS: 72.45
gaussWindow: 7, residualConnections: 5, sigma: 0.7, dilationIterations: 3, precision: 1.000, recall: 0.699, f1_Score: 0.823
