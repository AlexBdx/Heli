190710
In this run I added a cv2.erode after the dilation with identical parameters.
It tests for the following:
params = {
            'gaussWindow': range(1, 8, 2),
            'residualConnections': range(1, 8, 2),
            'sigma': np.linspace(0.1, 0.9, 5),
            'dilationIterations': range(1, 8, 2),
        }
Overall, a lot better on close range identification! But it makes life harder when the heli is far.
I might end up keeping this one!

# 190622_201853: [L] Good S->N landing. Very stable.
[INFO] Starting 3 iterations
[INFO] Using bbox frames 97 to 812
[INFO] Cached 1124 frames with shape x-1920 y-1080
[INFO] FPS: 63.80
gaussWindow: 1, residualConnections: 7, sigma: 0.7, dilationIterations: 7, precision: 0.761, recall: 0.602, f1_Score: 0.672
[INFO] FPS: 60.96
gaussWindow: 1, residualConnections: 7, sigma: 0.1, dilationIterations: 7, precision: 0.589, recall: 0.617, f1_Score: 0.602
[INFO] FPS: 49.09
gaussWindow: 7, residualConnections: 5, sigma: 0.9, dilationIterations: 7, precision: 0.793, recall: 0.501, f1_Score: 0.614
# 190622_202211: [T]
[INFO] Starting 3 iterations
[INFO] Using bbox frames 1 to 883
[INFO] Cached 1124 frames with shape x-1920 y-1080
[INFO] FPS: 41.77
gaussWindow: 5, residualConnections: 7, sigma: 0.9, dilationIterations: 7, precision: 0.960, recall: 0.757, f1_Score: 0.847
[INFO] FPS: 44.84
gaussWindow: 1, residualConnections: 7, sigma: 0.7, dilationIterations: 3, precision: 0.669, recall: 0.778, f1_Score: 0.720
[INFO] FPS: 42.53
gaussWindow: 5, residualConnections: 7, sigma: 0.9, dilationIterations: 7, precision: 0.960, recall: 0.757, f1_Score: 0.847
# 190624_200747: [FB]
[INFO] Starting 3 iterations
[INFO] Using bbox frames 1 to 1122
[INFO] Cached 1124 frames with shape x-1920 y-1080
[INFO] FPS: 50.27
gaussWindow: 1, residualConnections: 5, sigma: 0.1, dilationIterations: 3, precision: 0.496, recall: 0.202, f1_Score: 0.287
[INFO] FPS: 46.41
gaussWindow: 1, residualConnections: 5, sigma: 0.1, dilationIterations: 3, precision: 0.496, recall: 0.202, f1_Score: 0.287
[INFO] FPS: 45.94
gaussWindow: 7, residualConnections: 5, sigma: 0.3, dilationIterations: 1, precision: 0.559, recall: 0.076, f1_Score: 0.133
# 190703_230610: [T]
[INFO] Starting 3 iterations
[INFO] Using bbox frames 26 to 299
[INFO] Cached 351 frames with shape x-1920 y-1080
[INFO] FPS: 65.55
gaussWindow: 1, residualConnections: 1, sigma: 0.5, dilationIterations: 7, precision: 0.875, recall: 0.681, f1_Score: 0.766
[INFO] FPS: 66.67
gaussWindow: 1, residualConnections: 1, sigma: 0.1, dilationIterations: 7, precision: 0.845, recall: 0.684, f1_Score: 0.756
[INFO] FPS: 60.50
gaussWindow: 7, residualConnections: 5, sigma: 0.9, dilationIterations: 3, precision: 0.923, recall: 0.652, f1_Score: 0.765
# 190707_210056: [L]
[INFO] Starting 3 iterations
[INFO] Using bbox frames 1 to 243
[INFO] Cached 339 frames with shape x-1920 y-1080
[INFO] FPS: 56.03
gaussWindow: 7, residualConnections: 3, sigma: 0.9, dilationIterations: 7, precision: 0.996, recall: 0.705, f1_Score: 0.826
[INFO] FPS: 68.28
gaussWindow: 1, residualConnections: 1, sigma: 0.5, dilationIterations: 3, precision: 0.116, recall: 0.711, f1_Score: 0.200
[INFO] FPS: 57.22
gaussWindow: 7, residualConnections: 3, sigma: 0.9, dilationIterations: 7, precision: 0.996, recall: 0.705, f1_Score: 0.826
