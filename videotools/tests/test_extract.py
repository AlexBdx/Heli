import video_tools.extract as e
import unittest
import numpy as np


class TestExtract(unittest.TestCase):
    def setUp(self):
        self.s = (224, 224) # Size of the matrix to use for testing, MUST be EVEN here
        self.f = (40, 40)  # Size of the "1" pack that represent a mask
    
    # Test the mask_center_shift function
    def test_mask_center_shift_centered(self):
        a = np.zeros(self.s, dtype=np.uint8)
        a[self.s[0]//2-self.f[0]//2:self.s[0]//2+self.f[0]//2, self.s[1]//2-self.f[0]//2:self.s[1]//2+self.f[0]//2] = 255
        expected_dist = 0
        self.assertEqual(expected_dist, e.mask_center_shift(np.stack([a]*4, axis=2)))

    def test_mask_center_shift_up(self):
        b = np.zeros(s, dtype=np.uint8)
        b[:self.f[0], self.s[1]//2-self.f[1]//2:self.s[1]//2+self.f[1]//2] = 255
        expected_dist = self.s[0]//2 - self.f[0]//2
        self.assertEqual(expected_dist, e.mask_center_shift(np.stack([b]*4, axis=2)))

    def test_mask_center_shift_down(self):
        c = np.zeros(s, dtype=np.uint8)
        c[self.s[0]//2-self.f[0]//2:self.s[0]//2+self.f[0]//2, :self.f[1]] = 255
        expected_dist = self.s[1]//2 - self.f[1]//2
        self.assertEqual(expected_dist, e.mask_center_shift(np.stack([c]*4, axis=2)))

    def test_mask_center_shift_diagonal(self):
        d = np.zeros(s, dtype=np.uint8)
        d[:self.f[0], :self.f[1]] = 255
        expected_dist = int(np.linalg.norm([self.s[0]//2 - self.f[0]//2, self.s[1]//2 - self.f[1]//2]))
        self.assertEqual(expected_dist, e.mask_center_shift(np.stack([d]*4, axis=2)))


if __name__ == '__main__':
    unittest.main()
