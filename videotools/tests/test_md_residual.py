import unittest
from . import md_residual
import numpy as np


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.random_bbox_size = 1920


    def test_bb_intersection_over_union_no_intersection(self):
        """
        Test that the different cases generate meaningful results.
        bounding boxes are in the (x, y, w, h) format and converted to (x1, y1, x2, y2)
        :return:
        """
        bb_ground_truth = md_residual.xywh_to_x1y1x2y2((10, 20, 150, 100))
        bb_no_intersection = md_residual.xywh_to_x1y1x2y2((1000, 1000, 1000, 1000))
        bb_full_intersection = md_residual.xywh_to_x1y1x2y2((12, 22, 15, 10))
        bb_identical = md_residual.xywh_to_x1y1x2y2((10, 20, 150, 100))
        bb_partial = md_residual.xywh_to_x1y1x2y2((0, 0, 100, 120))

        self.assertEqual(md_residual.bb_intersection_over_union(bb_ground_truth, bb_no_intersection), 0)
        self.assertEqual(md_residual.bb_intersection_over_union(bb_ground_truth, bb_full_intersection), 0.01)
        self.assertEqual(md_residual.bb_intersection_over_union(bb_ground_truth, bb_identical), 1)
        self.assertEqual(md_residual.bb_intersection_over_union(bb_ground_truth, bb_partial), 0.5)

    def test_xywh_to_x1y1x2y2(self):
        bbox = tuple((np.random.randint(self.random_bbox_size) for _ in range(4)))
        self.assertEqual(md_residual.xywh_to_x1y1x2y2(bbox), (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))

    def test_centered_bbox(self):
        bbox = tuple((np.random.randint(self.random_bbox_size) for _ in range(4)))
        self.assertEqual(md_residual.centered_bbox(bbox), (bbox[0] + bbox[2]//2, bbox[1] + bbox[3]//2, bbox[2], bbox[3]))


if __name__ == '__main__':
    unittest.main()
