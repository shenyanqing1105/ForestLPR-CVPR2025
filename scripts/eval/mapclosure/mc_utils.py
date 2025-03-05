# Code based on ScanContext implementation: https://github.com/irapkaist/scancontext/blob/master/python/make_sc_example.py
# Partially vectorized implementation by Jacek Komorowski: https://github.com/jac99/Egonn/blob/main/third_party/scan_context/scan_context.py

import numpy as np
import numpy_indexed as npi
import cv2
class HBSTNode:
    def __init__(self):
        self.left = None
        self.right = None
        self.descriptors = []
        self.indices = []

class HBST:
    def __init__(self, max_leaf_capacity=100, depth=256):
        self.root = HBSTNode()
        self.max_leaf_capacity = max_leaf_capacity
        self.depth = depth

    def insert(self, descriptor, image_index, node=None, bit_index=0):
        if node is None:
            node = self.root

        # check bit_index 
        if bit_index >= len(descriptor) * 8:
            node.descriptors.append(descriptor)
            node.indices.append(image_index)
            return

        if bit_index >= self.depth or len(node.descriptors) < self.max_leaf_capacity:
            node.descriptors.append(descriptor)
            node.indices.append(image_index)
        else:
            byte_index = bit_index // 8
            bit_in_byte = bit_index % 8
            bit_value = (descriptor[byte_index] >> (7 - bit_in_byte)) & 1

            if bit_value == 0:
                if node.left is None:
                    node.left = HBSTNode()
                self.insert(descriptor, image_index, node.left, bit_index + 1)
            else:
                if node.right is None:
                    node.right = HBSTNode()
                self.insert(descriptor, image_index, node.right, bit_index + 1)

    def search(self, descriptor, node=None, bit_index=0):
        if node is None:
            node = self.root

        if bit_index >= len(descriptor) * 8:
            return node.descriptors, node.indices

        if bit_index >= self.depth:
            return node.descriptors, node.indices

        byte_index = bit_index // 8
        bit_in_byte = bit_index % 8
        bit_value = (descriptor[byte_index] >> (7 - bit_in_byte)) & 1

        if bit_value == 0 and node.left is not None:
            return self.search(descriptor, node.left, bit_index + 1)
        elif bit_value == 1 and node.right is not None:
            return self.search(descriptor, node.right, bit_index + 1)
        else:
            return node.descriptors, node.indices

    def hamming_distance(self, desc1, desc2):
        """ Hamming distance."""
        return np.count_nonzero(desc1 != desc2)

    def find_closest(self, descriptor):
        """search and find the closest descriptor and index"""
        candidates, indices = self.search(descriptor)
        if not candidates:
            return None, None, float('inf')

        min_distance = float('inf')
        closest_desc = None
        closest_index = None
        for cand, index in zip(candidates, indices):
            distance = self.hamming_distance(descriptor, cand)
            if distance < min_distance:
                min_distance = distance
                closest_desc = cand
                closest_index = index

        return closest_desc, closest_index, min_distance


def extract_orb(orb, image):
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def match_features(des1, des2):
    """BFMatcher."""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def ransac_homography(kp1, kp2, matches):
    """RANSAC."""
    if len(matches) < 4:
        return None, 0

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    inliers = np.sum(mask) if mask is not None else 0
    return H, inliers