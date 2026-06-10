import cv2
import numpy as np
from skimage.feature import hog

def extract_hog_features(images, resize=(28, 28)):
    features = []
    for img in images:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, resize)
        fd = hog(img, orientations=9, pixels_per_cell=(8,8),
                 cells_per_block=(2,2), visualize=False)
        features.append(fd)
    return np.array(features)