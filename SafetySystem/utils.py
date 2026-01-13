import cv2
import numpy as np
import math

def refine_mask(mask):
    """
    Aggressive Refinement to remove small noise and debris.
    """
    # Use a larger kernel (7x7 instead of 5x5) to 'melt' small objects
    kernel = np.ones((7, 7), np.uint8)

    # 1. Stronger Erosion: removes small objects (noise) from shelves
    eroded = cv2.erode(mask, kernel, iterations=2)

    # 2. Dilation: merges human fragments that survived erosion
    refined = cv2.dilate(eroded, kernel, iterations=3)

    return refined

def get_person_detections(refined_mask):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(refined_mask, connectivity=8)
    
    detections = []
    # Get image height to define a simple logical ROI
    img_h = refined_mask.shape[0]

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        
        # 1. Logical ROI Filter: Ignore objects 'floating' too high (e.g., on shelves)
        # We only consider objects whose base is in the bottom 50% of the image
        if (y + h) < (img_h * 0.5):
            continue

        # 2. Solidity Filter: Humans are solid objects, noise is often hollow
        solidity = area / float(w * h)
        
        # 3. Combined Filter: Minimum Area + Verticality + Solidity
        if area > 4500 and (h / float(w)) > 1.2 and solidity > 0.4:
            detections.append((x, y, w, h))
                
    return detections

