import cv2, numpy as np

def get_state_from_frame(f):
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    h, w = g.shape; roi = g[h//2:]
    _, b = cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY_INV)
    m = cv2.moments(b)
    cx = m["m10"] / m["m00"] if m["m00"] else w / 2
    a = m["m00"]; t = roi.size
    l, r = cv2.countNonZero(roi[:, :w//2]), cv2.countNonZero(roi[:, w//2:])
    tw = l + r + 1e-5
    return np.array([cx / w, a / t if a else 0.1, l / tw, r / tw], dtype=np.float32)