import cv2, numpy as np
from numba import njit

@njit
def compute_reward_numba(cx, center, line_found, distance_cm):
    if distance_cm is not None and distance_cm < 20:
        return -5.0
    if not line_found:
        return -2.0
    dev = abs(cx - center) / center
    r = max(1 - dev, -1.0)
    if dev < 0.1: r += 0.5
    elif dev > 0.9: r -= 0.5
    return r

def get_reward(frame, distance_cm=None, debug=False):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    roi = gray[h//2:]
    _, binary = cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY_INV)
    m = cv2.moments(binary)

    line_found = m["m00"] != 0
    cx = m["m10"] / m["m00"] if line_found else w / 2
    center = w / 2

    reward = compute_reward_numba(cx, center, line_found, distance_cm if distance_cm is not None else -1)

    if debug:
        print(f"Distance: {distance_cm} cm | Line: {'Yes' if line_found else 'No'} | "
              f"CX: {cx:.2f} | Dev: {abs(cx - center) / center:.2f} | Reward: {reward:.2f}")

    return reward