import cv2
import numpy as np

def get_state_from_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    roi = gray[height//2:, :]
    _, binary = cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY_INV)

    moments = cv2.moments(binary)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        norm_cx = cx / width
    else:
        norm_cx = 0.5

    area = moments["m00"]
    total_area = roi.shape[0] * roi.shape[1]
    line_area_ratio = area / total_area if area != 0 else 0.1

    # 좌/우 밝은 영역 비율 분석 (장애물 탐지용)
    left_half = roi[:, :width//2]
    right_half = roi[:, width//2:]

    left_white = cv2.countNonZero(left_half)
    right_white = cv2.countNonZero(right_half)
    total_white = left_white + right_white + 1e-5  # 0 나누기 방지

    left_ratio = left_white / total_white
    right_ratio = right_white / total_white

    return np.array([norm_cx, line_area_ratio, left_ratio, right_ratio], dtype=np.float32)
