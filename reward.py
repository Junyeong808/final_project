import cv2
import numpy as np

def get_reward(frame, distance_cm=None):
    """
    라인 중심에서의 거리 + 장애물 감지를 통한 보상 결정
    """

    # 초음파 기준으로 너무 가까우면 즉시 패널티
    if distance_cm is not None and distance_cm < 20:  # 초음파 거리 기준, 20cm 이하
        return -5  # 장애물에 너무 가까운 경우 패널티
    
    # 라인 중심 보상
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = gray[frame.shape[0]//2:, :]  # 하단 영역 (라인이 있을 가능성이 높은 하단)
    _, binary = cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY_INV)
    moments = cv2.moments(binary)

    if moments["m00"] != 0:  # 라인을 감지한 경우
        cx = int(moments["m10"] / moments["m00"])
        width = frame.shape[1]
        center = width // 2
        deviation = abs(cx - center) / (width / 2)  # 라인 중심과의 편차 계산
        reward = 1 - deviation
        reward = max(reward, -1)  # 최소 보상 -1

        # 라인이 정확히 중앙에 있을 때 보상 강화
        if deviation < 0.1:  # 중심에서 매우 가까운 경우
            reward += 0.5
        # 라인이 많이 벗어난 경우 패널티
        elif deviation > 0.9:  # 중심에서 멀리 벗어난 경우
            reward -= 0.5

        return reward
    else:
        return -2  # 라인 놓친 경우