import cv2
import torch
import os
import time
import pymysql
from datetime import datetime
from ultralytics import YOLO

class YOLOWebcam:
    def __init__(self, model_path='best.pt'):
        print("웹캠 초기화 시작")
        
        self.model = YOLO(model_path)  
        print("모델 로드 완료")
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("웹캠을 열 수 없습니다. 카메라가 연결되어 있는지 확인하세요.")
        print("웹캠 연결 성공")
        print("모델 클래스 목록:", self.model.names)

    def detect_and_render(self, frame):
        results = self.model(frame)
        img = frame.copy()

        if results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]

                if class_name != 'weed':
                    continue  # 잡초 이외의 객체는 무시

                # 바운딩 박스 좌표
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                # 신뢰도 0.6 이상일때만 :
                if conf >= 0.6 :
                    # 박스 및 클래스 이름 그리기
                    label = f'{class_name} {conf:.2f}'
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return img

    def run(self):
        """웹캠을 통해 객체 탐지 및 이미지 캡처를 실행"""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("프레임을 읽을 수 없습니다. 웹캠 연결 상태를 확인하세요.")
                    break

                detected_frame = self.detect_and_render(frame)

                cv2.imshow('YOLO Object Detection', detected_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    yolo_webcam = YOLOWebcam(model_path='best.pt')
    yolo_webcam.run()