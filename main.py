import cv2
import numpy as np
import os
import csv
import time
import matplotlib.pyplot as plt
import serial
from dqn_agent import DQNAgent
from camera_utils import get_state_from_frame
from reward import get_reward

def read_ultrasonic_distance(ser):
    """
    Read Ultrasonid Distance for Serial Port
    Expecte Value : "DIST:23.4\n"
    """
    try:
        line = ser.readline().decode().strip()
        if line.startswith("DIST:"):
            distance_str = line.split(":")[1]
            return float(distance_str)
    except Exception as e:
        print("Ultrasonic distance read error :", e)
    return None

# DQN기반 강화학습 agent 초기화 부분
state_size = 4  # 4차원 벡터(위치, 속도, 각도 변화율)
action_size = 4  # can통신으로 보낼 행동 제어문 4가지(forward, backward, right, left)
model_path = "model/dqn_model" # 저장할 모델 경로
log_path = "logs/rewards.csv"  # 저장할 로그기록의 경로
os.makedirs("model", exist_ok=True) # "model" directory가 없다면 만들어서 저장
os.makedirs("logs", exist_ok=True)  # "logs" directory가 없다면 만들어서 저장

# DQN 에이전트 초기화
agent = DQNAgent(state_size=4, action_size=4)

# 모델 불러오기
if os.path.exists(model_path):
    agent.model.load_weights(model_path)
    print("모델 불러오기")

# 보상 로그 초기화
if not os.path.exists(log_path):
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Total_Reward"])

# RS-485 시리얼 포트 설정
ser = serial.Serial(
    port='/dev/ttyUSB0',  # 포트는 실제 환경에 맞게 설정 (예: COM3, /dev/ttyUSB0 등)
    baudrate=9600,
    timeout=1
)

# 액션 번호에 대응하는 RS-485 명령어 설정
action_commands = {
    0: "LEFT\n",
    1: "FORWARD\n",
    2: "RIGHT\n",
    3: "BACKWARD\n"
}

# 카메라 연결
cap = cv2.VideoCapture(0)
episode = 0
reward_history = []

print("학습 시작 (q 키로 종료)")

should_exit = False

try:
    while not should_exit:
        total_reward = 0
        step = 0
        episode += 1

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            distance_cm = read_ultrasonic_distance(ser)
            if distance_cm is None:
                print("초음파 거리 읽기 오류")
                continue

            # 거리 기준으로 행동 결정
            if distance_cm > 20:
                # 물체와 멀리 떨어졌으면 무조건 직진
                command = "FORWARD\n"
                ser.write(command.encode())
                print("거리 충분 → 직진 명령 전송")
                time.sleep(0.1)  # 살짝 멈춰서 제어 안정화
                continue

            else:
                state = get_state_from_frame(frame)
                action = agent.act(state)

                actions = ["Left", "Forward", "Right", "Backward"]
                print(f"[EP {episode}] 예측 행동: {actions[action]}")

                # RS-485 명령 전송
                command = action_commands[action]
                ser.write(command.encode())  # 문자열을 바이트로 변환하여 전송
                print(f"RS-485로 전송된 명령: {command.strip()}")

                next_state = get_state_from_frame(frame)
                reward = get_reward(frame, distance_cm)  # 보상 계산
                done = False
                if reward == -5 or step >= 200:
                    done = True

                # 경험 저장 (ExperienceReplay 사용)
                agent.memory.store((state, action, reward, next_state, done))
                agent.learn()

                state = next_state
                total_reward += reward
                step += 1

            # 영상에 보상 점수 표시
            cv2.putText(frame, f"EP {episode} | Reward: {total_reward}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Camera", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27 :
                should_exit = True
                break
            if done:
                break

        # 보상 로그 저장
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, total_reward])

        reward_history.append(total_reward)
        print(f"Episode {episode} 완료 | 총 보상: {total_reward}")

        # 모델 주기 저장 (10회마다)
        if episode % 10 == 0 :
            agent.model.save(f"model/dqn_model_ep{episode}.h5")
            print("모델 저장")

            plt.figure(figsize=(8,4))
            plt.plot(reward_history, label='Total Reward')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('Reward Over Episodes')
            plt.legend()
            plt.grid()
            plt.savefig(f"logs/reward_plot_ep{episode}.png")
            plt.close()

# 무한 루프를 사용자가 중단하기 위해 넣어줌
except KeyboardInterrupt:
    print("수동 중단")

finally:
    # 종료 처리
    cap.release()  # 카메라 리소스 해제
    cv2.destroyAllWindows()  # OpenCV 윈도우 닫기
    ser.close()
    agent.model.save(model_path)  # 학습이 완료된 모델 저장
    print("작업 완료. 모델 저장")