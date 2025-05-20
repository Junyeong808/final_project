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

# ------------------------- 초기 설정 -------------------------
STATE_SIZE = 4
ACTION_SIZE = 4
MODEL_PATH = "model/dqn_model"
LOG_PATH = "logs/rewards.csv"
os.makedirs("model", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ------------------------- 시리얼 설정 -------------------------
def setup_serial(port='/dev/ttyUSB0', baudrate=9600):
    try:
        return serial.Serial(port, baudrate, timeout=1)
    except Exception as e:
        print(f"시리얼 연결 실패: {e}")
        return None

def read_ultrasonic_distance(ser):
    try:
        line = ser.readline().decode().strip()
        if line.startswith("DIST:"):
            return float(line.split(":")[1])
    except Exception as e:
        print("초음파 거리 읽기 오류:", e)
    return None

# ------------------------- 제어 명령 -------------------------
ACTION_COMMANDS = {
    0: "LEFT\n",
    1: "FORWARD\n",
    2: "RIGHT\n",
    3: "BACKWARD\n"
}

def send_action(ser, action):
    cmd = ACTION_COMMANDS.get(action, "STOP\n")
    ser.write(cmd.encode())
    print(f"전송된 명령: {cmd.strip()}")

# ------------------------- 보상 로그 -------------------------
def init_reward_log():
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "Total_Reward"])

def log_reward(episode, reward, reward_history):
    with open(LOG_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([episode, reward])
    reward_history.append(reward)

    if episode % 10 == 0:
        plt.figure()
        plt.plot(reward_history)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Reward Trend')
        plt.grid()
        plt.savefig(f"logs/reward_plot_ep{episode}.png")
        plt.close()

# ------------------------- 메인 루프 -------------------------
def main():
    cap = cv2.VideoCapture(0)
    ser = setup_serial()
    if not ser:
        return

    agent = DQNAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE)
    if os.path.exists(MODEL_PATH):
        agent.model.load_weights(MODEL_PATH)
        print("모델 불러오기 완료")

    init_reward_log()
    reward_history = []
    episode = 0
    should_exit = False

    try:
        while not should_exit:
            total_reward = 0
            step = 0
            episode += 1
            done = False

            while not done:
                ret, frame = cap.read()
                if not ret:
                    break

                distance = read_ultrasonic_distance(ser)
                if distance is None:
                    continue

                if distance > 20:
                    send_action(ser, 1)  # FORWARD
                    print("거리 충분 → 직진")
                    time.sleep(0.1)
                    continue

                # 학습 루프
                state = get_state_from_frame(frame)
                action = agent.act(state)
                send_action(ser, action)

                next_state = get_state_from_frame(frame)
                reward = get_reward(frame, distance)
                done = reward == -5 or step >= 200

                agent.memory.store((state, action, reward, next_state, done))
                agent.learn()

                total_reward += reward
                step += 1

                # 디스플레이
                cv2.putText(frame, f"EP {episode} | Reward: {total_reward}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Camera", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    should_exit = True
                    break

            log_reward(episode, total_reward, reward_history)
            print(f"[EP {episode}] 완료 | 총 보상: {total_reward}")

            if episode % 10 == 0:
                agent.model.save(f"model/dqn_model_ep{episode}.h5")
                print("모델 저장 완료")

    except KeyboardInterrupt:
        print("사용자 중단 감지")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        ser.close()
        agent.model.save(MODEL_PATH)
        print("모델 저장 및 종료")

if __name__ == "__main__":
    main()
