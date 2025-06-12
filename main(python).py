import cv2, os, csv, time, serial, matplotlib.pyplot as plt
from dqn_agent import Agent
from camera_utils import get_state_from_frame as gsf
from reward import get_reward as gr

def read_dist(ser):
    if ser.in_waiting:
        try:
            line = ser.readline().decode().strip()
            return float(line[5:]) if line.startswith("DIST:") else None
        except: return None

def main():
    os.makedirs("model", exist_ok=True); os.makedirs("logs", exist_ok=True)
    model_path, log_path = "model/dqn_model", "logs/rewards.csv"
    if not os.path.exists(log_path):
        with open(log_path,'w',newline='') as f: csv.writer(f).writerow(["Episode","Total_Reward"])

    agent = Agent(4,4)
    agent.model.load_weights(model_path) if os.path.exists(model_path) else None
    ser, cap = serial.Serial('/dev/ttyUSB0',9600,timeout=1), cv2.VideoCapture(0)
    cmds = {0:"LEFT\n",1:"FORWARD\n",2:"RIGHT\n",3:"BACKWARD\n"}
    rh, ep, step, learn_every = [], 0, 0, 4

    try:
        while True:
            ep += 1; R = s = 0
            while (ret:=cap.read())[0]:
                d = read_dist(ser)
                if d is None: continue
                if d>20: ser.write(b"FORWARD\n"); time.sleep(0.1); continue
                st = gsf(ret[1]); a = agent.act(st); ser.write(cmds[a].encode())
                ns, r = gsf(ret[1]), gr(ret[1], d)
                done = r == -5 or s >= 200
                agent.mem.store((st,a,r,ns,done))
                if (step:=step+1) % learn_every == 0: agent.learn()
                R, s = R+r, s+1
                cv2.putText(ret[1], f"EP {ep} | R: {R:.2f}", (10,30),0,0.6,(0,255,0),2)
                cv2.imshow("Cam", ret[1])
                if (cv2.waitKey(1) & 0xFF) in {27, ord('q')} or done: break
            with open(log_path,'a',newline='') as f: csv.writer(f).writerow([ep,R])
            rh.append(R); print(f"EP {ep} 완료 | Total Reward: {R}")
            if ep%10==0:
                agent.model.save(f"model/dqn_model_ep{ep}.h5")
                plt.plot(rh); plt.xlabel('Episode'); plt.ylabel('Reward'); plt.grid()
                plt.savefig(f"logs/reward_plot_ep{ep}.png"); plt.close()
    except KeyboardInterrupt: pass
    finally:
        cap.release(); cv2.destroyAllWindows(); ser.close()
        agent.model.save(model_path)
        print("종료 및 모델 저장 완료")

if __name__=="__main__": main()