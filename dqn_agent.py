import numpy as np
import tensorflow as tf
from keras import layers
import random
import collections
from collections import deque

# ExperienceReplay 클래스 정의 (메모리 관리)
class ExperienceReplay:

    def __init__(self, max_size):
        self.memory = collections.deque(maxlen=max_size)
        self.max_size = max_size
    
    def store(self, experience):
        self.memory.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def size(self):
        return len(self.memory)

# DQN 모델 정의
class DQN(tf.keras.Model):

    def __init__(self, state_size, action_size):

        super().__init__()

        # 제1은닉층
        self.dense1 = layers.Dense(64)
        self.leaky_relu1 = layers.LeakyReLU(alpha=0.01) # RuLU 대신 LeakyReLU를 쓴 이유 : -값을 받는 뉴런이 죽는 문제를 해결

        # 제2은닉층
        self.dense2 = layers.Dense(64)
        self.leaky_relu2 = layers.LeakyReLU(alpha=0.01)

        # 출력층
        self.out = layers.Dense(action_size)

    def call(self, x):
        x = self.dense1(x)
        x = self.leaky_relu1(x)
        x = self.dense2(x)
        x = self.leaky_relu2(x)
        return self.out(x)

# DQN 에이전트 클래스
class DQNAgent:

    def __init__(self, state_size, action_size):  # main.py의 state_size, action_size
        self.model = DQN(state_size, action_size)  # 
        self.target_model = DQN(state_size, action_size)
        self.target_model.set_weights(self.model.get_weights()) # 가중치 설정
        
        # ExperienceReplay 클래스를 사용하도록 변경(gamma, epsilon은 hyperpharameter)
        self.memory = ExperienceReplay(max_size=2000)  # deque 대신 ExperienceReplay로 변경
        self.batch_size = 64
        self.gamma = 0.95  # 감가율 : 범위, 0.0 ~ 1.0, 미래 보상에 가중치를 얼마나 반영할 것인가 결정하는 값, 0이면 즉시보상만 고려, 1이면 미래 보상도 현재와 동등하게 고려함
        self.epsilon = 1.0 # 무작위 행동을 할 확률을 결정하는 값
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.optimizer = tf.keras.optimizers.Adam(1e-3)
        self.state_size = state_size  # 4
        self.action_size = action_size  # 5
        self.steps = 0

    def act(self, state):

        # 상태의 차원을 (1, state_size)로 맞춤
        state = np.reshape(state, [1, self.state_size])  # (1, 2) 형태로 reshape

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q = self.model(state)
        return np.argmax(q[0])

    def learn(self):

        if self.memory.size() < self.batch_size:
            return
        
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        next_qs = self.target_model(next_states)
        target_qs = rewards + self.gamma * np.max(next_qs, axis=1) * (1 - dones)

        with tf.GradientTape() as tape:
            qs = self.model(states)
            qs = tf.reduce_sum(qs * tf.one_hot(actions, self.action_size), axis=1)
            loss = tf.reduce_mean(tf.keras.losses.Huber()(target_qs, qs))

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # epsilon을 점차적으로 줄여줌(Hyperpharameter)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 주기적으로 타겟 모델을 업데이트
        if self.steps % 100 == 0:
            self.target_model.set_weights(self.model.get_weights())
        self.steps += 1
        
    def save_model(self, model_path):
        try:
            self.model.save(model_path)
            print("모델 저장 완료")
        except Exception as e:
            print("모델 저장 실패:", e)