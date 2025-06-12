import numpy as np, tensorflow as tf, random, matplotlib.pyplot as plt
from keras import layers; from collections import deque

class Replay:
    def __init__(s, n): s.m = deque(maxlen=n)
    def store(s, x): s.m.append(x)
    def sample(s, k): return random.sample(s.m, k) if len(s.m) >= k else []
    def __len__(s): return len(s.m)

class DQN(tf.keras.Model):
    def __init__(s, ins, outs):
        super().__init__()
        s.fwd = tf.keras.Sequential([
            layers.Dense(64), layers.LeakyReLU(0.01),
            layers.Dense(64), layers.LeakyReLU(0.01),
            layers.Dense(outs)
        ])
    def call(s, x): return s.fwd(x)

class Agent:
    def __init__(s, ss, asz):
        s.model, s.tgt = DQN(ss, asz), DQN(ss, asz)
        s.tgt.set_weights(s.model.get_weights())
        s.mem, s.opt = Replay(2000), tf.keras.optimizers.Adam(1e-3)
        s.batch, s.gamma, s.eps, s.emin, s.edecay = 64, 0.95, 1.0, 0.01, 0.995
        s.ss, s.asz, s.steps, s.losses = ss, asz, 0, []

    def act(s, st):
        st = np.reshape(st, (1, s.ss))
        return random.randint(0, s.asz - 1) if np.random.rand() < s.eps else np.argmax(s.model(st)[0])

    def learn(s):
        if len(s.mem) < s.batch: return
        b = np.array(s.mem.sample(s.batch), dtype=object).T
        st, a, r, nst, done = map(np.array, b)
        q_target = r + s.gamma * np.max(s.tgt(nst), axis=1) * (1 - done)
        with tf.GradientTape() as t:
            q_pred = tf.reduce_sum(s.model(st) * tf.one_hot(a, s.asz), axis=1)
            loss = tf.reduce_mean(tf.keras.losses.Huber()(q_target, q_pred))
        s.opt.apply_gradients(t.gradient(loss, s.model.trainable_variables))
        if s.eps > s.emin: s.eps *= s.edecay
        if s.steps % 100 == 0: s.tgt.set_weights(s.model.get_weights())
        s.steps += 1; s.losses.append(loss.numpy())

    def save(s, path): s.model.save(path)
    def plot_loss(s, save=None):
        if not s.losses: return
        plt.plot(s.losses, label="Loss"); plt.xlabel("Steps"); plt.ylabel("Loss")
        plt.title("DQN Training Loss"); plt.grid(); plt.legend()
        if save: plt.savefig(save)
        else: plt.show(); plt.close()