"""
ConnectomeReservoir 模型（可导入模块）
"""
import numpy as np

class ConnectomeReservoir:
    def __init__(self, W_rec, n_in, n_out, input_scaling=1.0, leaky_rate=0.3):
        self.N = W_rec.shape[0]
        self.W_rec = W_rec
        self.leaky_rate = leaky_rate
        self.state = np.zeros(self.N)
        self.n_in = n_in
        W_in = np.zeros((self.N, n_in))
        for i in range(n_in):
            targets = np.random.choice(self.N, 30, replace=False)
            W_in[targets, i] = input_scaling * (2 * np.random.random(30) - 1)
        self.W_in = W_in
        self.W_out = np.zeros((n_out, self.N))

    def reset(self):
        self.state.fill(0)

    def step(self, u):
        u = np.asarray(u).flatten()
        drive = self.W_in @ u + self.W_rec @ self.state
        self.state = (1 - self.leaky_rate) * self.state + self.leaky_rate * np.tanh(drive)
        return self.state

    def collect(self, U, discard=0):
        n = U.shape[0]
        S = np.zeros((n - discard, self.N))
        self.reset()
        for i in range(n):
            self.step(U[i])
            if i >= discard:
                S[i - discard] = self.state
        return S

    def fit_ridge(self, U, Y, discard=0, alpha=1e-6):
        X = self.collect(U, discard=discard)
        Y = Y[discard:] if len(Y) > discard else Y
        m = min(len(X), len(Y)); X, Y = X[:m], Y[:m]
        self.W_out = np.linalg.solve(X.T @ X + alpha * np.eye(self.N), X.T @ Y).T

    def predict(self, U, discard=0):
        return self.collect(U, discard=discard) @ self.W_out.T
