# rl_agent.py
import random
from collections import defaultdict

class QLearningAgent:
    def __init__(self):
        self.action_size = 5
        self.alpha = 0.2       # 학습률
        self.gamma = 0.98      # 미래 보상 반영 정도

        self.epsilon = 1.0     # 초기 탐험 확률
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.999
        self.q_table = defaultdict(lambda: [0.0] * self.action_size)

    # 학습 중 행동 선택: 탐험(epsilon) + 활용(Q값 최대)
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        q_values = self.q_table[state]
        max_q = max(q_values)
        best_actions = [i for i, q in enumerate(q_values) if q == max_q]
        return random.choice(best_actions)

    # 데모 실행용: 탐험 없이 현재 Q-table에서 가장 좋은 행동 선택
    def best_action(self, state):
        q_values = self.q_table[state]
        max_q = max(q_values)
        best_actions = [i for i, q in enumerate(q_values) if q == max_q]
        return random.choice(best_actions)

    # Q-learning 업데이트
    def learn(self, state, action, reward, next_state, done):
        current_q = self.q_table[state][action]
        next_max_q = 0 if done else max(self.q_table[next_state])
        target = reward + self.gamma * next_max_q
        self.q_table[state][action] += self.alpha * (target - current_q)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

    def get_q_values(self, state):
        return self.q_table[state]
