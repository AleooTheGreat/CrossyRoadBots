import random


class SARSAAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.Q = {}

    def _ensure_state(self, state):
        """Initialize state in Q-table if not present."""
        if state not in self.Q:
            self.Q[state] = {a: 0.0 for a in self.actions}

    def select_action(self, state):
        self._ensure_state(state)

        # ε-greedy policy
        if random.random() < self.epsilon:
            return random.choice(self.actions)

        return max(self.Q[state], key=self.Q[state].get)

    def update(self, s, a, r, s_next, a_next):
        self._ensure_state(s)
        self._ensure_state(s_next)

        q = self.Q[s][a]
        q_next = self.Q[s_next][a_next]

        self.Q[s][a] = q + self.alpha * (r + self.gamma * q_next - q)
