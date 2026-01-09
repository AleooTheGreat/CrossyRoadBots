import random
import pickle
from typing import Any, Dict, List


class DoubleQAgent:
   
    def __init__(self, actions: List[str], alpha: float = 0.25, gamma: float = 0.99, epsilon: float = 1.0,
                 optimistic_init: float = 0.0):
        self.actions = list(actions)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.optimistic_init = float(optimistic_init)

        self.Q1: Dict[Any, Dict[str, float]] = {}
        self.Q2: Dict[Any, Dict[str, float]] = {}

    def _ensure_state(self, Q: Dict[Any, Dict[str, float]], state: Any) -> None:
        if state not in Q:
            Q[state] = {a: self.optimistic_init for a in self.actions}

    def q_sum(self, state: Any) -> Dict[str, float]:
        self._ensure_state(self.Q1, state)
        self._ensure_state(self.Q2, state)
        return {a: self.Q1[state][a] + self.Q2[state][a] for a in self.actions}

    def select_action(self, state: Any) -> str:

        if random.random() < self.epsilon:
            return random.choice(self.actions)

        qsum = self.q_sum(state)
        max_v = max(qsum.values())
        best = [a for a, v in qsum.items() if v == max_v]
        return random.choice(best)

    def update(self, s: Any, a: str, r: float, s_next: Any, done: bool) -> None:
        self._ensure_state(self.Q1, s)
        self._ensure_state(self.Q2, s)
        self._ensure_state(self.Q1, s_next)
        self._ensure_state(self.Q2, s_next)

        if done:
            target = r
            if random.random() < 0.5:
                self.Q1[s][a] += self.alpha * (target - self.Q1[s][a])
            else:
                self.Q2[s][a] += self.alpha * (target - self.Q2[s][a])
            return

        if random.random() < 0.5:
            a_star = max(self.Q1[s_next], key=self.Q1[s_next].get)
            target = r + self.gamma * self.Q2[s_next][a_star]
            self.Q1[s][a] += self.alpha * (target - self.Q1[s][a])
        else:
            a_star = max(self.Q2[s_next], key=self.Q2[s_next].get)
            target = r + self.gamma * self.Q1[s_next][a_star]
            self.Q2[s][a] += self.alpha * (target - self.Q2[s][a])

    def set_epsilon(self, eps: float) -> None:
        self.epsilon = float(eps)

    def set_alpha(self, alpha: float) -> None:
        self.alpha = float(alpha)

    def save(self, path: str, meta: dict | None = None) -> None:
        payload = {
            "Q1": self.Q1,
            "Q2": self.Q2,
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "actions": self.actions,
            "optimistic_init": self.optimistic_init,
            "meta": meta or {},
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def load(self, path: str) -> dict:
        with open(path, "rb") as f:
            ckpt = pickle.load(f)
        self.Q1 = ckpt["Q1"]
        self.Q2 = ckpt["Q2"]
        self.epsilon = ckpt.get("epsilon", self.epsilon)
        self.alpha = ckpt.get("alpha", self.alpha)
        self.gamma = ckpt.get("gamma", self.gamma)
        self.actions = ckpt.get("actions", self.actions)
        self.optimistic_init = ckpt.get("optimistic_init", self.optimistic_init)
        return ckpt.get("meta", {})
