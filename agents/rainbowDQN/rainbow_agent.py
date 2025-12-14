import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import random
import collections
from agents.base_agent import BaseAgent


# ============================================================
# 1. Prioritized Multi-Step Replay Buffer (PER + N-step)
# ============================================================

class PrioritizedNStepBuffer:
    def __init__(self, capacity, state_dim, n_step=3, gamma=0.99, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.ptr = 0
        self.full = False

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.priorities = np.ones(capacity, dtype=np.float32)

        self.n_step = n_step
        self.gamma = gamma
        self.buffer = collections.deque(maxlen=n_step)

        self.alpha = alpha
        self.beta = beta

    def store(self, s, s2, a, r, done):
        # Store transition in temporary n-step buffer
        self.buffer.append((s, s2, a, r, done))
        if len(self.buffer) < self.n_step:
            return

        # Calculate N-step return
        R = 0
        discount = 1
        for (_, _, _, r_i, d_i) in self.buffer:
            R += r_i * discount
            discount *= self.gamma
            if d_i:
                break

        # Get the actual state (s0) and the n-step next state (s_n)
        s0, _, a0, _, _ = self.buffer[0]
        _, s_n, _, _, d_n = self.buffer[-1]

        idx = self.ptr
        self.states[idx] = s0
        self.next_states[idx] = s_n
        self.actions[idx] = a0
        self.rewards[idx] = R
        self.dones[idx] = float(d_n)
        # New experiences get max priority
        self.priorities[idx] = self.priorities.max() if self.full or self.ptr > 0 else 1.0

        self.ptr = (self.ptr + 1) % self.capacity
        if self.ptr == 0: self.full = True

    def sample(self, batch_size):
        max_mem = self.capacity if self.full else self.ptr
        probs = self.priorities[:max_mem] ** self.alpha
        probs /= probs.sum()

        idx = np.random.choice(max_mem, batch_size, p=probs)
        weights = (max_mem * probs[idx]) ** (-self.beta)
        weights /= weights.max()

        return (
            self.states[idx], self.next_states[idx], self.actions[idx],
            self.rewards[idx], self.dones[idx], idx, weights.astype(np.float32)
        )

    def update_priorities(self, indexes, p):
        self.priorities[indexes] = p + 1e-6  # Add small epsilon


# ============================================================
# 2. Noisy Linear Layer
# ============================================================

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.mu_w = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.sigma_w = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.mu_b = nn.Parameter(torch.FloatTensor(out_features))
        self.sigma_b = nn.Parameter(torch.FloatTensor(out_features))

        self.register_buffer("epsilon_w", torch.FloatTensor(out_features, in_features))
        self.register_buffer("epsilon_b", torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        # Factorized gaussian noise initialization
        mu_range = 1 / np.sqrt(self.in_features)
        self.mu_w.data.uniform_(-mu_range, mu_range)
        self.mu_b.data.uniform_(-mu_range, mu_range)
        self.sigma_w.data.fill_(self.sigma_init)
        self.sigma_b.data.fill_(self.sigma_init)

    def reset_noise(self):
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # Outer product
        self.epsilon_w.copy_(epsilon_out.ger(epsilon_in))
        self.epsilon_b.copy_(epsilon_out)

    def scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    def forward(self, input):
        if self.training:
            return F.linear(input, self.mu_w + self.sigma_w * self.epsilon_w,
                            self.mu_b + self.sigma_b * self.epsilon_b)
        else:
            return F.linear(input, self.mu_w, self.mu_b)


# ============================================================
# 3. Rainbow Network (Dueling + Noisy + C51 Head)
# ============================================================

class RainbowNetwork(nn.Module):
    def __init__(self, state_size, action_size, n_atoms=51, v_min=-10, v_max=10):
        super(RainbowNetwork, self).__init__()
        self.n_actions = action_size
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max

        # Support vector for C51
        self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms))

        # Feature extraction
        self.fc1 = NoisyLinear(state_size, 128)
        self.fc2 = NoisyLinear(128, 128)

        # Value Stream (Dueling)
        self.val1 = NoisyLinear(128, 128)
        self.val2 = NoisyLinear(128, n_atoms)

        # Advantage Stream (Dueling)
        self.adv1 = NoisyLinear(128, 128)
        self.adv2 = NoisyLinear(128, action_size * n_atoms)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Value
        val = F.relu(self.val1(x))
        val = self.val2(val).view(-1, 1, self.n_atoms)

        # Advantage
        adv = F.relu(self.adv1(x))
        adv = self.adv2(adv).view(-1, self.n_actions, self.n_atoms)

        # Combine (Dueling aggregation: Q = V + (A - mean(A)))
        q_logits = val + (adv - adv.mean(dim=1, keepdim=True))

        # Output probabilities (Softmax along atom dimension)
        dist = F.softmax(q_logits, dim=2)
        return dist

    def reset_noise(self):
        self.fc1.reset_noise()
        self.fc2.reset_noise()
        self.val1.reset_noise()
        self.val2.reset_noise()
        self.adv1.reset_noise()
        self.adv2.reset_noise()

    def get_q_value(self, x):
        dist = self(x)
        # Expected value = sum(p * z)
        return torch.sum(dist * self.support, dim=2)


# ============================================================
# 4. The Agent
# ============================================================

class RainbowDQNAgent(BaseAgent):
    def __init__(self, agent_name, state_size, action_size,
                 level='medium',
                 memory_size=50000,
                 batch_size=32,
                 lr=1e-3,
                 n_step=3,
                 v_min=-10,
                 v_max=10,
                 n_atoms=51):

        super().__init__(agent_name, state_size, action_size, level)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.batch_size = batch_size
        self.gamma = 0.99
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.tau = 0.005  # Soft update parameter

        # Networks
        self.online_net = RainbowNetwork(state_size, action_size, n_atoms, v_min, v_max).to(self.device)
        self.target_net = RainbowNetwork(state_size, action_size, n_atoms, v_min, v_max).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)

        # Replay Buffer
        self.memory = PrioritizedNStepBuffer(memory_size, state_size, n_step=n_step, gamma=self.gamma)

        # Training metrics
        self.learn_step_counter = 0

    def select_action(self, state, training=True):
        # State preprocessing
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if training:
            # Resample noise for exploration in Noisy Nets
            self.online_net.reset_noise()

        with torch.no_grad():
            q_values = self.online_net.get_q_value(state)
            action = q_values.argmax(dim=1).item()

        return action

    def train_step(self, state, action, reward, next_state, done):
        # Store experience
        self.memory.store(state, next_state, action, reward, done)

        # Only train if we have enough samples
        if self.memory.ptr < self.batch_size and not self.memory.full:
            return

        # 1. Sample from PER buffer
        states, next_states, actions, rewards, dones, idxs, weights = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        # 2. Compute Distributional Targets (Categorical Projection)
        with torch.no_grad():
            self.target_net.reset_noise()
            self.online_net.reset_noise()

            # Double DQN Selection: Select action using Online Net
            next_q_online = self.online_net.get_q_value(next_states)
            next_actions = next_q_online.argmax(1)

            # Evaluate Action using Target Net distribution
            next_dist = self.target_net(next_states)  # [batch, actions, atoms]
            next_dist = next_dist[range(self.batch_size), next_actions]  # [batch, atoms]

            # Project distribution
            t_z = rewards + (1 - dones) * (self.gamma ** self.memory.n_step) * self.target_net.support.unsqueeze(0)
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)

            b = (t_z - self.v_min) / ((self.v_max - self.v_min) / (self.n_atoms - 1))
            l = b.floor().long()
            u = b.ceil().long()

            proj_dist = torch.zeros(next_dist.size(), device=self.device)

            # Distribute probabilities to neighboring atoms
            offset = torch.linspace(0, (self.batch_size - 1) * self.n_atoms, self.batch_size).long().unsqueeze(
                1).expand(self.batch_size, self.n_atoms).to(self.device)

            proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
            proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

        # 3. Compute Loss
        dist = self.online_net(states)
        log_p = torch.log(dist[range(self.batch_size), actions] + 1e-8)

        # KL Divergence / Cross Entropy
        elementwise_loss = -torch.sum(proj_dist * log_p, dim=1)
        loss = torch.mean(weights.squeeze() * elementwise_loss)

        # 4. Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 5.0)
        self.optimizer.step()

        # 5. Update Priorities
        loss_for_priorities = elementwise_loss.detach().cpu().numpy()
        self.memory.update_priorities(idxs, loss_for_priorities)

        # 6. Soft Update Target Network
        self.learn_step_counter += 1
        if self.learn_step_counter % 2 == 0:  # Update every few steps or strictly follow Tau
            for target_param, local_param in zip(self.target_net.parameters(), self.online_net.parameters()):
                target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def save(self, filepath):
        torch.save({
            'online_net': self.online_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, filepath)
        print(f"Agent saved to {filepath}")

    def load(self, filepath):
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath)
            self.online_net.load_state_dict(checkpoint['online_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"Agent loaded from {filepath}")
        else:
            print("Checkpoint not found!")