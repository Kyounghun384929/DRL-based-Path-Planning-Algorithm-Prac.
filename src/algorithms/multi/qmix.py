"""QMIX algorithm implementation."""

import torch
import torch.nn as nn
import torch.optim as optim
import random

from collections import deque
from src.networks import MixingNetwork, QNetwork


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (torch.stack(state),
                torch.stack(action),
                torch.stack(reward),
                torch.stack(next_state),
                torch.stack(done))
        
    def __len__(self):
        return len(self.buffer)


class QMIXAgent:
    def __init__(self, n_agents, state_dim, action_dim, **kwargs):
        self.n_agents   = n_agents
        self.state_dim  = state_dim
        self.action_dim = action_dim
        
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = kwargs.get('lr', 1e-4)
        self.gamma = kwargs.get('gamma', 0.99)
        
        # Individual Agent Net (Parameter sharing)
        self.agent_net = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_agent_net = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_agent_net.load_state_dict(self.agent_net.state_dict())
        
        # Mixing Network (QMIX)
        self.mix_net = MixingNetwork(self.n_agents, self.state_dim).to(self.device)
        self.target_mix_net = MixingNetwork(self.n_agents, self.state_dim).to(self.device)
        self.target_mix_net.load_state_dict(self.mix_net.state_dict())
        
        # Parameter Optim
        self.parameters = list(self.agent_net.parameters()) + list(self.mix_net.parameters())
        self.optimizer = optim.AdamW(self.parameters, lr=self.lr)
        
        
    def get_action(self, state, epsilon):
        # Epsilon-Greedy action selection
        if random.random() < epsilon:
            return torch.randint(0, self.action_dim, (self.n_agents,), device=self.device)
        else:
            with torch.no_grad():
                # Normalize state
                norm_state = state / 100.0
                q_values = self.agent_net(norm_state) # (N, Action_Dim)
                return q_values.argmax(dim=1) # (N,)
            
    
    def update(self, replay_buffer=ReplayBuffer(), batch_size=128):
        if len(replay_buffer) < batch_size:
            return
        
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        # Normalize states
        states = states / 100.0
        next_states = next_states / 100.0
        
        # current Q_tot
        curr_q_out = self.agent_net(states) # (B, N, Action_Dim)
        curr_q_i = curr_q_out.gather(2, actions.unsqueeze(-1)).squeeze(-1) # (B, N)
        
        # Mix to get Q_tot
        q_total = self.mix_net(curr_q_i, states) # (B, 1)
        
        # target Q_tot
        with torch.no_grad():
            next_q_out = self.target_agent_net(next_states) # (B, N, Action_Dim)
            max_next_q_i = next_q_out.max(dim=2)[0] # (B, N)
            
            target_q_total = self.target_mix_net(max_next_q_i, next_states) # (B, 1)
            
        # Loss
        global_reward = rewards.sum(dim=1) # (B, 1)
        global_done = dones.all(dim=1).float() # (B, 1)
        
        expected_q = global_reward + self.gamma * target_q_total * (1 - global_done) # (B, 1)
        loss = nn.SmoothL1Loss()(q_total, expected_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters, 1.0) # For stable learning
        self.optimizer.step()
        
        return loss.item()
        
    
    def update_target(self):
        self.target_agent_net.load_state_dict(self.agent_net.state_dict())
        self.target_mix_net.load_state_dict(self.mix_net.state_dict())
        

if __name__ == "__main__":
    from src.envs import Env2DMA
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = Env2DMA(num_agents=3, device=device)
    
    state_dim = env.state_dim
    action_dim = env.action_dim
    n_agents = env.num_agents
    
    agent = QMIXAgent(n_agents=n_agents, state_dim=state_dim, action_dim=action_dim, device=device)
    
    replay_buffer = ReplayBuffer()
    
    episodes = 1000
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        loss_sum = 0
        loss_count = 0
        done = False
        
        while True:
            action = agent.get_action(state, epsilon)
            next_state, reward, dones = env.step(action)
            
            replay_buffer.push(state, action, reward, next_state, dones)
            
            state = next_state
            episode_reward += reward.sum().item()
            
            loss = agent.update(replay_buffer)
            if loss is not None:
                loss_sum += loss
                loss_count += 1
            
            if dones.all():
                break
                
        if episode % 10 == 0:
            agent.update_target()
        
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        
        avg_loss = loss_sum / loss_count if loss_count > 0 else 0.0
        print(f"\n{'='*20} Episode {episode+1}/{episodes} {'='*20}")
        print(f"Total Reward: {episode_reward:.5f}, Epsilon: {epsilon:.4f}, Avg Loss: {avg_loss:.5f}")
        print("Last Positions:")
        for i, pos in enumerate(state.tolist()):
            # Check if reached goal
            goal = env.goal_pos[i]
            dist = torch.norm(state[i] - goal).item()
            reached = "Reached goal" if dist < 3.0 else "Not reached"
            print(f"  - Agent {i}: {[round(x, 2) for x in pos]} | {reached} (Dist: {dist:.2f})")
        print(f"{'='*50}")
    