"""Independent Q-Learning (IQL) algorithm implementation."""

import torch
import torch.nn as nn
import torch.optim as optim

import random

from collections import deque
from src.networks import QNetwork

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

class IQLAgent:
    def __init__(self, device, **kwargs):
        self.device = device
        
        self.action_dim = kwargs.get('action_dim')
        self.state_dim = kwargs.get('state_dim')
        self.lr = kwargs.get('lr', 1e-3)
        self.gamma = kwargs.get('gamma', 0.99)
        
        # Q-Network & Target Network
        self.q_net = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.t_net = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.t_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.AdamW(self.q_net.parameters(), lr=self.lr)
        
    def get_action(self, state, epsilon):
        if random.random() < epsilon:
            # 모든 에이전트에 대해 무작위 행동 선택
            return torch.randint(0, self.action_dim, (state.size(0),), device=self.device)
        else:
            with torch.no_grad():
                q_values = self.q_net(state) # (N, Action_Dim)
                return q_values.argmax(dim=1) # (N,)
    
    def update(self, replay_buffer=ReplayBuffer(), batch_size=128):
        if len(replay_buffer) < batch_size:
            return
        
        # batch sampling
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        # Flatten
        states = states.view(-1, states.size(-1)).to(self.device)
        actions = actions.view(-1).to(self.device)
        rewards = rewards.view(-1, 1).to(self.device)
        next_states = next_states.view(-1, next_states.size(-1)).to(self.device)
        dones = dones.view(-1, 1).to(self.device)
        
        # Q-value update
        curr_q = self.q_net(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q = self.t_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones.float()) * self.gamma * next_q
            
        loss = nn.MSELoss()(curr_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def update_target_network(self):
        self.t_net.load_state_dict(self.q_net.state_dict())
        

if __name__ == "__main__":
    from src.envs import Env2DMA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Env2DMA(num_agents=5, device=device)
    state_dim = env.state_dim
    action_dim = env.action_dim
    
    agent = IQLAgent(device, state_dim=state_dim, action_dim=action_dim, lr=1e-4)
    replay_buffer = ReplayBuffer()
    
    num_episodes = 1000
    epsilon = 1
    epsilon_decay = 0.995
    epsilon_min = 0.001
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            action = agent.get_action(state, epsilon)
            next_state, reward, dones = env.step(action)
            
            replay_buffer.push(state, action, reward, next_state, dones)
            state = next_state
            
            agent.update(replay_buffer)
            
            if dones.all():
                break
        
        if episode % 10 == 0:
            agent.update_target_network()
        
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
            
        print(f"\n{'='*20} Episode {episode+1}/{num_episodes} {'='*20}")
        print(f"Total Reward: {reward.sum().item():.5f}")
        print("Last Positions:")
        for i, pos in enumerate(state.tolist()):
            # Check if reached goal
            goal = env.goal_pos[i]
            dist = torch.norm(state[i] - goal).item()
            reached = "Reached goal" if dist < 3.0 else "Not reached"
            print(f"  - Agent {i}: {[round(x, 2) for x in pos]} | {reached} (Dist: {dist:.2f})")
        print(f"{'='*50}")