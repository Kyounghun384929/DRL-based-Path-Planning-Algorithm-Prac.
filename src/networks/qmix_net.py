import torch
import torch.nn as nn
import torch.nn.functional as F

class MixingNetwork(nn.Module):
    def __init__(self, n_agents, state_dim, hidden_dim=64):
        super(MixingNetwork, self).__init__()
        self.n_agents   = n_agents
        self.state_dim  = state_dim * n_agents
        self.hidden_dim = hidden_dim
        
        self.hyper_w1 = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_agents * hidden_dim)
        )
        
        self.hyper_b1 = nn.Linear(self.state_dim, hidden_dim)
        
        self.hyper_w2 = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 1)
        )
        
        self.hyper_b2 = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        
    def forward(self, q_values, states):
        """
        Args:
            q_values (torch.Tensor): Shape (batch_size, n_agents) - individual Q-values from each agent
            states (torch.Tensor): Shape (batch_size, state_dim * n_agents) - global state
        """
        
        batch_size = q_values.size(0)
        
        # Global state flatten (Batch, N * State_Dim)
        states = states.view(batch_size, -1)
        
        # Individual Q-values reshape (Batch, 1, N)
        q_values = q_values.view(batch_size, 1, self.n_agents)
        
        # First layer
        w1 = torch.abs(self.hyper_w1(states))  # (Batch, N * Hidden_Dim) - Ensure Monotonicity 
        w1 = w1.view(batch_size, self.n_agents, self.hidden_dim)  # (Batch, N, Hidden_Dim)
        b1 = self.hyper_b1(states).view(batch_size, 1, self.hidden_dim)  # (Batch, 1, Hidden_Dim)
        
        # Q * w1 + b1
        hidden = F.elu(torch.bmm(q_values, w1) + b1)  # (Batch, 1, Hidden_Dim)
        
        # Second layer
        w2 = torch.abs(self.hyper_w2(states))  # (Batch, Hidden_Dim * 1) - Absolute
        w2 = w2.view(batch_size, self.hidden_dim, 1)
        b2 = self.hyper_b2(states).view(batch_size, 1, 1)  # (Batch, 1, 1)
        
        # Hidden * w2 + b2
        q_total = torch.bmm(hidden, w2) + b2  # (Batch, 1, 1)
        
        return q_total.view(batch_size, 1)  # (Batch, 1)