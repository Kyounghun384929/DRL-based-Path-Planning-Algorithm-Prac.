import torch
import torch.nn as nn
import torch.optim as optim

class PPOAgent:
    def __init__(self, model, lr=0.0003, gamma=0.99, K_epochs=3, eps_clip=0.2, device='cpu'):
        self.gamma       = gamma
        self.eps_clip    = eps_clip
        self.K_epochs    = K_epochs
        self.device      = device
        
        # 네트워크 및 옵티마이저
        self.policy = model.to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # GAE 파라미터
        self.lmbda = 0.95
        self.mse_loss = nn.MSELoss()
        
        # 데이터 버퍼 (Trajectory 저장용)
        self.data = []
        
    def put_data(self, transition):
        """
        transition: (state, action, reward, next_state, prob_a, done)
        """
        self.data.append(transition)