import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

if __name__ == "__main__":
    state_dim = 2
    action_dim = 4
    model = QNetwork(state_dim, action_dim)
    sample_input = torch.randn((1, state_dim))
    output = model(sample_input)
    print("Q-Network output:", output)
    print("Max Q-value action index:", torch.argmax(output, dim=1).item())