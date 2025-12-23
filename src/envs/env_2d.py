import torch

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Simple2DGridENV:
    def __init__(self, **kwargs):
        """
        Inputs (kwargs) {
            "state_dim": int, dimension of the state space (default: 2)
            "action_dim": int, dimension of the action space (default: 4)
            "max_episode_steps": int, maximum steps per episode (default: 200)
            "device": str or torch.device, device to run the environment on (default: "cpu")
        }
        """
        self.state_dim         = kwargs.get("state_dim", 2)
        self.action_dim        = kwargs.get("action_dim", 4)
        self.max_episode_steps = kwargs.get("max_episode_steps", 200)
        self.device            = kwargs.get("device", "cpu")
        self.current_step      = 0
        self.state             = None
        
        self.reset()
        
    def reset(self):
        self.env_size = torch.tensor([100.0, 100.0], device=self.device)
        self.init_pos = torch.tensor([10.0, 10.0], device=self.device)
        self.goal_pos = torch.tensor([90.0, 90.0], device=self.device)
        self.state    = self.init_pos.clone()
        self.current_step = 0
        return self.state.clone()
    
    def action_space(self, action):
        """Grid action space: up, down, left, right"""
        if action == 0:   # up
            delta = torch.tensor([0.0, 1.0], device=self.device)
        elif action == 1: # down
            delta = torch.tensor([0.0, -1.0], device=self.device)
        elif action == 2: # left
            delta = torch.tensor([-1.0, 0.0], device=self.device)
        elif action == 3: # right
            delta = torch.tensor([1.0, 0.0], device=self.device)
        else:
            raise ValueError("Invalid action")
        return delta
    
    def step(self, action):
        action_delta = self.action_space(action)
        
        self.state += action_delta
        
        # Boundary conditions
        self.state = torch.clamp(self.state, torch.tensor([0.0, 0.0], device=self.device), self.env_size)
        
        self.current_step += 1
        
        reward, done = self.compute_reward_done()
        return self.state.clone(), reward, done
    
    def compute_reward_done(self):
        dist_reward_norm = torch.abs(torch.norm(torch.tensor([0.0, 0.0], device=self.device) - self.env_size))
        distance_to_goal = torch.norm(self.state - self.goal_pos)
        
        if distance_to_goal < 3.0:
            reward = 100.0
            done = True
        else:
            reward = -distance_to_goal.item()
            done = False
        
        if self.current_step >= self.max_episode_steps:
            done = True
        
        return reward/dist_reward_norm.item(), done
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    env = Simple2DGridENV(state_dim=2, action_dim=4, max_episode_steps=200, device=device)
    state = env.reset()
    done = False
    total_reward = 0.0
    
    while not done:
        action = torch.randint(0, env.action_dim, (1,), device=device).item()
        state, reward, done = env.step(action)
        total_reward += reward
        print(f"Step: {env.current_step}, State: {state.cpu().numpy()}, Reward: {reward}, Done: {done}")
        
    print(f"Episode finished with total reward: {total_reward}")