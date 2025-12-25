import torch


class Simple2DContinuousENV:
    def __init__(self, **kwargs):
        """
        A simple 2D continuous environment where an agent moves in a 2D space
        to reach a goal position. The agent can move in any direction with
        continuous actions.
        
        Args:
            state_dim (int): Dimension of the state space (default: 2).
            action_dim (int): Dimension of the action space (default: 2).
            max_episode_steps (int): Maximum steps per episode (default: 200).
            device (str): Device to run the environment on (default: "cpu").
        Returns:
            state (torch.Tensor): The current state of the environment.
        """
        self.state_dim         = kwargs.get("state_dim", 2)
        self.action_dim        = kwargs.get("action_dim", 2)
        self.max_episode_steps = kwargs.get("max_episode_steps", 200)
        self.device            = kwargs.get("device", "cpu")
        self.current_step      = 0
        self.state             = None
        
        self.max_speed = torch.tensor(1.0, device=self.device)  # Maximum movement per step
        
        self.reset()
    
    def reset(self):
        self.env_size = torch.tensor([100.0, 100.0], device=self.device)
        self.init_pos = torch.tensor([10.0, 10.0], device=self.device)
        self.goal_pos = torch.tensor([90.0, 90.0], device=self.device)
        self.state    = self.init_pos.clone()
        self.current_step = 0
        return self.state.clone()
    
    def action_space(self, action):
        """Continuous action space: 2D vector indicating movement direction and magnitude"""
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, device=self.device, dtype=torch.float32)
        if tuple(action.shape) != (self.action_dim,):
            raise ValueError(f"Action must be of shape ({self.action_dim},)")
        return action
    
    def step(self, action):
        # Action
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, device=self.device, dtype=torch.float32)
        
        # Vector cal (neural net output * max speed)
        delta = action * self.max_speed
        
        # Update state
        self.state += delta
        
        # Boundary conditions
        self.state = torch.clamp(self.state, torch.tensor([0.0, 0.0], device=self.device), self.env_size)
        
        self.current_step += 1
        reward, done = self.compute_reward_done()
        
        return self.state.clone(), reward, done
    
    def compute_reward_done(self):
        dist_reward_norm = torch.abs(torch.norm(torch.tensor([0.0, 0.0], device=self.device) - self.env_size))
        distance_to_goal = torch.norm(self.state - self.goal_pos)
        
        if distance_to_goal < 3.0:
            reward = torch.tensor(100.0, device=self.device, dtype=torch.float32)
            done = torch.tensor(True, device=self.device, dtype=torch.bool)
        else:
            reward = -distance_to_goal / dist_reward_norm
            done = torch.tensor(False, device=self.device, dtype=torch.bool)
        
        if self.current_step >= self.max_episode_steps:
            done = torch.tensor(True, device=self.device, dtype=torch.bool)
        
        return reward, done


if __name__ == "__main__":
    env = Simple2DContinuousEnv(device="cpu")
    state = env.reset()
    done = False
    total_reward = 0.0
    
    while not done:
        action = torch.rand(env.action_dim) * 2 - 1  # Random action in range [-1, 1]
        state, reward, done = env.step(action)
        total_reward += reward.item()
        print(f"State: {state}, Reward: {reward}, Done: {done}")
    
    print(f"Total Reward: {total_reward}")