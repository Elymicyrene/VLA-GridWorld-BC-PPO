import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class GridWorldEnv(gym.Env):
    """
    A simple gridworld environment for instruction following.
    Observation (修改后):
        - features: (7,) 结构化特征向量 [agent_r, agent_c, target_r, target_c, is_red, is_green, is_blue]
        - instruction: text string
    Action:
        - Discrete(4): up, down, left, right
    """
    def __init__(self, grid_size=10, max_steps=50):
        super(GridWorldEnv, self).__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        
        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        
        # Observation space (修改关键点)
        # 特征向量: 智能体位置(归一化), 目标位置(归一化), 目标颜色(one-hot)
        self.observation_space = spaces.Dict({
            'features': spaces.Box(low=0, high=1, shape=(7,), dtype=np.float32),
            'instruction': spaces.Text(max_length=100)
        })
        
        self.agent_pos = [0, 0]
        self.objects = [] # List of (pos, color_name)
        self.target_color = ""
        self.current_step = 0
        
        self.color_map = {
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255)
        }
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self.current_step = 0
        self.agent_pos = [random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)]
        
        # Spawn 3 objects of different colors
        self.objects = []
        possible_colors = ['red', 'green', 'blue']
        occupied = {tuple(self.agent_pos)}
        
        for color in possible_colors:
            pos = [random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)]
            while tuple(pos) in occupied:
                pos = [random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)]
            occupied.add(tuple(pos))
            self.objects.append({'pos': pos, 'color': color})
            
        # Select one as target
        target_obj = random.choice(self.objects)
        self.target_color = target_obj['color']
        self.instruction = f"Go to the {self.target_color} object"
        
        # 返回结构化特征，而不是图像
        return self._get_obs(), {}
        
    def step(self, action):
        self.current_step += 1
        reward = -0.01
        terminated = False
        truncated = False
        
        # Move agent
        if action == 0: # Up
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1: # Down
            self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)
        elif action == 2: # Left
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 3: # Right
            self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)
            
        # Check if reached any object
        for obj in self.objects:
            if self.agent_pos == obj['pos']:
                if obj['color'] == self.target_color:
                    reward = 1.0
                    terminated = True
                else:
                    # Reached wrong object
                    reward = -0.5
                    terminated = True
                break
        
        if self.current_step >= self.max_steps:
            truncated = True
            
        return self._get_obs(), reward, terminated, truncated, {}
        
    def _get_obs(self):
        # 找到目标对象的位置
        target_pos = None
        for obj in self.objects:
            if obj['color'] == self.target_color:
                target_pos = obj['pos']
                break
        
        agent_r, agent_c = self.agent_pos
        target_r, target_c = target_pos
        
        # 构建结构化特征向量 (7维)
        # [智能体行坐标, 智能体列坐标, 目标行坐标, 目标列坐标, 是红色目标, 是绿色目标, 是蓝色目标]
        features = np.array([
            agent_r / (self.grid_size - 1),          # 归一化到 [0, 1]
            agent_c / (self.grid_size - 1),
            target_r / (self.grid_size - 1),
            target_c / (self.grid_size - 1),
            1.0 if self.target_color == 'red' else 0.0,
            1.0 if self.target_color == 'green' else 0.0,
            1.0 if self.target_color == 'blue' else 0.0,
        ], dtype=np.float32)
        
        return {
            'features': features,
            'instruction': self.instruction
        }

    def render(self):
        # 保持为空，因为我们使用特征而非图像
        pass