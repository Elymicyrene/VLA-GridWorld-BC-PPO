# part0_baselines.py
import numpy as np
from env import GridWorldEnv
import matplotlib.pyplot as plt

def run_random_agent(num_episodes=100):
    """随机智能体基准"""
    env = GridWorldEnv()
    successes = 0
    total_rewards = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = env.action_space.sample()  # 随机选择动作
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            if truncated:
                done = True
        
        # 检查是否成功（最终奖励为1.0）
        if episode_reward > 0.5:  # 因为成功奖励是1.0，失败奖励是-0.5或负的步数惩罚
            successes += 1
        
        total_rewards.append(episode_reward)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{num_episodes}, 当前成功率: {successes/(episode+1):.2%}")
    
    success_rate = successes / num_episodes
    avg_reward = np.mean(total_rewards)
    
    print(f"\n✅ 随机智能体结果:")
    print(f"成功率: {success_rate:.2%}")
    print(f"平均奖励: {avg_reward:.3f}")
    
    return success_rate, avg_reward

def run_heuristic_agent(num_episodes=100):
    """启发式智能体基准 - 简单朝向目标移动"""
    env = GridWorldEnv()
    successes = 0
    total_rewards = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        # 获取目标颜色和位置（这在实际任务中是未知的，但这里用于启发式）
        target_color = env.target_color
        target_pos = None
        for obj in env.objects:
            if obj['color'] == target_color:
                target_pos = obj['pos']
                break
        
        while not done:
            # 启发式策略：向目标移动
            agent_pos = env.agent_pos
            r, c = agent_pos
            tr, tc = target_pos
            
            # 选择朝向目标的动作
            if r > tr:
                action = 0  # 上
            elif r < tr:
                action = 1  # 下
            elif c > tc:
                action = 2  # 左
            elif c < tc:
                action = 3  # 右
            else:
                action = env.action_space.sample()  # 已经在目标位置
            
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            if truncated:
                done = True
        
        if episode_reward > 0.5:
            successes += 1
        
        total_rewards.append(episode_reward)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{num_episodes}, 当前成功率: {successes/(episode+1):.2%}")
    
    success_rate = successes / num_episodes
    avg_reward = np.mean(total_rewards)
    
    print(f"\n✅ 启发式智能体结果:")
    print(f"成功率: {success_rate:.2%}")
    print(f"平均奖励: {avg_reward:.3f}")
    
    return success_rate, avg_reward

if __name__ == "__main__":
    print("=== Part 0: 基线智能体评估 ===\n")
    
    print("1. 随机智能体:")
    random_success, random_reward = run_random_agent(num_episodes=50)  # 先测试50个episode
    
    print("\n" + "="*50 + "\n")
    
    print("2. 启发式智能体:")
    heuristic_success, heuristic_reward = run_heuristic_agent(num_episodes=50)
    
    print("\n" + "="*50)
    print("📊 基线对比结果:")
    print(f"随机智能体成功率: {random_success:.2%}")
    print(f"启发式智能体成功率: {heuristic_success:.2%}")
    print(f"提升: {((heuristic_success - random_success) / random_success * 100):.1f}%" if random_success > 0 else "N/A")