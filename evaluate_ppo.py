# evaluate_final.py
import torch
from models import VLA_POLICY
from env import GridWorldEnv
import numpy as np
import sys

VOCAB = { "<pad>": 0, "go": 1, "to": 2, "the": 3, "red": 4, "green": 5, "blue": 6, "object": 7 }

def tokenize(text):
    tokens = [VOCAB.get(word.lower(), 0) for word in text.split()]
    padded = tokens + [0] * (10 - len(tokens))
    return torch.tensor(padded[:10], dtype=torch.long)

def evaluate_policy(model_path, num_episodes=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = VLA_POLICY(action_dim=4, vocab_size=len(VOCAB)).to(device)
    
    try:
        policy.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ 成功加载模型: {model_path}")
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return None, None
    
    policy.eval()
    env = GridWorldEnv(max_steps=50)
    successes = 0
    total_rewards = []
    
    print(f"正在评估 {num_episodes} 个回合...")
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            features = torch.tensor(obs['features'], dtype=torch.float32).unsqueeze(0).to(device)
            text = tokenize(obs['instruction']).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits, _ = policy(features, text)
                action = torch.argmax(logits, dim=1).item()
            
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        total_rewards.append(episode_reward)
        if episode_reward > 0.5:
            successes += 1
        
        if (episode + 1) % 20 == 0:
            current_rate = successes / (episode + 1)
            print(f"  回合 [{episode+1:3d}/{num_episodes}]，当前成功率: {current_rate:.2%}")
    
    success_rate = successes / num_episodes
    avg_reward = np.mean(total_rewards)
    
    print(f"\n{'='*60}")
    print(f"📊 最终模型评估结果: {model_path}")
    print(f"{'='*60}")
    print(f"评估总回合数: {num_episodes}")
    print(f"成功回合数: {successes}")
    print(f"最终成功率: {success_rate:.2%}")
    print(f"最终平均奖励: {avg_reward:.3f}")
    print(f"奖励范围: [{min(total_rewards):.3f}, {max(total_rewards):.3f}]")
    print(f"{'='*60}")
    
    return success_rate, avg_reward

if __name__ == "__main__":
    # 优先评估最终模型，如果没有则评估最新的检查点
    model_to_evaluate = "rl_policy_final.pth"
    
    import os
    if not os.path.exists(model_to_evaluate):
        print(f"⚠️  未找到最终模型文件: {model_to_evaluate}")
        # 尝试寻找最新的检查点
        import glob
        checkpoints = glob.glob("rl_policy_iter*.pth")
        if checkpoints:
            # 按迭代次数排序，取最大的
            checkpoints.sort(key=lambda x: int(x.split('iter')[1].split('.pth')[0]))
            model_to_evaluate = checkpoints[-1]
            print(f"将评估最新的检查点: {model_to_evaluate}")
        else:
            print("❌ 未找到任何PPO模型文件。")
            sys.exit(1)
    
    success_rate, avg_reward = evaluate_policy(model_to_evaluate, num_episodes=1000)
    
    # 对比BC基线
    print("\n📈 性能对比:")
    print(f"{'模型':<25} {'成功率':<10} {'平均奖励':<10}")
    print(f"{'-'*45}")
    print(f"{'启发式基线':<25} {'~82%':<10} {'~0.68':<10}")
    print(f"{'BC模型 (特征工程后)':<25} {'87%':<10} {'0.756':<10}")
    print(f"{'PPO微调策略 (最终)':<25} {f'{success_rate:.2%}':<10} {f'{avg_reward:.3f}':<10}")