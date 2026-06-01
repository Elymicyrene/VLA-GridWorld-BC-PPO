import torch
from models import VLA_POLICY
from env import GridWorldEnv
import numpy as np

VOCAB = {
    "<pad>": 0,
    "go": 1,
    "to": 2,
    "the": 3,
    "red": 4,
    "green": 5,
    "blue": 6,
    "object": 7
}

def tokenize(text):
    tokens = [VOCAB.get(word.lower(), 0) for word in text.split()]
    padded = tokens + [0] * (10 - len(tokens))
    return torch.tensor(padded[:10], dtype=torch.long)

def evaluate_bc_policy(num_episodes=100, model_path="bc_policy_structured.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    policy = VLA_POLICY(action_dim=4, vocab_size=len(VOCAB)).to(device)
    try:
        policy.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ 成功从 '{model_path}' 加载模型。")
    except FileNotFoundError:
        print(f"❌ 错误：未找到模型文件 '{model_path}'，请确保已运行 train_bc.py 完成训练。")
        return None
    except Exception as e:
        print(f"❌ 加载模型时出错: {e}")
        return None
    
    policy.eval()  # 设置为评估模式
    
    # 创建环境
    env = GridWorldEnv(max_steps=50)
    successes = 0
    total_rewards = []
    
    print(f"开始评估 {num_episodes} 个回合...")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # 预处理观察：使用特征向量
            features = torch.tensor(obs['features'], dtype=torch.float32).unsqueeze(0).to(device)
            text = tokenize(obs['instruction']).unsqueeze(0).to(device)
            
            # 模型前向传播（选择概率最高的动作，即确定性策略）
            with torch.no_grad():
                logits, _ = policy(features, text)  # 输入是特征
                action = torch.argmax(logits, dim=1).item()
            
            # 在环境中执行动作
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            obs = next_obs
        
        # 判断该回合是否成功：成功回合的总奖励 > 0.5
        if episode_reward > 0.5:
            successes += 1
        total_rewards.append(episode_reward)
        
        if (episode + 1) % 10 == 0:
            current_success_rate = successes / (episode + 1)
            print(f"  回合 [{episode+1:3d}/{num_episodes}]，当前成功率: {current_success_rate:.2%}")
    
    success_rate = successes / num_episodes
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    
    print("\n" + "="*60)
    print("📊 BC 策略评估最终结果 (结构化特征版本)")
    print("="*60)
    print(f"评估总回合数: {num_episodes}")
    print(f"成功回合数: {successes}")
    print(f"成功率: {success_rate:.2%} ({successes}/{num_episodes})")
    print(f"平均回合总奖励: {avg_reward:.3f} ± {std_reward:.3f}")
    print(f"奖励范围: [{min(total_rewards):.3f}, {max(total_rewards):.3f}]")
    print("="*60)
    
    # 给出诊断建议
    if success_rate >= 0.7:
        print("✅ BC模型表现优秀，可以作为PPO的坚实起点。")
    elif success_rate >= 0.5:
        print("✅ BC模型表现良好，可以作为PPO的起点。")
        print("   建议在启动PPO前，调整环境奖励函数以提供更强的学习信号。")
    elif success_rate >= 0.3:
        print("⚠️  BC模型表现中等，PPO微调可能有挑战。")
        print("   可以尝试重新训练BC模型以获取更好的初始策略。")
    else:
        print("❌ BC模型表现不佳，需要重新检查模型或数据。")
        print("   建议检查特征工程和模型结构。")
    
    return success_rate, avg_reward, total_rewards

if __name__ == "__main__":
    evaluate_bc_policy(num_episodes=1000)