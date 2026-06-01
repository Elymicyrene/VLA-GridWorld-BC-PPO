import torch
import torch.nn as nn
from torch.optim import Adam
import sys
import importlib

# ----- 关键修复：尝试强制重新加载 models 模块 -----
try:
    # 先尝试从当前目录显式导入
    sys.path.insert(0, '.')
    import models
    # 强制重新加载模块，清除可能的旧缓存
    importlib.reload(models)
    from models import VLA_POLICY
    print("✅ 已重新加载 'models' 模块。")
except ImportError as e:
    print(f"❌ 导入模型失败: {e}")
    print("请确保 models.py 文件在当前目录。")
    exit(1)

from env import GridWorldEnv
import numpy as np
import random

# Simple Tokenizer (Same as BC)
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

def compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    gae = 0
    returns = []
    advantages = []
    
    values_np = np.array(values + [next_value], dtype=np.float32)
    rewards_np = np.array(rewards, dtype=np.float32)
    dones_np = np.array(dones, dtype=bool)
    
    next_value_temp = next_value
    
    for step in reversed(range(len(rewards))):
        if dones_np[step]:
            delta = rewards_np[step] - values_np[step]
            next_value_temp = 0.0
        else:
            delta = rewards_np[step] + gamma * next_value_temp - values_np[step]
        
        gae = delta + gamma * lam * gae * (not dones_np[step])
        advantages.insert(0, gae)
        returns.insert(0, gae + values_np[step])
        next_value_temp = values_np[step]
        
    return torch.tensor(returns, dtype=torch.float32), torch.tensor(advantages, dtype=torch.float32)

def train_rl():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 初始化策略模型 (使用重新加载后的类定义)
    policy = VLA_POLICY(action_dim=4, vocab_size=len(VOCAB)).to(device)
    
    # ========== 修改的BC权重加载逻辑 ==========
    bc_checkpoint_path = "bc_policy_structured.pth"
    load_successful = False
    
    if True:  # 保持缩进，方便你后续可能添加条件判断
        try:
            print(f"尝试加载BC权重: {bc_checkpoint_path}")
            checkpoint = torch.load(bc_checkpoint_path, map_location=device)
            
            # 方法1: 尝试直接严格加载
            policy.load_state_dict(checkpoint)
            print("✅ 方法1成功: 直接加载权重。")
            load_successful = True
            
        except RuntimeError as e:
            print(f"⚠️  方法1失败: {e}")
            print("尝试方法2: 只加载匹配的权重...")
            try:
                # 方法2: 遍历检查点，只加载当前模型中有的参数
                model_dict = policy.state_dict()
                
                # 1. 筛选出两个字典中都有的键
                pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
                # 2. 确保维度匹配
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if v.shape == model_dict[k].shape}
                
                if len(pretrained_dict) == 0:
                    print("❌ 方法2失败: 未找到任何可加载的匹配参数。")
                else:
                    # 3. 更新模型字典
                    model_dict.update(pretrained_dict)
                    # 4. 加载
                    policy.load_state_dict(model_dict)
                    loaded_keys = len(pretrained_dict)
                    total_keys = len(checkpoint)
                    print(f"✅ 方法2成功: 加载了 {loaded_keys}/{total_keys} 个参数。")
                    load_successful = True
                    
            except Exception as e2:
                print(f"❌ 方法2也失败: {e2}")
                
        except FileNotFoundError:
            print(f"❌ 检查点文件不存在: {bc_checkpoint_path}")
            print("请先运行 train_bc.py 生成BC模型。")
            return
    
    if not load_successful:
        print("\n⚠️  无法加载预训练权重。有2个选择:")
        print("   1. 按 Ctrl+C 中断，检查 models.py 文件。")
        print("   2. 继续运行，PPO将从随机初始化开始训练。")
        user_input = input("请输入你的选择 (1或2，然后按回车): ")
        if user_input.strip() == '1':
            print("程序退出。请确保 models.py 是最新版本。")
            return
        else:
            print("继续: PPO将从随机初始化开始训练。")
    
    optimizer = Adam(policy.parameters(), lr=5e-6)
    env = GridWorldEnv(max_steps=50)
    
    print("\n" + "="*60)
    print("Starting PPO Fine-tuning...")
    if load_successful:
        print("模式: BC Pretrained + PPO Fine-tuning")
    else:
        print("模式: PPO from Scratch (随机初始化)")
    print("="*60)
    
    # PPO 超参数 (基于BC微调调整得保守一些)
    num_iterations = 200
    steps_per_iter = 512
    gamma = 0.99
    lam = 0.92
    clip_epsilon = 0.15
    update_epochs = 6
    minibatch_size = 128
    
    # 用于跟踪每个完整episode的总奖励
    episode_returns = []
    
    for iteration in range(num_iterations):
        # 1. 收集轨迹
        buffer_features, buffer_text, buffer_actions = [], [], []
        buffer_log_probs, buffer_rewards, buffer_values, buffer_dones = [], [], [], []
        
        current_episode_reward = 0
        episode_return_tracker = []
        
        obs, _ = env.reset()
        
        for step in range(steps_per_iter):
            # 关键修改：使用特征，而不是图像
            features = torch.tensor(obs['features'], dtype=torch.float32).unsqueeze(0).to(device)
            text = tokenize(obs['instruction']).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits, val = policy(features, text)  # 输入是特征
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            buffer_features.append(features)
            buffer_text.append(text)
            buffer_actions.append(action)
            buffer_log_probs.append(log_prob)
            buffer_rewards.append(reward)
            buffer_values.append(val.item())
            buffer_dones.append(done)
            
            current_episode_reward += reward
            obs = next_obs
            
            if done:
                episode_return_tracker.append(current_episode_reward)
                current_episode_reward = 0
                obs, _ = env.reset()
        
        if current_episode_reward > 0:
            episode_return_tracker.append(current_episode_reward)
        
        # 计算平均episode奖励
        if episode_return_tracker:
            mean_episode_return = np.mean(episode_return_tracker)
            episode_returns.append(mean_episode_return)
        else:
            mean_episode_return = 0.0
        
        # Bootstrap value for last step
        features = torch.tensor(obs['features'], dtype=torch.float32).unsqueeze(0).to(device)
        text = tokenize(obs['instruction']).unsqueeze(0).to(device)
        with torch.no_grad():
            _, next_val = policy(features, text)
            next_val = next_val.item()
            
        # 2. 计算GAE和Returns
        returns, advantages = compute_gae(
            rewards=buffer_rewards,
            values=buffer_values,
            dones=buffer_dones,
            next_value=next_val,
            gamma=gamma,
            lam=lam
        )
        returns = returns.to(device)
        advantages = advantages.to(device)
        
        if advantages.std() > 0:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 展平缓冲区
        b_features = torch.cat(buffer_features)
        b_text = torch.cat(buffer_text)
        b_actions = torch.cat(buffer_actions)
        b_log_probs = torch.cat(buffer_log_probs).detach()
        
        # 3. PPO 更新
        dataset_size = b_features.size(0)
        indices = np.arange(dataset_size)
        
        policy_loss_epoch, value_loss_epoch = 0, 0
        num_updates = 0
        
        for _ in range(update_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, minibatch_size):
                end = start + minibatch_size
                idx = indices[start:end]
                
                mb_features = b_features[idx]
                mb_text = b_text[idx]
                mb_actions = b_actions[idx]
                mb_old_log_probs = b_log_probs[idx]
                mb_advantages = advantages[idx]
                mb_returns = returns[idx]
                
                logits, values = policy(mb_features, mb_text)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()
                
                ratio = (new_log_probs - mb_old_log_probs).exp()
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (mb_returns - values.squeeze()).pow(2).mean()
                
                loss = policy_loss + value_loss - 0.005 * entropy
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.3)
                optimizer.step()
                
                policy_loss_epoch += policy_loss.item()
                value_loss_epoch += value_loss.item()
                num_updates += 1
        
        avg_policy_loss = policy_loss_epoch / num_updates if num_updates > 0 else 0
        avg_value_loss = value_loss_epoch / num_updates if num_updates > 0 else 0
        
        # 打印滑动平均奖励
        if len(episode_returns) >= 1:
            recent_returns = episode_returns[-min(10, len(episode_returns)):]
            moving_avg = np.mean(recent_returns)
        else:
            moving_avg = 0.0
        
        print(f"Iter {iteration+1:3d} | "
              f"EpReturn: {mean_episode_return:6.2f} | "
              f"MovAvg: {moving_avg:6.2f} | "
              f"PLoss: {avg_policy_loss:7.4f} | "
              f"VLoss: {avg_value_loss:7.4f}")
        
        # 每20次迭代保存一次检查点
        if (iteration + 1) % 20 == 0:
            checkpoint_name = f"rl_policy_iter{iteration+1}.pth"
            torch.save(policy.state_dict(), checkpoint_name)
            print(f"💾 检查点已保存: {checkpoint_name}")
    
    # 保存最终模型
    final_model_name = "rl_policy_final.pth"
    torch.save(policy.state_dict(), final_model_name)
    print(f"\n✅ PPO训练完成！最终模型已保存: {final_model_name}")
    
    # 绘制训练曲线（可选）
    try:
        if len(episode_returns) > 1:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.plot(episode_returns, label='Episode Return')
            window_size = min(20, len(episode_returns) // 4)
            if window_size > 1:
                moving_avg = np.convolve(episode_returns, np.ones(window_size)/window_size, mode='valid')
                plt.plot(range(window_size-1, len(episode_returns)), moving_avg, 'r-', label=f'Moving Avg')
            plt.xlabel('Iteration')
            plt.ylabel('Average Episode Return')
            plt.title(f'PPO Training ({("BC+PPO" if load_successful else "PPO from Scratch")})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig('ppo_training_curve.png')
            plt.close()
            print("📈 训练曲线已保存: ppo_training_curve.png")
    except ImportError:
        print("⚠️  无法导入matplotlib，跳过绘制训练曲线")

if __name__ == "__main__":
    train_rl()