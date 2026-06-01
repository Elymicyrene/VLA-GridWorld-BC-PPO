# final_evaluation.py
import torch
import numpy as np
import time
from env import GridWorldEnv

VOCAB = {"<pad>": 0, "go": 1, "to": 2, "the": 3, "red": 4, "green": 5, "blue": 6, "object": 7}

def tokenize(text):
    tokens = [VOCAB.get(word.lower(), 0) for word in text.split()]
    padded = tokens + [0] * (10 - len(tokens))
    return torch.tensor(padded[:10], dtype=torch.long)

class ComprehensiveEvaluator:
    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        self.model_path = model_path
        self.policy = None
        
        if model_path:
            from models import VLA_POLICY
            self.policy = VLA_POLICY(action_dim=4, vocab_size=len(VOCAB)).to(self.device)
            try:
                self.policy.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"✅ 成功加载模型: {model_path}")
            except Exception as e:
                print(f"❌ 加载模型失败: {e}")
                return
            self.policy.eval()
    
    def evaluate_heuristic(self, num_episodes=1000):
        """评估启发式基线策略"""
        env = GridWorldEnv(max_steps=50)
        successes = 0
        total_rewards = []
        steps_to_success = []
        
        print(f"\n📊 评估启发式基线 (共 {num_episodes} 回合)...")
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            target_color = env.target_color
            target_pos = None
            
            for obj in env.objects:
                if obj['color'] == target_color:
                    target_pos = obj['pos']
                    break
            
            done = False
            episode_reward = 0
            step_count = 0
            
            while not done:
                agent_pos = env.agent_pos
                r, c = agent_pos
                tr, tc = target_pos
                
                if r > tr:
                    action = 0  # 上
                elif r < tr:
                    action = 1  # 下
                elif c > tc:
                    action = 2  # 左
                elif c < tc:
                    action = 3  # 右
                else:
                    action = np.random.randint(0, 4)
                
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                step_count += 1
                
                if terminated and reward > 0.5:  # 成功
                    steps_to_success.append(step_count)
            
            total_rewards.append(episode_reward)
            if episode_reward > 0.5:
                successes += 1
            
            if (episode + 1) % 100 == 0:
                current_rate = successes / (episode + 1)
                print(f"  进度: [{episode+1:4d}/{num_episodes}]，当前成功率: {current_rate:.2%}")
        
        success_rate = successes / num_episodes
        avg_reward = np.mean(total_rewards)
        
        result = {
            'model': 'Heuristic Baseline',
            'success_rate': success_rate,
            'avg_reward': avg_reward,
            'successes': successes,
            'total_episodes': num_episodes,
            'avg_steps_to_success': np.mean(steps_to_success) if steps_to_success else 0,
            'std_reward': np.std(total_rewards),
            'min_reward': np.min(total_rewards),
            'max_reward': np.max(total_rewards)
        }
        
        return result
    
    def evaluate_policy(self, num_episodes=1000):
        """评估训练好的策略模型"""
        if self.policy is None:
            print("❌ 未加载模型，无法评估")
            return None
        
        env = GridWorldEnv(max_steps=50)
        successes = 0
        total_rewards = []
        steps_to_success = []
        
        print(f"\n📊 评估模型: {self.model_path} (共 {num_episodes} 回合)...")
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            step_count = 0
            
            while not done:
                features = torch.tensor(obs['features'], dtype=torch.float32).unsqueeze(0).to(self.device)
                text = tokenize(obs['instruction']).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    logits, _ = self.policy(features, text)
                    action = torch.argmax(logits, dim=1).item()
                
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                step_count += 1
                
                if terminated and reward > 0.5:  # 成功
                    steps_to_success.append(step_count)
            
            total_rewards.append(episode_reward)
            if episode_reward > 0.5:
                successes += 1
            
            if (episode + 1) % 100 == 0:
                current_rate = successes / (episode + 1)
                print(f"  进度: [{episode+1:4d}/{num_episodes}]，当前成功率: {current_rate:.2%}")
        
        success_rate = successes / num_episodes
        avg_reward = np.mean(total_rewards)
        
        result = {
            'model': self.model_path.split('/')[-1],
            'success_rate': success_rate,
            'avg_reward': avg_reward,
            'successes': successes,
            'total_episodes': num_episodes,
            'avg_steps_to_success': np.mean(steps_to_success) if steps_to_success else 0,
            'std_reward': np.std(total_rewards),
            'min_reward': np.min(total_rewards),
            'max_reward': np.max(total_rewards)
        }
        
        return result
    
    def calculate_confidence_interval(self, success_rate, num_episodes, confidence_level=0.95):
        """计算置信区间（使用正态近似）"""
        import scipy.stats as stats
        
        z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        margin = z * np.sqrt((success_rate * (1 - success_rate)) / num_episodes)
        
        return margin, (max(0, success_rate - margin), min(1, success_rate + margin))
    
    def run_comprehensive_evaluation(self, num_episodes=1000):
        """运行全面的评估"""
        print("=" * 80)
        print(f"🎯 全面评估实验 - {num_episodes} 回合")
        print("=" * 80)
        
        start_time = time.time()
        
        # 1. 评估启发式基线
        heuristic_result = self.evaluate_heuristic(num_episodes)
        
        # 2. 评估BC模型
        bc_result = None
        try:
            from models import VLA_POLICY
            bc_policy = VLA_POLICY(action_dim=4, vocab_size=len(VOCAB)).to(self.device)
            bc_policy.load_state_dict(torch.load("bc_policy_structured.pth", map_location=self.device))
            bc_policy.eval()
            
            # 临时保存原始策略
            original_policy = self.policy
            original_model_path = self.model_path
            
            # 评估BC
            self.policy = bc_policy
            self.model_path = "bc_policy_structured.pth"
            bc_result = self.evaluate_policy(num_episodes)
            
            # 恢复原始策略
            self.policy = original_policy
            self.model_path = original_model_path
        except Exception as e:
            print(f"⚠️  无法评估BC模型: {e}")
        
        # 3. 评估PPO模型（如果存在）
        ppo_result = None
        if self.policy:
            ppo_result = self.evaluate_policy(num_episodes)
        
        # 计算置信区间
        results = []
        if heuristic_result:
            margin, ci = self.calculate_confidence_interval(
                heuristic_result['success_rate'], 
                heuristic_result['total_episodes']
            )
            heuristic_result['confidence_interval'] = ci
            heuristic_result['margin_of_error'] = margin
            results.append(heuristic_result)
        
        if bc_result:
            margin, ci = self.calculate_confidence_interval(
                bc_result['success_rate'], 
                bc_result['total_episodes']
            )
            bc_result['confidence_interval'] = ci
            bc_result['margin_of_error'] = margin
            results.append(bc_result)
        
        if ppo_result:
            margin, ci = self.calculate_confidence_interval(
                ppo_result['success_rate'], 
                ppo_result['total_episodes']
            )
            ppo_result['confidence_interval'] = ci
            ppo_result['margin_of_error'] = margin
            results.append(ppo_result)
        
        end_time = time.time()
        evaluation_time = end_time - start_time
        
        # 打印结果表格
        self.print_results_table(results, evaluation_time)
        
        return results
    
    def print_results_table(self, results, evaluation_time):
        """打印漂亮的评估结果表格"""
        print("\n" + "=" * 100)
        print("📈 综合评估结果 (1000回合)")
        print("=" * 100)
        
        print(f"{'模型':<30} {'成功率':<12} {'置信区间(95%)':<20} {'平均奖励':<12} {'平均步数':<12}")
        print("-" * 100)
        
        for result in results:
            model_name = result['model'][:28] + "..." if len(result['model']) > 28 else result['model']
            ci_str = f"[{result['confidence_interval'][0]:.3f}, {result['confidence_interval'][1]:.3f}]"
            
            print(f"{model_name:<30} "
                  f"{result['success_rate']:.3%}±{result['margin_of_error']:.3%} "
                  f"{ci_str:<20} "
                  f"{result['avg_reward']:.3f}±{result['std_reward']:.3f} "
                  f"{result['avg_steps_to_success']:.1f}")
        
        print("-" * 100)
        print(f"⏱️  总评估时间: {evaluation_time:.1f}秒")
        
        # 计算改进百分比
        if len(results) >= 2:
            print("\n📊 性能改进对比:")
            base_rate = results[0]['success_rate']
            for i, result in enumerate(results[1:], 1):
                improvement = ((result['success_rate'] - base_rate) / base_rate * 100)
                print(f"  {result['model']} 相对于启发式基线的改进: {improvement:+.1f}%")
        
        print("=" * 100)

if __name__ == "__main__":
    # 选择要评估的模型（优先使用PPO最终模型）
    models_to_evaluate = ["rl_policy_final.pth"]
    
    # 如果找不到PPO模型，尝试使用BC模型
    import os
    import glob
    
    available_models = []
    for model in models_to_evaluate:
        if os.path.exists(model):
            available_models.append(model)
    
    if not available_models:
        # 查找最新的PPO检查点
        checkpoints = glob.glob("rl_policy_iter*.pth")
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split('iter')[1].split('.pth')[0]))
            available_models.append(checkpoints[-1])
        elif os.path.exists("bc_policy_structured.pth"):
            available_models.append("bc_policy_structured.pth")
    
    if available_models:
        print(f"🔍 将评估模型: {available_models[0]}")
        evaluator = ComprehensiveEvaluator(model_path=available_models[0])
        evaluator.run_comprehensive_evaluation(num_episodes=1000)
    else:
        print("❌ 未找到任何模型文件。请先运行训练脚本。")
        print("   仅评估启发式基线...")
        evaluator = ComprehensiveEvaluator(model_path=None)
        evaluator.run_comprehensive_evaluation(num_episodes=1000)