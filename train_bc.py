import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from models import VLA_POLICY
from env import GridWorldEnv
import numpy as np
import random

# Simple Tokenizer
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

class DemoDataset(Dataset):
    def __init__(self, num_samples=5000):
        self.data = []
        env = GridWorldEnv(max_steps=50)
        
        print(f"正在生成 {num_samples} 个高质量的专家演示...")
        generated_samples = 0
        
        while generated_samples < num_samples:
            obs, _ = env.reset()
            target_color = env.target_color
            target_pos = None
            for obj in env.objects:
                if obj['color'] == target_color:
                    target_pos = obj['pos']
                    break
            
            done = False
            step_count = 0
            
            while not done and generated_samples < num_samples:
                agent_pos = env.agent_pos
                
                # 计算启发式动作
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
                    action = random.randint(0, 3)
                
                # 存储特征向量，而不是原始图像
                features = torch.tensor(obs['features'], dtype=torch.float32)
                text = tokenize(obs['instruction'])
                
                self.data.append((features, text, torch.tensor(action, dtype=torch.long)))
                generated_samples += 1
                
                # 执行动作，进入下一个状态
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                step_count += 1
                
                if step_count > 100:  # 安全限制
                    break
                
                if generated_samples % 500 == 0:
                    print(f"  已生成 {generated_samples}/{num_samples} 个样本")
                    
                if generated_samples >= num_samples:
                    break
        
        print(f"✅ 专家数据生成完成，共 {len(self.data)} 个样本。")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def train_bc():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化策略
    policy = VLA_POLICY(action_dim=4, vocab_size=len(VOCAB)).to(device)
    
    # 使用较低的学习率和权重衰减
    optimizer = Adam(policy.parameters(), lr=3e-4, weight_decay=1e-5)
    
    # 使用标签平滑的交叉熵损失
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    dataset = DemoDataset(num_samples=5000)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    print("Starting Behavior Cloning with structured features...")
    
    for epoch in range(30):  # 训练30个epoch
        policy.train()  # 确保启用Dropout（如果模型中有）
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in dataloader:
            features, texts, actions = batch
            features = features.to(device)
            texts = texts.to(device)
            actions = actions.to(device)
            
            logits, _ = policy(features, texts)  # 注意：现在输入是特征，不是图像
            loss = criterion(logits, actions)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == actions).sum().item()
            total += actions.size(0)
            
        acc = correct / total
        print(f"Epoch {epoch+1:2d} | Loss: {total_loss/len(dataloader):.4f} | Acc: {acc:.2%}")
    
    torch.save(policy.state_dict(), "bc_policy_structured.pth")
    print("Model saved to bc_policy_structured.pth")

if __name__ == "__main__":
    train_bc()