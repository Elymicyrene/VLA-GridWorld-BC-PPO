import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageEncoder(nn.Module):
    """注意：此图像编码器在新的特征工程方案中已不再需要。
       保留此类仅是为了避免导入错误，但VLA_POLICY将不再使用它。"""
    def __init__(self, output_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        # 注意：此处的线性层输入维度需要根据实际卷积输出调整，此处仅为示例
        self.fc = nn.Linear(64 * 8 * 8, output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.fc(x)
        return x

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.rnn(embedded)
        return hidden.squeeze(0)

class VLA_POLICY(nn.Module):
    """修改后的策略网络，接受结构化特征输入而非原始图像。"""
    def __init__(self, action_dim=4, vocab_size=100):
        super().__init__()
        # 文本编码器 (可适当降低维度，因为任务变简单了)
        self.text_encoder = TextEncoder(vocab_size=vocab_size, embedding_dim=32, hidden_dim=64)
        
        # 特征融合层
        # 输入维度: 特征(7) + 文本编码(64) = 71
        self.fusion = nn.Sequential(
            nn.Linear(7 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # 输出头
        self.action_head = nn.Linear(64, action_dim)
        self.value_head = nn.Linear(64, 1)  # 为PPO保留

    def forward(self, features, text):
        """
        参数:
            features: (batch, 7) 结构化特征向量
            text: (batch, seq_len) 文本指令
        返回:
            logits: (batch, action_dim) 动作logits
            value: (batch, 1) 状态价值估计
        """
        # 编码文本
        txt_emb = self.text_encoder(text)
        
        # 拼接特征和文本嵌入
        combined = torch.cat([features, txt_emb], dim=1)
        
        # 融合特征
        x = self.fusion(combined)
        
        # 输出动作logits和价值估计
        logits = self.action_head(x)
        value = self.value_head(x)
        
        return logits, value

    def act(self, features, text, deterministic=False):
        """用于评估或PPO数据收集的便捷方法。"""
        logits, _ = self.forward(features, text)
        probs = F.softmax(logits, dim=-1)
        
        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            
        return action