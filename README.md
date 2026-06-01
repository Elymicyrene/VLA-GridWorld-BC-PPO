# VLA-GridWorld-BC-PPO

Vision-Language-Action Agent for Instruction Following in GridWorld: from Behavior Cloning to PPO Fine-tuning.

## 主要结果

| 模型 | 成功率 | 平均奖励 |
| --- | --- | --- |
| 随机基线 | 18% | -0.33 |
| 启发式专家 | 82% | 0.68 |
| BC（特征工程后） | **87%** | 0.756 |
| BC + PPO（最终） | **88%** | 0.766 |

## 项目结构

- `env.py` – GridWorld 环境（结构化特征）
- `models.py` – VLA 策略网络
- `train_bc.py` – 行为克隆训练
- `train_rl.py` – PPO 微调
- `test_bc_policy.py` – BC 模型评估
- `evaluate_final.py` – 最终模型评估
- `ppo_training_curve.png` – 训练曲线

## 快速开始

```bash
# 安装依赖
pip install torch torchvision numpy matplotlib gymnasium tqdm

# 运行基线
python part0_baselines.py

# 训练 BC
python train_bc.py

# 评估 BC
python test_bc_policy.py

# PPO 微调
python train_rl.py

# 评估最终模型
python evaluate_final.py
```

## 实验结果

PPO 微调后的策略成功率达到 **88%**，平均奖励 **0.766**，性能稳定超越启发式基线。

## 仓库链接

[https://github.com/Elymicyrene/VLA-GridWorld-BC-PPO](https://github.com/Elymicyrene/VLA-GridWorld-BC-PPO)
