# paper-notes

## 2021.11.2

- Concept2Robot: Learning Manipulation Concepts from Instructions and Human Demonstrations
  - 创新点：从人类指令和演示视频中学习机器人技能，在环境和指令上都具有泛化性
  - 和我的联系：启发能够通过训练好的分类器网络作为监督（奖励），结合自然语言进行机器人技能学习

### 2021.10.26

- Morphology-Agnostic Visual Robotic Control
  - 创新点：形态学未知（不精确执行、未知机器人形态、未知相机位姿）的视觉伺服控制（通过响应粒子获得自我认知）
  - 和我的联系：考虑结合视觉进行机器人控制

### 2021.10.25

- Learning Agile Locomotion via Adversarial Training
  - 创新点：对抗训练（一个追一个逃）学习四足机器人的敏捷运动
  - 和我的联系：状态设计包括关节角、关节角速度、相对对抗者的位置

### 2021.10.20

- Robots that can adapt like animals
  - 创新点：试错学习算法（建立行为-表现映射，并在试错中不断更新）帮助机器人快速适应损伤
  - 和我的联系：研究机器人快速学习适应新的场景（遭受损伤），优化找到最优策略（选择-测试-更新）

### 2021.10.19

- Data Efficient Reinforcement Learning for Legged Robots
  - 创新点：提出高效的腿足机器人强化学习框架（10min学习）——多步损失进行长序列预测，GPU并行采样进行实时规划，动力学模型进行模型预测补偿延时
  - 和我的联系：通过轨迹生成器（参考轨迹）鼓励更安全的探索，延时对控制效果影响很大（考虑规划控制分两个线程，异步控制）

### 2021.10.18

- Reinforcement Learning with Evolutionary Trajectory Generator: A General Approach for Quadrupedal Locomotion
  - 创新点：相对于固定步态+强化学习控制器框架，步态通过进化算法进一步优化
  - 和我的联系：步态不一定完美适合特定的任务，可以考虑进行优化，行为克隆、域随机化、观测随机化实现仿真到实物迁移

### 2021.10.15

- Sim-to-Real: Learning Agile Locomotion For Quadruped Robots
  - 创新点：固定步态＋强化学习控制器框架，四足机器人仿真到实物迁移（电机模型建模，仿真延时，动力学随机化）
  - 和我的联系：框架类似（参考轨迹＋强化学习控制器），状态设计（roll、pitch、两轴角速度和八个电机角度）可以考虑去掉yaw角，考虑加入仿真延时

### 2021.10.8

- LEO: Learning Energy-based Models in Graph Optimization
  - 创新点：将观测模型学习问题建模为基于能量学习，解决图优化器不可微的限制
  - 和我的联系：不同于基于梯度优化固定的目标函数，基于能量学习的方法改变目标函数的形状（真值处低，其他高），增量式高斯牛顿法进行实时推理

