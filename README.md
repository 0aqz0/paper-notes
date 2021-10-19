# paper-notes

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
  - 创新点：
  - 和我的联系：

