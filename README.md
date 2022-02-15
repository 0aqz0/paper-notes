# paper-notes

## 2022.2.14

- HybrIK: A Hybrid Analytical-Neural Inverse Kinematics Solution for 3D Human Pose and Shape Estimation
  - 创新点：同时使用逆运动学和神经网络对人体3D位姿和形状进行估计，提高了求解的可解释权和利用了神经网络的灵活性，结合3D关键点估计和身体形状估计的优势
  - 和我的联系：利用单目视频估计人体位姿，既可以做基于视频的动作迁移，也可以做人体状态估计

## 2022.1.13

- Locomotion Skills for Simulated Quadrupeds
  - 创新点：提出基于物理仿真的四足机器人的控制框架（关节控制＋虚拟力控制、模型抽象）并实现各种步态和技能
  - 和我的联系：步态图通过归一化的相位表示，相似步态直接转换否则需要设计特定的转换控制器

## 2022.1.12

- Dynamics Randomization Revisited: A Case Study for Quadrupedal Locomotion
  - 创新点：提出动力学随机化对于仿真到实物的迁移不是必须的，更好的设计选择和关键参数随机化更重要
  - 和我的联系：不能盲目地加入动力学随机化，重点应该减少延时、电机模型、速度反馈上的差异

## 2022.1.7

- Legged Robots that Keep on Learning: Fine-Tuning Locomotion Policies in the Real World
  - 创新点：提出自动策略微调的框架（更高效的强化学习算法REDQ、自动重置控制器），使得机器人能够在实物环境中进一步学习
  - 和我的联系：机器人总会遇到新的环境，通过实物数据进一步学习，可以改善在未见过的实物环境的表现

## 2022.1.6

- Sim-to-Real Transfer of Robotic Control with Dynamics Randomization
  - 创新点：提出随机化仿真器的动力学参数，来减少仿真和现实世界之间的差异；通过循环神经网络的状态隐式地辨识系统的参数
  - 和我的联系：同样使用动力学参数随机化进行仿真到实物的迁移，尝试使用循环神经网络进行系统辨识

## 2022.1.5

- Graph networks as learnable physics engines for inference and control
  - 创新点：提出使用图神经网络作为可微分的物理引擎，用于动力学模型建模（图的节点表示物体，边表示物体之间的关系）、隐式系统辨识、基于模型的控制算法
  - 和我的联系：图神经网络的归纳偏置能够更好地学习物体之间的关系，隐式系统辨识通过一段状态轨迹得到系统属性的隐层表示

## 2022.1.4

- Reinforcement Learning with Evolutionary Trajectory Generator: A General Approach for Quadrupedal Locomotion
  - 创新点：提出进化轨迹生成器结合强化学习进行四足机器人运动控制，轨迹生成器通过进化策略优化轨迹
  - 和我的联系：参考轨迹＋强化学习的框架，不同之处在于通过进化策略对参考轨迹（初始轨迹通过CPG得到）进一步优化；都是从Pybullet仿真环境到实物机器人迁移（行为克隆、域适应、观测随机化）

## 2022.1.2

- Online Learning of Unknown Dynamics for Model-Based Controllers in Legged Locomotion
  - 创新点：提出在线的局部线性残差模型用于补偿模型的预测误差，解决动力学不准确或未知的问题
  - 和我的联系：实物和仿真的动力学不一致，实物机器人能够通过在线补偿模型减少和仿真的差距

## 2021.12.29

- Hierarchical visuomotor control of humanoids
  - 创新点：提出分层控制框架（底层电机控制器&高层任务控制器）实现仿人机器人视觉伺服控制
  - 和我的联系：底层电机控制框架也是强化学习训练得到，通过跟踪效果（位置、速度等）设计奖励

## 2021.12.28

- Reinforcement and Imitation Learning for Diverse Visuomotor Skills
  - 创新点：结合强化学习和模仿学习实现多样的视觉伺服技能学习（奖励设计同时包含强化学习奖励和模仿学习奖励；演示用于初始化回合；从状态中学习值网络；与物体有关、与机器人无关的辨别器；辅助任务帮助网络更好学习）
  - 和我的联系：结合强化学习和模仿学习，这篇论文中模仿学习通过GAIL实现

## 2021.12.27

- Learning human behaviors from motion capture by adversarial imitation
  - 创新点：通过对抗模仿学习GAIL（生成器用于生成动作，辨别器用于判断是否是人类动作），学习跟人的动作更接近的行为；只需要访问部分状态观测，不需要动作；通过语义标签学习基于任务的控制
  - 和我的联系：仿真中实现了仿人机器人行走，同样不需要访问动作只需要部分状态；通过语义标签能实现基于任务的多技能学习；环境初始状态可以从动捕数据中采样得到

## 2021.12.26

- Robust Imitation of Diverse Behaviors
  - 创新点：结合变分自编码器（能够建模多种多样的动作，但是学习的策略不够鲁棒）和对抗生成网络（学习的策略鲁棒，但是动作缺少多样性）实现鲁棒且多样的动作模仿
  - 和我的联系：GAIL能够帮助使用更少的演示样本进行模仿学习

## 2021.12.20

- Neural probabilistic motor primitives for humanoid control
  - 创新点：将大量鲁棒的专家策略压缩在一个共享的策略网络，学习运动基元的隐空间
  - 和我的联系：也使用了隐空间学习，不过重点在于解决如何学习一个通用的网络表达多种多样的专家策略

## 2021.12.16

- Actionable Models: Unsupervised Offline Reinforcement Learning of Robotic Skills
  - 创新点：提出机器人的通用目标是到达特定的目标状态，通过以目标为条件的Q-learning和后期重新标注，实现从已收集的数据中进行机器人技能的离线学习
  - 和我的联系：论文中提出了几个关键问题（1）如何定义机器人通用训练目标；（2）能否通过单个模型表示丰富多样的机器人技能；（3）能否将模型应用于零样本泛化或其他任务

## 2021.12.15

- Learning agile and dynamic motor skills for legged robots
  - 创新点：结合随机刚体建模、执行器网络、仿真强化学习，实现实物四足机器人技能学习
  - 和我的联系：首次提出执行器网络，通过监督学习将指令动作映射为机器人力矩，减少仿真和实物差异（相当于更加逼近了实物执行的效果）

## 2021.12.14

- RLOC: Terrain-Aware Legged Locomotion using Reinforcement Learning and Optimal Control
  - 创新点：结合强化学习（落脚点规划、实物适应跟踪、状态恢复）和最优控制（全身运动控制）实现复杂地形的四足机器人行走
  - 和我的联系：能够通过执行器网络对实物机器人动力学进行建模，实现仿真到实物迁移

## 2021.12.13

- Generative Adversarial Imitation Learning
  - 创新点：结合模仿学习和对抗生成网络，辨别器用于判断是专家动作还是策略动作，生成器生成动作用于混淆辨别器
  - 和我的联系：对抗生成网络也可以用于最小化仿真到实物的差异，生成器用于补偿差异，辨别器判断是仿真还是实物

## 2021.12.10

- Why Generalization in RL is Difficult: Epistemic POMDPs and Implicit Partial Observability
  - 创新点：分析强化学习难以泛化的原因（有限训练样本导致部分可观性，将MDP转化为POMDP），通过多个贝叶斯模型集成解决POMDP问题
  - 和我的联系：能够训练多个不同环境设定下的模型，集成得到一个对未知环境更鲁棒的模型

## 2021.11.22

- Embed to Control: A Locally Linear Latent Dynamics Model for Control from Raw Images
  - 创新点：通过VAE对高维的图像进行降维，在低维的隐空间上进行最优控制，并将非线性问题局部线性化，实现基于图像输入的控制
  - 和我的联系：隐空间的作用之一就是对高维数据进行降维，启发在隐空间上能够做更多事情，例如规划、控制

## 2021.11.16

- Planning in Learned Latent Action Spaces for Generalizable Legged Locomotion
  - 创新点：分层控制框架中高层控制器使用隐动作控制替代用户定义的动作
  - 和我的联系：隐动作规划时通过梯度下降寻找最佳隐动作（隐空间优化），底层控制使用模型预测控制
- Robot Motion Planning in Learned Latent Spaces
  - 创新点：在隐空间进行机器人运动规划（自编码器网络+动力学网络+碰撞检测网络），RRT在隐空间进行状态采样
  - 和我的联系：隐变量通过采样得到，隐空间可以起到降低状态维度的作用

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

