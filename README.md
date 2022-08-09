# paper-notes

## 2022.8.8

- MixSTE: Seq2seq Mixed Spatio-Temporal Encoder for 3D Human Pose Estimation in Video
  - 创新点：提出使用时间transformer建模每个关节的时序动作和空间transformer学习关节之间的空间关系，以更好地学习时空特征编码
  - 和我的联系：考虑使用transformer模型学习时序特征，每个关节可以分开单独建模

## 2022.8.7

- HuMoR: 3D Human Motion Model for Robust Pose Estimation
  - 创新点：提出使用VAE作为生成模型学习人体位姿变化的分布，并将其作为动作先验从不同模态的观测中优化人体位姿估计的结果
  - 和我的联系：学习位姿的变化量而不是绝对量，使用过去两帧的数据作为输入，训练损失包括重构损失、KL损失、正则化损失

## 2022.8.6

- Probabilistic Modeling for Human Mesh Recovery
  - 创新点：提出预测人体3D位姿的分布而不是确定量，不同的任务场景（估计3D位姿、结合2D关键点、多视角融合等）可以通过最大化分布概率和特定的任务损失实现
  - 和我的联系：人体3D位姿的分布通过Normalizing Flows进行表示而不是VAE，可以进行采样和概率计算

## 2022.8.5

- What Matters in Learning from Offline Human Demonstrations for Robot Manipulation
  - 创新点：分析从人类离线数据中学习机器人操作技能的关键挑战，并提出在算法设计、示教数据质量等方面的建议
  - 和我的联系：依赖历史数据的模型更加高效，观测空间和超参数的选择很重要，使用大规模人类数据集在复杂任务上表现更好

## 2022.7.22

- Learning 3D Human Dynamics from Video
  - 创新点：提出从视频中学习3D人体动力学模型对未来进行预测；能够只用2D的标签进行半监督学习（2D投影损失和对抗性先验损失）；通过2D位姿检测器生成标签扩充数据集
  - 和我的联系：学习时序信息有利于减少不确定性和抖动，可参考输入多帧的数据；相当于卡尔曼滤波的运动模型，预测的是变化量而不是绝对量，但是没有构建隐空间；训练VAE的数据可以参考使用动捕数据集

## 2022.7.21

- Tracking People with 3D Representations
  - 创新点：提出使用3D人体表示（外表、3D位姿）而不是2D人体表示跟踪视频中的人
  - 和我的联系：都能估计3D人体位姿；人体位姿通过网络编码的特征向量表示；没有对未来预测

## 2022.7.20

- Human Mesh Recovery from Multiple Shots
  - 创新点：解决从包含跳变的视频中恢复人体蒙皮的问题，将问题建模为不同视角下的同一场景；将不同视角下的人体模型转换到标准视角，对平滑性进行优化，生成数据集用于训练
  - 和我的联系： 可用于复杂场景下（视角变化、遮挡等）的人体蒙皮恢复任务

## 2022.7.19

- Forecasting Characteristic 3D Poses of Human Actions
  - 创新点：提出预测语义上有意义的人体位姿（而不是一定时间间隔后的人体位姿），基于注意力机制设计编码器网络，并将关节角度概率分布建模为多个条件概率相乘
  - 和我的联系：更关注于语义位姿预测任务，语义位姿有助于实现长时间预测

## 2022.7.18

- Tracking People by Predicting 3D Appearance, Location & Pose
  - 创新点：解决从单目视频中跟踪人体3D位姿问题，首先从单帧中估计3D位姿并进行预测，使用匈牙利算法对观测进行数据关联
  - 和我的联系：整体框架接近（估计3D状态并进行预测，融合2D观测）；预测使用的是恒定速度模型；跟踪的不只是位姿还包括外观，能够多人跟踪；3D位姿表示是通过实际物理量来表征

## 2022.7.17

- MASD: A Multimodal Assembly Skill Decoding System for Robot Programming by Demonstration
  - 创新点：提出多模态分层模型用于识别演示动作，以及优化算法将装配技能分割成多个动作
  - 和我的联系：一段装配演示视频中包含多个技能，单个技能又包含多个动作；动作集是预定义的

## 2022.7.12

- Stacked Hourglass Networks for Human Pose Estimation
  - 创新点：提出沙漏型网络结构设计（先池化然后上采样和残差），提升2D人体位姿估计算法效果
  - 和我的联系：可作为2D人体位姿模块提供卡尔曼滤波的观测

## 2022.7.11

- DoubleFusion: Real-time Capture of Human Performances with Inner Body Shapes from a Single Depth Sensor
  - 创新点：提出双层的人体表面表示方法，同时优化内部人体模型参数和外表面参数，实现实时从深度相机重建人体表面和跟踪关节动作
  - 和我的联系：关注于人体表面重建问题，也需要初始化人体模型

## 2022.7.7

- Monocap: Monocular human motion capture using a CNN coupled with a geometric prior
  - 创新点：解决从无标记点的单目RGB图像中恢复人体三维位姿的问题，融合2D、3D和时序信息恢复人体三维位姿，2D信息通过CNN获得人体关节热力图，3D信息通过动作库获得人体骨架位姿的先验，EM算法最大化人体位姿的期望
  - 和我的联系：解决从2D到3D的问题；构建更自然的人体位姿模型，通过对字典中人体位姿进行加权实现

## 2022.7.3

- Mastering Atari with Discrete World Models
  - 创新点：提出直接从世界模型在隐空间上的预测进行学习的强化学习算法
  - 和我的联系：隐空间上构建世界模型，在隐空间上的动力学模型可实现长期的预测；世界模型由编码器、循环状态空间模型、预测器组成

## 2022.7.2

- Perception of Demonstration for Automatic Programing of Robotic Assembly: Framework, Algorithm, and Validation
  - 创新点：提出概率感知框架同时推理装配演示中的动作和零件，动作通过SVM分类器分类，零件通过期望最大化算法估计位姿
  - 和我的联系：PBD任务需要机器人通过传感器观测来理解人类专家的演示，并生成学习反馈

## 2022.7.1

- DayDreamer: World Models for Physical Robot Learning
  - 创新点：提出世界模型预测未来用于减少与实际环境交互的次数，世界模型在隐空间上对未来进行预测
  - 和我的联系：世界模型通过编码器将状态编码成隐变量，预测下一步隐变量，并使用解码器解码未来状态

## 2022.6.30

- Masked World Models for Visual Control
  - 创新点：将机器人视觉控制问题解耦为视觉表示学习、动力学模型学习，视觉表示学习使用掩码自编码器实现，动力学模型直接作用于自编码器得到的视觉表示
  - 和我的联系：自编码器构建隐空间，动力学模型在隐空间上进行状态转移，可参考框架实现细节

## 2022.6.29

- RaLL: End-to-end Radar Localization on Lidar Map Using Differentiable Measurement Model
  - 创新点：解决使用毫米波在雷达地图上进行定位的问题，将两种模态编码到共享的特征空间用于计算偏置的概率得到观测，使用可微分的卡尔曼滤波进行端对端训练
  - 和我的联系：可以参考可微分卡尔曼滤波的框架进行人体位姿估计

## 2022.6.28

- NeMF: Neural Motion Fields for Kinematic Animation
  - 创新点：使用隐空间表征一段连续的动作（通过输入t获得对应时刻的动作），并使用随机向量z控制动作风格，进而网络能够表征完整的连续动作空间
  - 和我的联系：隐空间既可以表征单帧的动作，也可以用于表征连续动作；生成模型通过随机变量进行生成

## 2022.6.27

- BlazePose: On-device Real-time Body Pose tracking
  - 创新点：提出轻量级人体位姿估计算法能够在移动设备以每秒30帧运行
  - 和我的联系：可以作为2D人体位姿检测模块用于卡尔曼滤波

## 2022.6.10

- Robots State Estimation and Observability Analysis Based on Statistical Motion Models
  - 创新点：解决拓展卡尔曼滤波中控制量未知问题，通过白噪声统计模型代替控制量，并进行能观性分析
  - 和我的联系：与人体位姿估计的建模方式类似，通过噪声近似模拟控制量，控制量需要更好建模

## 2022.5.29

- Imitation and Adaptation Based on Consistency: A Quadruped Robot Imitates Animals from Videos Using Deep Reinforcement Learning
  - 创新点：解决从视频中模仿四足机器人运动问题，从视频中提取关节关键点，通过动作适应模块生成参考轨迹，强化学习训练策略网络补偿参考轨迹并保持平衡
  - 和我的联系：基本思路是参考轨迹加强化学习补偿，实物实验通过域随机化实现

## 2022.5.28

- DPCN++: Differentiable Phase Correlation Network for Versatile Pose Registration
  - 创新点：解决不同模态的数据2D-3D位姿匹配问题，通过可微分的相位匹配模块解耦并求解位移、旋转、尺度7个自由度的相对位姿关系，结合特征提取网络进行端对端训练
  - 和我的联系：基本思路是通过可学习参数的网络结合无学习参数可微分的模块进行训练

## 2022.5.27

- Capturing Hands in Action using Discriminative Salient Points and Physics Simulation
  - 创新点：解决手部动作捕捉中跟其他手或者物体的交互问题，通过结合生成模型和关键点检测、碰撞检测、物理仿真，构建目标函数解优化问题
  - 和我的联系：涉及mesh碰撞检测可以用于动画人物重定向任务中减少穿模

## 2022.4.21

- Learning Character-Agnostic Motion for Motion Retargeting in 2D
  - 创新点：解决视频2D-2D动作迁移（不同骨架、动作、相机视角）问题，采用多编码器/单解码器结构，多编码器解耦骨架、动作、相机视角，单解码器根据不同的组合重建动作
  - 和我的联系：输入输出模态不同，使用视频生成视频，可参考尝试视频输入

## 2022.4.20

- Neural kinematic networks for unsupervised motion retargetting
  - 创新点：解决无监督数据的动作迁移问题，设计可微分的正运动学层使其能够直接优化关节旋转而无需真值，使用循环一致性损失（类似GAN）进行对抗学习
  - 和我的联系：同样是使用可微分正运动学模块，设计无监督损失训练；损失函数可以尝试循环一致性损失

## 2022.4.19

- Skeleton-aware networks for deep motion retargeting
  - 创新点：解决如何对不同拓扑结构（骨架长度、节点数）的骨架进行动作迁移的问题，假设不同结构对应相同的原始骨架（共享隐空间），再通过对应解码器生成动作，设计基于骨架的池化卷积算子
  - 和我的联系：相同之处为按照图的方式进行建模和卷积，编码器-解码器架构；不同在于通过非配对数据训练目标骨架解码器，必须包含目标骨架的动作数据

## 2022.4.18

- Contact-Aware Retargeting of Skinned Motion
  - 创新点：解决动作迁移中如何保留自接触和减少穿模的问题，通过基于几何的循环神经网络生成初解，再通过定义的能量函数优化（能量函数中考虑了自接触、减少穿模、地面接触、运动相似性等）
  - 和我的联系：同样是在隐空间进行优化，优化的目标函数不同；动画人物的减少穿模、自接触也可以使用约束进行描述

## 2022.3.16

- How2Sign: A Large-scale Multimodal Dataset for Continuous American Sign Language
  - 创新点：提出多模态、多视角、大规模连续手语数据集How2Sign，包含语义、翻译、RGB、深度等模态
  - 和我的联系：数据集的多样性足够丰富可用于训练前馈神经网络，视频模态可用于进行基于视频的手语动作迁移学习

## 2022.3.14

- FrankMocap: A Monocular 3D Whole-Body Pose Estimation System via Regression and Integration
  - 创新点：提出基于单目视觉的全身位姿估计算法（身体、手、脸），思路是不同部分分别回归最后进行融合，既保留了估计精度也提供了统一的位姿结果
  - 和我的联系：整合这一模块可实现基于单目视觉的动作迁移

## 2022.3.2

- Force-feedback based Whole-body Stabilizer for Position-Controlled Humanoid Robots
  - 创新点：同时考虑仿人机器人全身动力学和六维力信号，解二次规划实现重心跟踪和接触力跟踪
  - 和我的联系：仿人机器人可分为位置控制和力矩控制两类，位置控制一般框架分为规划器（生成参考轨迹）、稳定器（跟踪重心和接触力）、逆运动学（生成对应关节位置）、关节伺服控制器（转换为力矩）

## 2022.3.1

- Self-Supervised Motion Retargeting with Safety Guarantee
  - 创新点：自监督方式生成数据进行训练，非参数回归保证动作可行性、安全性
  - 和我的联系：通过自监督学习产生数据，能够避免由于数据量不足导致算法性能下降，类似于数据增广；安全性保证、从RGB视频进行动作迁移可以参考

## 2022.2.16

- Stair Climbing Stabilization of the HRP-4 Humanoid Robot using Whole-body Admittance Control
  - 创新点：提出结合末端和质心策略实现全身的阻抗控制，解决仿人双足机器人在开放工厂环境上楼梯问题
  - 和我的联系：研究仿人双足机器人行走问题，采用传统控制算法（DCM控制+全身阻抗控制）

## 2022.2.15

- Deep learning can accelerate grasp-optimized motion planning
  - 创新点：使用神经网络拟合优化结果作为优化的初值，从初值开始进一步优化，加快优化收敛速度，并进一步提升效果
  - 和我的联系：同样使用神经网络得到优化的初值，但是不同之处在于优化使用的是二次规划算法，任务是物体抓取和放置

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

## 2021.10.26

- Morphology-Agnostic Visual Robotic Control
  - 创新点：形态学未知（不精确执行、未知机器人形态、未知相机位姿）的视觉伺服控制（通过响应粒子获得自我认知）
  - 和我的联系：考虑结合视觉进行机器人控制

## 2021.10.25

- Learning Agile Locomotion via Adversarial Training
  - 创新点：对抗训练（一个追一个逃）学习四足机器人的敏捷运动
  - 和我的联系：状态设计包括关节角、关节角速度、相对对抗者的位置

## 2021.10.20

- Robots that can adapt like animals
  - 创新点：试错学习算法（建立行为-表现映射，并在试错中不断更新）帮助机器人快速适应损伤
  - 和我的联系：研究机器人快速学习适应新的场景（遭受损伤），优化找到最优策略（选择-测试-更新）

## 2021.10.19

- Data Efficient Reinforcement Learning for Legged Robots
  - 创新点：提出高效的腿足机器人强化学习框架（10min学习）——多步损失进行长序列预测，GPU并行采样进行实时规划，动力学模型进行模型预测补偿延时
  - 和我的联系：通过轨迹生成器（参考轨迹）鼓励更安全的探索，延时对控制效果影响很大（考虑规划控制分两个线程，异步控制）

## 2021.10.18

- Reinforcement Learning with Evolutionary Trajectory Generator: A General Approach for Quadrupedal Locomotion
  - 创新点：相对于固定步态+强化学习控制器框架，步态通过进化算法进一步优化
  - 和我的联系：步态不一定完美适合特定的任务，可以考虑进行优化，行为克隆、域随机化、观测随机化实现仿真到实物迁移

## 2021.10.15

- Sim-to-Real: Learning Agile Locomotion For Quadruped Robots
  - 创新点：固定步态＋强化学习控制器框架，四足机器人仿真到实物迁移（电机模型建模，仿真延时，动力学随机化）
  - 和我的联系：框架类似（参考轨迹＋强化学习控制器），状态设计（roll、pitch、两轴角速度和八个电机角度）可以考虑去掉yaw角，考虑加入仿真延时

## 2021.10.8

- LEO: Learning Energy-based Models in Graph Optimization
  - 创新点：将观测模型学习问题建模为基于能量学习，解决图优化器不可微的限制
  - 和我的联系：不同于基于梯度优化固定的目标函数，基于能量学习的方法改变目标函数的形状（真值处低，其他高），增量式高斯牛顿法进行实时推理
