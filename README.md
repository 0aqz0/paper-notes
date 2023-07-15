# paper-notes

## 2023.7.15

- Learning Bipedal Walking On Planned Footsteps For Humanoid Robots
  - 创新点：提出以落足点为输入的强化学习算法学习仿人机器人双足行走，能够实现全向走路、转弯、站立、爬楼梯
  - 和我的联系：仿人机器人行走动作强化学习，不需要参考动作和演示

## 2023.7.13

- VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models
  - 创新点：提出结合语言大模型和视觉语言大模型，零样本学习实现面向开集物体的开集操作指令（大模型生成可操作性和约束价值地图，作为目标函数进行运动规划生成机器人末端轨迹）
  - 和我的联系：可结合大模型监督动作生成

##  2023.6.29

- Intuitive and Versatile Full-body Teleoperation of A Humanoid Robot
  - 创新点：提出了易操作且能实现全身运动控制的仿人机器人遥操作系统（人类穿戴VR头盔、惯性动作捕捉服，上本身动作映射，下半身动作模式判断）
  - 和我的联系：动作映射只考虑上半身，通过向量代数的方式计算关节角

## 2023.6.28

- System Design and Balance Control of a Novel Electrically-driven Wheel-legged Humanoid Robot
  - 创新点：提出轮式腿足机器人同时实现轮式运动和行走运动（机械结构设计、动力驱动设计、平衡控制算法）
  - 和我的联系：全身稳定控制采用线性二次型调节器实现

## 2023.6.26

- Trace and Pace: Controllable Pedestrian Animation via Guided Trajectory Diffusion
  - 创新点：提出用户可控制的行人轨迹动画生成算法，由生成轨迹的扩散模型和基于物理仿真的仿人控制器组成
  - 和我的联系：在基于物理的动作模仿的基础上加入了轨迹生成；动作模仿考虑了自身状态、环境特征、身体特征和目标轨迹

## 2023.6.21

- Perpetual Humanoid Control for Real-time Simulated Avatars
  - 创新点：提出使用单一策略进行动作模仿（渐进地学习运动基元，并通过组合器进行切换），能够实现实时虚拟角色控制，并从失败状态中恢复，以及跟动作估计和生成算法结合
  - 和我的联系：能够适用于不同动作（在动作上泛化），缺点是训练成本高昂

## 2023.6.20

- Aura Mesh: Motion Retargeting to Preserve the Spatial Relationships between Skinned Characters
  - 创新点：提出交互空间（对蒙皮进行膨胀，膨胀后的蒙皮进行碰撞检测）作为多个虚拟角色交互的语义表征，实现蒙皮层面（不是骨架层面）的交互语义保持
  - 和我的联系：语义表征的一种方式，适用于多个角色的动作语义

## 2023.6.12

- Learning Human Mesh Recovery in 3D Scenes
  - 创新点：提出无需优化且考虑场景的人体蒙皮恢复算法（场景接触作为点云分类任务，交叉注意力构建人体位姿和场景几何的联合分布）
  - 和我的联系：考虑了人体跟场景的接触，每个顶点判断是否发生接触

## 2023.6.8

- AdaSGN: Adapting Joint Number and Model Size for Efficient Skeleton-Based Action Recognition
  - 创新点：提出自适应减少骨架关节点数量以降低计算成本，实现轻量级的动作识别
  - 和我的联系：自适应改变关节点数量，预定义不同关节点数量并通过策略网络进行选择

## 2023.6.2

- BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models
  - 创新点：提出高效的视觉-语言预训练策略用来降低大模型高昂的训练成本，冻结大模型的参数，采用查询Transformer构建视觉表征和语言表征之间的关系
  - 和我的联系：视觉表征和语言表征可推广到动作表征和语言表征，构建动作语义关系

## 2023.6.1

- Multicontact Motion Retargeting Using Whole-Body Optimization of Full Kinematics and Sequential Force Equilibrium
  - 创新点：提出多接触动作重定向框架用于高自由度机器人遥操作（满足物理约束、安全平衡），将多接触动作构建为序列二次规划问题（非线性优化求解）
  - 和我的联系：在动作重定向中考虑与环境接触，接触通过接触力进行表征

## 2023.5.15

- Skeleton-Based Action Recognition with Directed Graph Neural Networks
  - 创新点：提出基于有向图神经网络的动作识别算法，有向无环图建模人体关节骨骼，通过更新邻接矩阵参数自适应修改拓扑图结构
  - 和我的联系：图的方式建模人体；自适应拓扑结构

## 2023.5.14

- CALM: Conditional Adversarial Latent Models for Directable Virtual Characters
  - 创新点：提出有条件的对抗隐模型用于生成多样的用户交互的虚拟角色动作，顶层策略在隐空间控制底层策略，通过有限状态机组合不同的策略模型
  - 和我的联系：基于物理仿真的虚拟角色动作生成，分层控制架构

## 2023.5.13

- Transfer Learning of Shared Latent Spaces between Robots with Similar Kinematic Structure
  - 创新点：提出共享高斯过程隐变量模型（每种机器人共享隐空间，但对应不同的解码器参数），用于结构相似机器人的知识迁移
  - 和我的联系：共享隐空间的思想，不足是需要为每种机器人设计解码器

## 2023.5.10

- SimPoE: Simulated Character Control for 3D Human Pose Estimation
  - 创新点：提出结合图像运动学推理和基于物理动力学建模的三维人体位姿估计算法（生成运动学估计后，再控制仿真角色执行），实现物理可行的人体位姿估计
  - 和我的联系：先运动学后动力学的思路，可参考自动生成物理仿真模型


## 2023.5.9

- A Scalable Approach to Control Diverse Behaviors for Physically Simulated Characters
  - 创新点：提出为多样的异构动作学习统一的控制器模型（动作聚类，每种类别训练专家策略，然后训练一个通用策略）
  - 和我的联系：解决了之前的方法每个动作单独训练一个模型的问题，可用于通用策略学习

## 2023.5.8

- MoDi: Unconditional Motion Synthesis from Diverse Data
  - 创新点：提出从多样化非结构无标签数据中训练的无监督动作生成算法，基于StyleGAN结构支持语义聚类、编辑等下游任务
  - 和我的联系：编码器能将真实动作映射到正常自然的隐空间，避免病态问题

## 2023.5.7

- Skinned Motion Retargeting with Residual Perception of Motion Semantics & Geometry
  - 创新点：提出残差动作重定向网络保留动作语义（关节归一化距离矩阵）、接触信号和减少穿模（排斥和吸引距离场），先进行关节角复制再进行微调
  - 和我的联系：动作重定向中考虑了动作语义，不足是通过人为定义的

## 2023.5.6

- DiffMimic: Efficient Motion Mimicking with Differentiable Physics
  - 创新点：提出采用可微分物理仿真进行动作模仿，化简为状态匹配问题并直接通过梯度优化策略，并通过演示重放策略减少局部最优和长序列梯度爆炸/消失
  - 和我的联系：相对于强化学习算法不需要奖励设计，具有更高的样本效率和收敛速度

## 2023.5.5

- AvatarGen: a 3D Generative Model for Animatable Human Avatars
  - 创新点：提出可控制外表、姿态和相机视角的非刚体虚拟形象生成算法，使用二维图像训练而不需要多视角三维数据训练
  - 和我的联系：不同于预定义的模板驱动虚拟角色，采用参数化人体模型引导的三平面表征并进行图像渲染

## 2023.5.4

- Residual Force Control for Agile Human Behavior Imitation and Extended Motion Synthesis
  - 创新点：提出使用虚拟力对仿人模型跟人类模型动力学不一致进行补偿，实现模仿复杂人类动作和动作生成（运动学策略生成未来动作+控制策略在物理仿真模仿）
  - 和我的联系：对强化学习模仿动作进行改进，拓展至更复杂的动作；无法应用于实物机器人

## 2023.5.3

- Learning Physically Simulated Tennis Skills from Broadcast Videos
  - 创新点：提出从视频中学习物理仿真角色网球技能的算法，由视频动作估计（人体位姿估计）、物理仿真修正（物理仿真训练模仿策略）、动作表征（动作变分自编码器）和上层策略控制（直接输出隐变量和关节修正量）四部分组成
  - 和我的联系：多种动作模仿算法整合成系统；物理仿真修正部分与其他工作类似

## 2023.4.2

- Learning Humanoid Locomotion with Transformers
  - 创新点：提出完全基于学习算法的仿人机器人控制算法（Transformer+RL+两阶段训练），并直接进行实物迁移（域随机化）
  - 和我的联系：Transformers能够进行in-context learning而不需要再微调

## 2023.4.1

- ASE: Large-Scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters
  - 创新点：提出为物理仿真角色学习可复用技能编码的算法（预训练和任务训练），克服每个任务从头开始训练、训练难度大、任务奖励设计复杂耗时、容易产生不自然的动作等问题
  - 和我的联系：可用于训练通用的动作模仿，顶层策略在隐空间控制底层策略

## 2023.2.1

- PMnet: Learning of Disentangled Pose and Movement for Unsupervised Motion Retargeting
  - 创新点：提出将局部位姿和全局运动解耦的动作重定向算法，并对全局运动进行归一化处理
  - 和我的联系：类似的架构（局部旋转和根关节位移）和归一化处理，不足是只能处理单一拓扑

## 2023.1.31

- VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training
  - 创新点：提出简单高效的视频掩码自编码器，用于视频自监督预训练（加入掩码然后训练重构任务）
  - 和我的联系：掩码自监督训练可用于在小数据集上更高效地学习特征

## 2023.1.30

- DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills
  - 创新点：提出使用强化学习算法进行物理仿真角色的动作模仿，结合动作模仿奖励和任务目标奖励
  - 和我的联系：在物理仿真中进行动作模仿的经典算法（强化学习）

## 2023.1.29

- MotionCLIP: Exposing Human Motion Generation to CLIP Space
  - 创新点：提出将动作隐空间跟CLIP隐空间对齐（文本、图像两种损失）以利用语义知识和解耦表征空间，可用于文本-动作、动作编辑等下游任务
  - 和我的联系：利用预训练大模型结合动作和语义信息；不足是需要文本标注以及使用单帧图像无时序信息

## 2023.1.28

- PADL: Language-Directed Physics-Based Character Control
  - 创新点：提出通过自然语言交互控制物理仿真角色的算法，由技能表征（动作、自然语言共享隐空间）、策略训练（使用隐变量加强化学习训练仿真角色）和多任务聚合（多选择问答确定调用哪种策略）三部分组成
  - 和我的联系：加入自然语言作为交互方式，但限制在任务+技能；策略训练采用对抗模仿学习

## 2023.1.27

- DReCon: Data-Driven Responsive Control of Physics-Based Characters
  - 创新点：提出用户实时控制物理仿真角色的算法，由运动学控制器（动作匹配）和仿真角色控制器（强化学习）两部分组成
  - 和我的联系：自动生成物理仿真角色以匹配人类身体结构；基于强化学习训练物理角色控制器

## 2023.1.26

- Imitation Learning of Dual-Arm Manipulation Tasks in Humanoid Robots
  - 创新点：提出使用隐马尔科夫模型学习人类演示动作并在双臂仿人机器人复现
  - 和我的联系：隐马尔科夫模型是深度学习兴起前的重要方法之一

## 2023.1.25

- Gaze-Based Dual Resolution Deep Imitation Learning for High-Precision Dexterous Robot Manipulation
  - 创新点：提出利用人类注视数据学习穿针任务，先用低分辨率图像靠近目标，再用高分辨率图像精准控制穿针
  - 和我的联系：涉及灵巧操作任务、不同状态切换、从失败状态恢复

## 2023.1.24

- Transformer-based deep imitation learning for dual-arm robot manipulation
  - 创新点：提出采用Transformer的自注意力架构关注重要的输入特征，用于解决双臂操作任务
  - 和我的联系：可参考改进网络模型结构

## 2023.1.23

- Generating Diverse and Natural 3D Human Motions from Text
  - 创新点：提出从文本中生成多样和变长的人类动作分布（动作自编码器、从文本预测动作长度、从文本和动作长度生成动作），并构建了大规模人类动作数据集
  - 和我的联系：可作为从文本到动作的基准模型

## 2023.1.22

- Learning Transferable Visual Models From Natural Language Supervision
  - 创新点：提出从自然语言中学习视觉表征实现零样本学习，图像和文本编码器学习表征并计算相似性
  - 和我的联系：对比表征学习（图像和文本的相似性分数）；文本编码器提取语义表征

## 2023.1.21

- Action2Motion: Conditioned Generation of 3D Human Motions
  - 创新点：提出给定动作类别生成人类动作的问题（动作识别问题的逆过程），采用时序变分自编码器结构作为生成模型，以及李代数表示旋转
  - 和我的联系：可提供动作类别到人类动作的数据集

## 2023.1.20

- MotioNet: 3D Human Motion Reconstruction from Monocular Video with Skeleton Consistency
  - 创新点：提出从单目视频中重建人体动作，将视频中提取的二维关节位置映射为静态骨架、动态旋转数据，确保骨架一致性
  - 和我的联系：结合人体位姿估计和动作重定向；使用关节角速度计算对抗损失以获得合理的关节速度

## 2023.1.19

- Transferring and Animating a non T-pose Model to a T-pose Model
  - 创新点：提出从任意位姿转换为T pose的方法，计算起始和最终位姿再线性插值
  - 和我的联系：可用于生成位姿转换过程的中间位姿

## 2023.1.18

- Executing your Commands via Motion Diffusion in Latent Space
  - 创新点：结合变分自编码器和扩散模型，在降维的隐空间上表征动作和条件输入（文本、动作类别），加快模型收敛和计算时间
  - 和我的联系：可参考文本-动作、动作-动作网络设计结构

## 2023.1.17

-  Human Motion Diffusion Model
  - 创新点：提出动作扩散模型用于人类动作生成，可用于文本到动作、动作到动作、动作补全、动作修改等任务
  - 和我的联系：可参考文本编码的方式提取文本表征

## 2023.1.16

- Teleoperation of Humanoid Robots: A Survey
  - 创新点：仿人机器人遥操作的综述（遥操作系统和设备、仿人机器人重定向和控制、辅助遥操作、通讯信道、评估指标、应用和发展前景）
  - 和我的联系：系统架构基本由重定向和规划、稳定器、全身控制器、关节控制组成

## 2023.1.13

- Denoising Diffusion Probabilistic Models
  - 创新点：提出DDPM模型，重参数化并简化扩散模型的目标函数，建立扩散概率模型和降噪分数匹配之间的关联
  - 和我的联系：可作为SOTA生成模型使用（图像、动作等）

## 2023.1.12

- RigNet: Neural Rigging for Articulated Characters
  - 创新点：提出端到端的角色骨架绑定网络（骨架关节预测、关节连接性预测、蒙皮权重预测）
  - 和我的联系：可尝试将角色蒙皮面片作为输入，GMEdgeNet可用于提取角色形状特征

## 2023.1.11

- PhysDiff: Physics-Guided Human Motion Diffusion Model
  - 创新点：提出考虑物理约束的扩散模型，使用物理仿真训练的动作模仿策略加入物理约束
  - 和我的联系：可尝试加入物理仿真器或者物理约束解决滑步、穿模等问题

## 2023.1.10

- Pretrained Diffusion Models for Unified Human Motion Synthesis
  - 创新点：提出统一的人体动作生成模型，能够适用于不同的任务和不同的骨架（长度比例、初始姿态、关节点个数）
  - 和我的联系：骨架适应器相当于不同骨架之间的重定向，直接处理旋转角，排列矩阵类似于旋转角复制，需要监督数据进行优化，再优化后相当于一对一不能共享参数，使用拓扑信息计算解释性更强

## 2023.1.9

- Skeleton-free Pose Transfer for Stylized 3D Characters
  - 创新点：提出不需要骨架的角色位姿迁移方法，将角色模型分解为独立部分，预测角色的蒙皮权重和位姿变换矩阵
  - 和我的联系：可尝试不只是优化关节角，同时优化蒙皮权重和变换矩阵

## 2022.12.15

- BEAT: A Large-Scale Semantic and Emotional Multi-Modal Dataset for Conversational Gestures Synthesis
  - 创新点：提出包含动捕、表情、语音、文本的大规模人类数据集用于数字人动作合成，以及多模态输入的基准模型和语义评估指标
  - 和我的联系：可用于验证手部动作重定向算法以及判断语义一致性

## 2022.12.14

- RT-1: Robotics Transformer for Real-World Control at Scale
  - 创新点：提出使用任务未知的训练和高容量模型实现多任务可泛化的机器人策略，Transformer模型编码语言指令、图像信息输出机器人指令
  - 和我的联系：机器人任务中可采用类似自然语言处理、计算机视觉的大规模模型

## 2022.12.7

- An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling
  - 创新点：在序列建模任务上深入对比卷积神经网络和循环神经网络，提出通用的TCN网络模型（因果卷积、膨胀卷积、残差连接）达到比一般RNN网络更好的性能
  - 和我的联系：可参考TCN结构进行时序信息提取

## 2022.11.30

- Auto-Encoding Variational Bayes
  - 创新点：提出自编码变分贝叶斯求解算法和变分自编码器结构，学习有向概率图模型的隐分布以及数据生成
  - 和我的联系：隐变量可用于状态表征，变分自编码器结构可用于数据生成

## 2022.11.29

- Accurate 3D Hand Pose Estimation for Whole-Body 3D Human Mesh Estimation
  - 创新点：提出精确手部位姿估计算法，结合掌指关节信息估计手腕旋转，去除身体信息估计手指
  - 和我的联系：能够代替Frankmocap提供手部位姿更精确的估计

## 2022.11.28

- Stochastic Scene-Aware Motion Prediction
  - 创新点：提出目标驱动的角色-场景交互的生成模型，由目标网络、动作网络、路径规划组成
  - 和我的联系：类似于机器人导航问题，不同点在于将机器人换成了人体模型；模型架构cVAE

## 2022.11.24

- Attention Is All You Need
  - 创新点：提出Transformer模型只使用自注意力机制和全连接层，不需要循环神经网络或卷积神经网络，位置编码用于获得位置信息
  - 和我的联系：可尝试使用Transformer模型，可参考注意力机制算法流程

## 2022.11.23

- Embodied hands: Modeling and capturing hands and bodies together
  - 创新点：提出手部模型MANO，结合人体模型SMPL（SMPL+H）同时建模人体身体和手部
  - 和我的联系：结合手部和身体可作为创新点，全身一体的建模能让虚拟角色的行为更真实

## 2022.11.22

- TEMOS: Generating diverse human motions from textual descriptions
  - 创新点：提出从文本到三维人类动作的生成模型，生成动作的分布而不是单一动作，基于Transformer结构和BERT语言模型，通用性强可用于SMPL模型
  - 和我的联系：SMPL实验可用于验证通用性，语言处理可使用预训练大规模语言模型

## 2022.11.21

- Camera Distance-aware Top-down Approach for 3D Multi-person Pose Estimation from a Single RGB Image
  - 创新点：提出通用框架解决从单目RGB图像中估计三维人体绝对位姿的问题，由人体检测器、根关节三维位置定位、相对根关节的位姿估计组成
  - 和我的联系：估计绝对位姿而不是相对位姿，框架通用性可兼容不同的以前的方法

## 2022.11.17

- Action-Conditioned 3D Human Motion Synthesis with Transformer VAE
  - 创新点：提出基于Transformer的变分自编码器学习人类动作的隐式表示，将Transformer和VAE结合，能够根据动作类别、时间长度生成多样变长的人类动作，隐式表示表征整个动作序列
  - 和我的联系：可尝试使用Transformer结构，隐式表示既可以表征单帧也可以表征序列

## 2022.11.16

- Model simplification using vertex-clustering
  - 创新点：提出顶点聚类算法简化三维物体模型，能实现低计算开销、高数据压缩率、高质量和平滑性
  - 和我的联系：可用于简化动画角色人物的蒙皮，加速计算过程

## 2022.11.15

- Mesh-based Dynamics with Occlusion Reasoning for Cloth Manipulation
  - 创新点：提出感知模型同时考虑衣服被遮挡的部分（测试时再优化的方法重建衣服），使用动力学模型进行机械臂规划
  - 和我的联系：测试时再优化能进一步提升效果，动力学模型可通过额外数据集训练

## 2022.11.14

- Implicit Neural Representations for Variable Length Human Motion Generation
  - 创新点：提出变分隐式神经表示用于人类动作生成，采用类似自解码器的架构对每个动作优化，将隐式表示分为基于动作和基于序列两类，能够处理变长动作序列
  - 和我的联系：隐式表示分类后可直接拼接，自解码器的架构只使用解码器

## 2022.10.26

- On Self-Contact and Human Pose
  - 创新点：提出包含自接触关系的人体位姿估计数据集和优化算法（将发生接触的顶点拉近，将穿模的顶点推离）
  - 和我的联系：自接触的定义、优化目标函数可用于动作重定向

## 2022.10.25

- On The Continuity of Rotation Representations in Neural Networks
  - 创新点：提出三维旋转的6D连续表示（格拉姆-施密特正交化后取旋转矩阵前两列），能够帮助神经网络更好地训练，欧拉角和四元数表示都不是连续表示
  - 和我的联系：关节旋转数据可采用6D连续表示

## 2022.10.24

- CoolMoves: User Motion Accentuation in Virtual Reality
  - 创新点：提出实时生成丰富多样的VR虚拟形象全身动作，利用动捕数据库KNN匹配、插值合成动作
  - 和我的联系：该算法能够实时生成全身动作，数据集规模较小，动作人物及类别有限，不一定适用于多样的人物和动作；没有考虑重定向问题

## 2022.10.23

- Contact and Human Dynamics from Monocular Video
  - 创新点：采用基于物理的轨迹优化，解决人体位姿估计结果违反物理约束（脚底穿模地面、脚底悬空、身体倾斜角度）的问题，通过预测网络估计脚底接触情况
  - 和我的联系：动作重定向任务中也需要考虑满足物理约束（脚底接触），VR估计人体位姿结果出现脚底穿模问题

## 2022.10.18

- TEACH: Temporal Action Composition for 3D Humans
  - 创新点：提出使用自然语言序列生成3D人类语义动作，采用Transformer架构生成平滑过渡动作（conditioned在上一个动作的最后几帧）
  - 和我的联系：语义通过自然语言表征，可尝试自然语言生成动画角色动作

## 2022.10.17

- AvatarPoser: Articulated Full-Body Pose Tracking from Sparse Motion Sensing
  - 创新点：提出使用稀疏传感器（头和双手）预测人体全身位姿（世界坐标系下），采用Transformer网络架构，解耦全局运动并通过逆运动学微调优化
  - 和我的联系：稀疏输入更适用于VR场景，可以尝试引入隐空间，没有考虑足底接触和形状

## 2022.10.16

- BlazePose GHUM Holistic: Real-time 3D Human Landmarks and Pose Estimation
  - 创新点：提出轻量化实时3D人体位姿估计算法，神经网络前向传播，能够以15fps实时运行
  - 和我的联系：能够实时估计人体3D位姿，坐标系以人体髋部为原点

## 2022.8.29

- Learning Kalman Network: A Deep Monocular Visual Odometry for On-Road Driving
  - 创新点：提出基于神经网络的观测模型和状态转移模型用于单目视觉里程计卡尔曼滤波
  - 和我的联系：都将卡尔曼滤波的观测模型和预测模型用网络代替，任务不同

## 2022.8.28

- AMASS: Archive of Motion Capture As Surface Shapes
  - 创新点：提出将动捕数据转化为SMPL模型的优化算法（目标函数包含标记点位置、形状一致等），将多个动捕数据集格式统一化，解决动捕数据集数据格式不同、大小多样性有限的问题
  - 和我的联系：能够用于人体数据采集，从动捕数据中恢复人体蒙皮数据

## 2022.8.23

- 3D Human Pose Estimation with Spatial and Temporal Transformers
  - 创新点：提出首个完全使用transformer模型代替卷积神经网络完成人体三维位姿估计任务，空间transformer和时间transformer分别提取时空特征
  - 和我的联系：通过时序信息解决歧义问题，transformer网络结构更有效可用于代替动力学模型

## 2022.8.22

- Expressive Body Capture: 3D Hands, Face, and Body from a Single Image
  - 创新点：提出SMPL-X模型同时建模人体身体、手部和脸部，并能够通过求解优化问题从单张RGB图像中恢复人体三维模型
  - 和我的联系：穿模损失可用于动画角色动作迁移，SMPL-X模型可用于人体位姿表示

## 2022.8.21

- End-to-end Recovery of Human Shape and Pose
  - 创新点：提出端对端框架从单张RGB图像中恢复人体三维蒙皮，训练损失使用2D投影误差和辨别器网络损失，辨别器网络用于解决缺少深度信息带来的歧义
  - 和我的联系：都是从单目数据恢复人体三维位姿，使用2D观测作为损失，不同在于输出还包括人体蒙皮信息；可参考使用辨别器网络解决歧义问题

## 2022.8.20

- Deep Kalman Filter
  - 创新点：提出使用隐状态进行卡尔曼滤波的框架，并推导似然损失的下限用于监督网络的学习
  - 和我的联系：都是隐空间进行卡尔曼滤波，偏向框架，任务不同；网络使用全连接网络和循环神经网络

## 2022.8.10

- CLIFF: Carrying Location Information in Full Frames into Human Pose and Shape Estimation
  - 创新点：结合全局位置信息和重新设计重投影误差用于解决人体3D位姿估计任务中丢失全局旋转信息的问题
  - 和我的联系：使用SMPL模型作为人体模型表示，损失函数由SMPL损失、3D损失、2D损失组成

## 2022.8.9

- Learning Object Manipulation Skills from Video via Approximate Differentiable Physics
  - 创新点：提出求解可微分的常微分方程近似地模拟物理定律（重力、摩檫力、物体接触等），用于生成物理上可行的机器人动作轨迹
  - 和我的联系：物理定律能以可微分的方式进行建模并用于优化问题

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
