# 内窥镜视频序列相机Egomotion估计 SOTA 调研报告

**调研时间：2026年8月 | 信息来源：arXiv官方API、GitHub API、Europe PMC、论文全文（WebFetch验证）**

**调研方法说明**：本报告所有方法均通过arXiv API/PMC/GitHub逐一核实，标注"未找到"处为确实未能检索到的信息。注意：通用网络搜索工具返回的链接多次出现幻觉（错误arXiv编号），故本报告仅采用经直接验证的数据。

---

## 第一部分：内窥镜/手术场景视觉里程计与SLAM SOTA

### 1.1 SCARED数据集背景

SCARED（Stereo Correspondence and Scene Reconstruction for Endoscopy，Collins et al., ICRA 2021）是内窥镜VO/SLAM最权威的基准：da Vinci Xi手术机器人对离体（ex-vivo）猪组织采集的立体内窥镜视频，含3个猪体、35个视频序列、约22,950帧。真值来源：结构光扫描的深度（仅约35个关键帧可靠）+ 光学跟踪/机器人运动学的位姿。2026年的SCARED-C工作（arXiv 2605.16628）指出：非关键帧深度真值依赖机器人运动学、误差很大，可靠标注原本仅限35个关键帧；SCARED-C用COLMAP重估全部帧位姿并以关键帧深度恢复度量尺度，将可靠RGB-D对从35对扩展到17,135对。

**重要提醒：SCARED上存在多种互不兼容的评测协议**（详见第五部分），不同论文数值不可直接横比。

### 1.2 主要方法详述

**（1）EndoSLAM / Endo-SfMLearner** | 2020 | MICCAI 2020 | arXiv 2006.16670
- **核心思想**：发布大规模内窥镜SLAM数据集（8种离体猪GI器官、3D点云、胶囊/常规内窥镜、合成胶囊内镜数据，35个子数据集带6D位姿真值），并提出Endo-SfMLearner无监督深度+位姿方法：残差网络+空间注意力模块聚焦高纹理组织区域，亮度感知光度损失应对快速光照变化。
- **SCARED性能**：不适用（在自建EndoSLAM数据集评测）。
- **开源**：是（github.com/CapsuleEndoscope/EndoSLAM）。

**（2）Endo-Depth-and-Motion** | 2021-2022 | arXiv 2103.16525 | ICRA 2022
- **核心思想**：单目内窥镜视频的6-DoF位姿+稠密3D重建管线。用自监督深度网络（Monodepth2风格）生成伪RGBD帧，光度残差跟踪相机位姿，体素表示融合配准深度图。
- **SCARED性能**：未找到（主要在Hamlyn数据集评测）。
- **开源**：是（论文声明发布全部模型与代码）。

**（3）SimCol-to-3D / SimCol3D挑战赛** | 2022-2023 | MICCAI 2022 / EndoVis sub-challenge | arXiv 2307.11261
- **核心思想**：合成数据驱动结肠镜深度与位姿估计的标杆。从3个不同人体的CT提取结肠网格，在**Unity（EndoVR渲染管线）**中渲染合成结肠镜序列，沿中心路径加随机扰动生成轨迹，同步记录RGB/深度/位姿。SimCol I、II各12训练+3测试轨迹（每条1201帧），训练集共14,412帧。挑战赛设三个子任务：合成深度、合成位姿、真实位姿。
- **关键结论**（对合成数据建库极具参考价值）：合成图像上的**深度预测"已基本可解"**（冠军FCBFormer-D的L1误差仅0.030cm），**位姿估计"仍是开放问题"**；COLMAP在Synthetic Colon I上**93%的帧重建失败**；真实位姿任务（EndoMapper数据，COLMAP伪真值）冠军MIVA（SC-SfMLearner+DenseDepth+CycleGAN真实→合成域转换）ATE平均3.59（单位dm），亚军EndoAI为7.16。
- **开源**：数据与冠军方案开源（github.com/ESandML/SimCol-Entry）。

**（4）DeDepth** | 2023 | MICCAI 2023
- **核心思想**：面向腹腔镜图像的自监督深度估计，强调域泛化（尺度-平移感知的深度一致性+校正网络）。
- **注意**：arXiv API与Europe PMC均未检索到该论文记录，**其准确标题、作者与SCARED数值未找到**（仅通用搜索工具给出方向性描述，可靠性存疑）。谨慎引用。

**（5）SSL-SLAM** | 未找到
- arXiv、Europe PMC、GitHub多渠道检索均**未找到名为"SSL-SLAM"的确切论文**。可能是某具体工作的别名或笔误，建议核对原始出处。

**（6）EndoDAC** | 2024 | MICCAI 2024 | arXiv 2405.08672
- **核心思想**：高效自监督内窥镜深度估计框架。以基础模型（Depth Anything）为骨干，用DV-LoRA（动态向量低秩适配）+卷积neck做参数高效域适配（可训练仅1.6M参数）；无内参时用位姿编码器自监督估计相机内参。仅需单目手术视频即可训练。
- **SCARED性能**（遵循AF-SfMLearner划分15351/1705/551帧；5帧位姿协议，两条轨迹ATE）：给内参0.0741/0.0512，无内参0.0762/0.0487；深度AbsRel 0.051、δ 0.981。对比：Monodepth2 0.0769/0.0554，AF-SfMLearner 0.0742/0.0478。
- **开源**：是（github.com/BeileiCui/EndoDAC）。

**（7）DyEndoVO** | 2025 | IJCARS（Int J CARS, vol.21, pp.723-733, DOI 10.1007/s11548-025-03549-0）| PMC13194247全文
- **核心思想**：动态手术场景的位姿估计。两大模块：(a) 运动检测模块——基于OCLR改造，RAFT计算前向/后向光流+立体流，CNN U-Net+transformer瓶颈，用**单一motion query**替代对象-查询一一对应设计（因变形组织"模糊且不可数"），输出逐像素运动概率图M；(b) 可微加权位姿优化——用(1−M)对重投影残差加权降低动态区域权重，端到端仅用位姿真值训练。
- **数据贡献**：**DynaSCARED半合成数据集**——SCARED真实背景 + 器械贴片（单应变换模拟刚性运动，取自Robotic Instrument Segmentation）+ 组织贴片（薄板样条模拟非刚性变形，取自CholecSeg8k），8种场景类别、带运动掩码与轨迹真值；训练/验证4000/500序列×30帧。
- **性能**：**未在SCARED原始数据上报告ATE**（SCARED仅作背景）。StereoMIS真实数据：移动相机Bowel片段RPE 0.152mm / ATE 1.444mm，优于Hayoz et al.（1.581）、EDaM（1.405为ATE最优但RPE并列）、DroidSLAM（1.185 ATE最优但RPE差）；**仅用合成DynaSCARED训练的模型优于用真实StereoMIS训练的模型**（Liver ATE 3.268 vs 3.364）——合成数据有效性的有力证据。
- **开源**：声称开源（github.com/giiinger98/DyEndoVO），**但仓库当前为空**。

**（8）Endo-FASt3r** | 2025 | arXiv 2503.07204（v4 2025-08更新）
- **核心思想**：首个将基础模型同时用于单目SSL深度与位姿估计的内窥镜框架。扩展Reloc3r为Reloc3rX（改造以在SSL中收敛），提出DoMoRA适配技术（高秩更新、更快收敛）。
- **SCARED性能**（5帧位姿协议）：**ATE-T1 0.0702 / T2 0.0438，当前该协议最优**（比次优DARES/Yang et al.提升6.6%/12.0%）；深度AbsRel 0.051、δ 0.998。完整对比表：DeFeat-Net 0.1765/0.0995，SC-SfMLearner 0.0767/0.0509，Monodepth2 0.0769/0.0554，Endo-SfM 0.0759/0.0500，AF-SfMLearner 0.0757/0.0501，Yang et al. 0.0723/0.0474，DARES 0.0752/0.0498，Zero-shot Reloc3r 0.0938/0.0735。
- **开源**：未在摘要中确认。

**（9）Endo3R** | 2025 | arXiv 2504.03198（目标MICCAI 2025方向）
- **核心思想**：统一3D基础模型，从单目手术视频在线做尺度一致重建，无需标定/器械先验/离线优化。预测全局对齐pointmap+尺度一致视频深度+相机参数；核心是**不确定性感知双记忆机制**（短期动态token+长期空间一致性token），将成对重建模型扩展为长期增量式动态重建。
- **SCARED性能**（全轨迹协议）：**ATE 0.112**（vs EndoDAC 0.124、AF-SfM 0.125、Robust 0.131、Endo-SfM 0.157）；深度AbsRel 0.124 / RMSE 1.209 / δ 0.839（vs MonST3R 0.198/1.965/0.626）；19.17 FPS。
- **开源**：项目页cut3r风格，未确认具体链接。

**（10）EndoGS** | 2024 | arXiv 2401.11535
- **核心思想**：高斯泼溅重建可变形内窥镜组织：变形场处理动态场景、深度引导监督+时空权重掩码处理单视角器械遮挡、表面对齐正则。输入：单视角视频+估计深度+器械掩码标注。
- **SCARED性能**：主要在EndoNeRF（da Vinci手术视频）评测渲染质量，SCARED数值未找到。
- **开源**：是（github.com/HKU-MedAI/EndoGS）。

**（11）LumenGSLAM** | 2025 | GitHub项目（论文venue未找到）
- **核心思想**：单目内窥镜在线**物理渲染（PBR）+3D高斯泼溅**重建与跟踪：分离base color/diffuse light等光照成分实现重光照，SuperPoint+LightGlue高斯耦合关键点跟踪应对快速运动与光照变化；在C3VD和SCARED上评测。
- **SCARED性能**：README未给出数值（占位符状态），**未找到**。
- **开源**：是（github.com/FrancescoLeni/LumenGSLAM，代码完整）。

**（12）SurgCUT3R** | 2026 | arXiv 2603.06971（项目页暗示ICRA 2026）
- **核心思想**：将CUT3R统一3D重建模型系统性适配到手术域。三贡献：(a) **数据生成管线**——用FoundationStereo从公开立体内窥镜数据（SCARED/StereoMIS，剔除标定有误的Dataset 4/5）生成大规模度量尺度伪GT深度图；(b) 伪GT+几何自校正混合监督；(c) **层次化双模型推理**（全局稳定模型+局部精确模型）抑制长视频位姿漂移。
- **SCARED性能**（ATE/RTE，mm单位）：**SurgCUT3R 5.514/0.752（前馈类最优）**；对比：EndoDAC 10.225/0.963，Spann3R 10.258/1.260，AF-SfMLearner 10.312/0.971，CUT3R裸用9.361，MonST3R w/Opt 21.774/1.582，优化类方法MegaSaM 2.002/0.315（全场最优但仅0.7FPS）；深度AbsRel 0.057；19.7 FPS。StereoMIS零样本：ATE 25.939mm（MegaSaM 19.705最优）。
- **开源**：未确认。

**（13）EndoVGGT** | 2026 | arXiv 2603.24577
- **核心思想**：VGGT的内窥镜变体。DeGAT（变形感知图注意力）在特征空间动态构建语义图、跨遮挡传播结构线索，处理低纹理、镜面高光、器械遮挡导致的几何连续性断裂。
- **SCARED性能**：PSNR较先前SOTA提升24.6%、SSIM提升9.1%（重建质量指标，非ATE）；零样本跨域到SCARED与EndoNeRF。
- **开源**：未确认。

**（14）EndoSfM3D** | 2025 | arXiv 2510.22359
- **核心思想**：自监督基础模型（Depth Anything V2）+DoRA高效微调，联合预测深度、位姿与**内参**（应对连续变焦/镜筒旋转等真实内窥镜内参不可标定问题），注意力位姿网络。
- **SCARED性能**：深度估计优于近期SOTA（具体表格未在摘要层展开）。
- **开源**：未确认。

**（15）其他2024-2026值得关注的内窥镜方法**：
- **BodySLAM**（arXiv 2408.03078, 2024）：CycleVO无监督位姿+ZoeDepth深度+3D重建；在Hamlyn/EndoSLAM/SCARED三数据集评测但**仅以箱线图呈现，无具体数值**；CycleVO推理最快（SCARED 38.57s vs OneSLAM 336.65s）。
- **CudaSIFT-SLAM**（arXiv 2405.16932, 2024）：SIFT+CudaSIFT暴力匹配替代ORB/DBoW2的**首个能实时处理完整人体结肠镜的多地图V-SLAM**；C3VD与真实全流程结肠镜评测。
- **EndoFlow-SLAM**（arXiv 2506.21420, 2025）：3DGS内窥镜SLAM加**光流损失作为几何约束**（应对非朗伯面导致的光度不一致）+深度正则。
- **SAGS**（arXiv 2510.27318, 2025）：自自适应抗混叠4D高斯泼溅（3D平滑滤波+2D Mip滤波+注意力动态权重变形解码器），EndoNeRF与SCARED上渲染指标领先。
- **Diff2DGS**（arXiv 2602.18314, 2026）：扩散视频模型时序先验修复器械遮挡区域+2DGS可学习变形模型；在SCARED上做深度定量（38.02dB PSNR on EndoNeRF）。
- **Local-EndoGS**（arXiv 2602.17473, 2026）：任意大相机运动下的单目4D手术重建：渐进式窗口化全局表示+局部可变形场景模型+粗到细初始化（多视图几何+跨窗口信息+单目深度先验），摆脱对立体深度/SfM初始化的依赖。
- **Bridging Ex-Vivo to In-Vivo Gap**（arXiv 2512.23786, 2025）：指出公开数据集（离体）与真实手术（在体，严重镜面反射+积液变形）间的"ex-vivo到in-vivo鸿沟"；用DA2几何先验+DV-LoRA，SCARED上Sq Rel高镜面区域降低17%+；发布**ROCAL-T 90**（真实术中CT对齐腹腔镜轨迹90条）。
- **NFL-Depth/PPSNet**（arXiv 2403.17915, 2024）：利用**近场光照**（内窥镜自带光源的逐像素明暗着色表示）做单目深度，C3VD上SOTA，代码开源（ppsnet.github.io）。

### 1.3 SCARED上的排名小结（按协议分组）

**协议A：5帧位姿估计，两条轨迹ATE**（EndoDAC/Endo-FASt3r系，归一化单位）：
Endo-FASt3r (0.0702/0.0438) > Yang et al. (0.0723/0.0474) > EndoDAC/DARES ≈ AF-SfMLearner ≈ Endo-SfM ≈ SC-SfMLearner ≈ Monodepth2 (0.075-0.077区间) >> DeFeat-Net (0.1765/0.0995)

**协议B：全序列在线重建ATE**（Endo3R协议，归一化）：
Endo3R (0.112) > EndoDAC (0.124) > AF-SfM (0.125) > Robust (0.131) > Endo-SfM (0.157)

**协议C：全序列ATE，毫米单位**（SurgCUT3R协议，2026最新）：
- 优化类：MegaSaM (2.002mm)
- 前馈类：**SurgCUT3R (5.514mm)** > CUT3R裸用 (9.361) > EndoDAC (10.225) > Spann3R (10.258) > AF-SfMLearner (10.312) > MonST3R w/Opt (21.774)

---

## 第二部分：通用相机位姿估计/egomotion SOTA（2024-2026）

### 2.1 前馈式3D重建与位姿（DUSt3R谱系）

**（1）DUSt3R** | 2023.12 | CVPR 2024 | arXiv 2312.14132
- **核心思想**：将无约束双视图3D重建表述为**pointmap回归**（每像素直接回归3D点），摆脱射影相机模型硬约束，统一单目/双目情形；多视图时用全局对齐策略把成对pointmap表达进公共坐标系。无需标定与位姿先验。开源（naver/dust3r）。

**（2）MASt3R** | 2024.06 | ECCV 2024 | arXiv 2406.09756
- **核心思想**：把图像匹配从2D问题提升为3D任务：在DUSt3R上加稠密局部特征头+匹配损失，配合快速互惠匹配方案解决稠密匹配的二次复杂度。极端视角变化下鲁棒且精确。开源。

**（3）MASt3R-SfM** | 2024.09 | 3DV 2025 | arXiv 2409.19152
- **核心思想**：基于MASt3R的完整SfM管线：低内存方法把局部重建对齐到全局坐标系；用基础模型自身做图像检索（免额外开销），复杂度从二次降为线性；可处理任意有序/无序图像集合。

**（4）Spann3R** | 2024.08 | 3DV 2025 | arXiv 2408.16061
- **核心思想**：外部**空间记忆**机制——transformer直接在全局坐标系回归逐图像pointmap（DUSt3R是每对图像的局部坐标系），免优化全局对齐；查询空间记忆预测下一帧全局3D结构，可实时处理有序图像集。

**（5）Fast3R** | 2025.01 | CVPR 2025 | arXiv 2501.13928
- **核心思想**：DUSt3R的多视图并行化推广：transformer单次前向处理N张（1000+）图像，跳过迭代对齐；位姿估计与3D重建上SOTA，显著降低误差累积。

**（6）MonST3R** | 2024.10 | ICLR 2025 | arXiv 2410.03825
- **核心思想**：动态场景几何优先：为**每个时间步**估计pointmap（把DUSt3R的静态场景表示适配到动态场景），通过在有限的"动态、带位姿、带深度标签视频"数据上微调即可处理动态（无需显式运动表示）。**关键论断：动态带位姿带深度标注视频的稀缺是核心瓶颈**——这正是仿真数据的用武之地。开源。

**（7）CUT3R** | 2025.01 | CVPR 2025 | arXiv 2501.12387
- **核心思想**：有状态循环模型，随每帧新观测持续更新状态表示，在线生成**度量尺度**pointmap且共处同一坐标系，可累积为随输入更新的稠密重建；还能通过"虚拟未观测视图探测"推断未见区域；天然支持变长输入（视频流或无序照片、静态或动态内容）。项目页cut3r.github.io。

**（8a）π³ (Pi3)** | 2025.07 | ICLR 2026 | arXiv 2507.13347
- **核心思想**：完全**排列等变**的前馈几何网络，取消参照视图归纳偏置（VGGT/DUSt3R 锚定首帧时参照选差会崩）。输入无序图像集/视频，预测仿射不变位姿 + 尺度不变局部 pointmap。Sintel 相机 ATE：VGGT 0.167 → **π³ 0.074**；推理 57.4 FPS。本机暂无权重（CC BY-NC），评测接口可后续挂接。开源（github.com/yyfz/Pi3）。

**（8）VGGT** | 2025.03 | CVPR 2025 | arXiv 2503.11651
- **核心思想**：前馈网络一次性从1到数百视图**直接推断全部3D属性**：相机参数（内参+外参）、pointmap、深度图、3D点跟踪。亚秒级完成重建，无需后处理视觉几何优化即在相机参数估计、多视图深度、稠密点云、3D点跟踪多任务SOTA；预训练权重可作下游骨干（非刚性点跟踪、前馈新视图合成）。**开源**（github.com/facebookresearch/vggt）。其2026年衍生：VGGT-SLAM++、VGGT-Long（公里级长序列）、VGGT-World、VGGT-X等，以及内窥镜域的EndoVGGT与SurgCUT3R。

**（9）Easi3R** | 2025.03 | arXiv 2503.24391
- **核心思想**：**免训练**从DUSt3R解耦运动（动态物体掩码），通过注意力图分析，适合快速适配动态场景。

**（10）未找到的方法**：**Fluid3R、MAPRT、REFrame（VO语境）** 在arXiv API中均未检索到（截至今日）。REFrame仅有CVPR 2025动态相机去模糊论文（arXiv 2504.07817）与位姿估计无关；如用户所指另有其文，请提供线索。

### 2.2 深度学习VO/SLAM经典与近期

**（11）DROID-SLAM** | 2021.08 | NeurIPS 2021 | arXiv 2108.10869
- **核心思想**：循环迭代更新相机位姿与逐像素深度，核心是**稠密Bundle Adjustment层**（构造帧间运动与光度残差的线性系统，Net取代LM迭代）；仅用单目视频训练，测试时可利用立体/RGB-D。精度高、灾难性失败少。开源。**SCARED上数值：未找到官方数字**。

**（12）DPVO** | 2022-2023 | CVPR 2023 | arXiv 2208.04726
- **核心思想**：稀疏patch跟踪替代稠密光流：面向patch对应的循环更新算子+可微束调整；内存仅DROID的1/3、平均快3倍。开源。arXiv v2版无SCARED实验（最终期刊版可能含，未验证）。

**（13）DPV-SLAM** | 2024.08 | arXiv 2408.01654
- **核心思想**：DPVO扩展为完整SLAM：邻近回环（单一共享patch图中混合里程计与回环因子，单向边省显存）+经典回环（dBoW2检索+RANSAC+**Umeyama**点云配准估计**Sim(3)**漂移）；单GPU 50FPS、5G显存（DROID需20G）。

**（14）TartanVO** | 2020-2021 | ICRA 2021 | arXiv 2011.00359
- **核心思想**：首个可跨数据集泛化的学习式VO：**仅用TartanAir合成数据训练**，凭借up-to-scale损失+内参输入设计，零微调泛化到KITTI/EuRoC真实数据并超越几何方法——合成数据训练VO的里程碑工作。开源。

**（15）DeepV2D** | 2018-2020 | CVPR 2020 | arXiv 1812.04605
- **核心思想**：可微SfM：运动估计与深度估计交替优化、端到端可微，视频到深度的经典框架。

**（16）DF-VO** | 2020 | AAAI 2020
- **核心思想**：深度-流先验融合的VO（单目深度网络+光流网络结合几何一致性）。arXiv确切编号未能验证，**SCARED数值未找到**。

### 2.3 3D高斯泼溅SLAM

**（17）MonoGS（Gaussian Splatting SLAM）** | 2023.12 | CVPR 2024 | arXiv 2312.06741
- **核心思想**：首个单目3DGS-SLAM：高斯作为唯一3D表示，直接对3D高斯做相机跟踪优化（宽收敛域），几何验证与正则化处理增量稠密重建歧义，3fps在线运行。开源。SCARED数值未找到（评测于Replica/TUM/7-Scenes/AteC）。

**（18）SplaTAM** | 2023.12 | CVPR 2024 | arXiv 2312.02126
- **核心思想**：RGB-D 3DGS-SLAM：silhouette掩码建模场景密度存在性，在线跟踪-建图，位姿估计较先前方法最高2倍提升。开源。

**（19）GS-SLAM** | 2023.11 | CVPR 2024 | arXiv 2311.11700
- **核心思想**：首个将3DGS引入SLAM系统：自适应扩展策略（增删高斯）+粗到细位姿优化（选择可靠高斯），平衡效率与精度。开源。SCARED数值未找到（评测Replica/TUM）。

### 2.4 视频位姿大模型与其他

**（20）NeRF-VO** | 2023.12 | RA-L 2024 | arXiv 2312.13471
- **核心思想**：单目VO=学习式稀疏VO（低延迟跟踪）+NeRF场景表示（细粒度稠密重建与NVS）：稀疏VO初始化位姿、单目预测网络给稠密几何先验、统一尺度后作监督信号训练神经隐式表示，滑窗联合优化关键帧位姿与稠密几何。

**（21）FeatUp** | 2024.03 | CVPR 2024 | arXiv 2403.10516
- **核心思想**：模型无关框架，把任意视觉骨干特征上采样到任意高分辨率——为稠密对应/位姿估计提供高质量特征。

**（22）SceneScript** | 2024.03 | CVPR 2024 | arXiv 2403.13064
- **核心思想**：自回归结构化语言模型重建场景（用结构化语言描述几何），2025年12月出现加速版Fast SceneScript（多token预测）。

**（23）MegaSaM** | CVPR 2025 |（SurgCUT3R论文中引用为对比方法）
- **核心思想**：动态场景的运动估计与深度的概率优化系统（Mega-深度+位姿联合优化）。**SCARED上ATE 2.002mm为全场最优**（但属优化类、0.7FPS非前馈）。

---

## 第三部分：医疗场景特殊挑战与应对

| 挑战 | 代表性应对方法 |
|---|---|
| **组织非刚性变形**（心跳/呼吸/牵拉） | DyEndoVO（运动概率图加权位姿残差）；EndoGS/SAGS/Diff2DGS/Local-EndoGS（变形场+4DGS）；MonST3R/CUT3R（逐时间步/持续状态pointmap，天然支持动态）；Endo3R（不确定性感知双记忆：短期动态+长期空间一致token） |
| **湿润镜面高光/非朗伯表面** | EndoFlow-SLAM（光流几何约束替代纯光度约束）；LRED（镜面感知朗伯重建）；Bridging Ex-Vivo to In-Vivo（物理分层评测协议，高镜面区域Sq Rel降低17%+）；LumenGSLAM（PBR显式光照建模分离镜面成分） |
| **缺乏纹理** | EndoSLAM（空间注意力聚焦高纹理区）；SurgCUT3R（几何自校正）；EndoVGGT（特征空间语义图跨遮挡传播结构线索） |
| **光照变化/近场光源** | EndoSLAM（亮度感知光度损失）；NFL-Depth（显式利用内窥镜近场光照的shading先验）；LumenGSLAM（PBR重光照） |
| **器械存在与遮挡** | DyEndoVO（器械贴片运动建模）；Diff2DGS（扩散模型修复遮挡组织）；EndoGS（器械掩码加权） |
| **烟雾/雾气** | 主要在SimCol3D与CudaSIFT-SLAM的失败模式分析中被列为跟踪丢失原因（遮挡、模糊、水花、工具交互），专门方法较少 |
| **近距离大基线/尺度模糊** | EndoSfM3D与EndoDAC（联合估计内参/位姿）；Endo3R与CUT3R（度量尺度pointmap）；SCARED-C（COLMAP+关键帧深度恢复度量尺度） |
| **域偏移（离体→在体、器官间、相机间）** | EndoDAC（DV-LoRA高效适配，无内参可用位姿编码器自估）；SimCol3D任务3冠军（CycleGAN真实→合成域转换）；ColonCrafter（扩散先验+保几何风格迁移）；DyEndoVO实验证明合成训练泛化超真实训练 |

---

## 第四部分：仿真/合成数据在egomotion训练中的作用

**（1）TartanAir**（IROS 2020，arXiv 2003.14338）：光真仿真环境（Unreal）大规模SLAM数据：立体RGB、深度、分割、光流、位姿、LiDAR全模态真值；自动管线（建图、轨迹采样、轨迹处理、验证）。**直接催生TartanVO——证明纯合成训练可零样本泛化到真实VO**。DUSt3R系模型训练数据的重要组成部分。

**（2）SimCol-to-3D / EndoVR（Unity）**：CT提取结肠网格+Unity渲染+中心路径随机扰动轨迹。核心发现：合成→深度任务基本解决（L1 0.030cm），合成→位姿仍开放；合成→真实位姿需域转换（CycleGAN）辅助。

**（3）C3VD**（Medical Image Analysis 2023, Bobrow et al.）：半合成范式标杆——高清临床结肠镜拍摄高保真物理结肠模型（真实光学/光照），通过2D-3D注册获得逐帧深度/法线/光流/位姿真值与OBJ网格；提供数据加载器与点云重投影脚本。规避纯渲染的纹理域差。

**（4）C3VDv2 / C3VD-DEFCOL**（arXiv 2606.07891, 2026）：DEFCOL框架在C3VD网格+轨迹上施加**参数化变形**（蠕动波、中心线运动），逐帧渲染深度/法线/光流/位姿/时戳3D网格，再用**LTX-2.3视频扩散模型做sim-to-real翻译**（保几何、换真实黏膜外观）；110条视频×11种结肠几何。首个"真实外观+稠密时变3D真值+非刚性变形"数据集——直接回应了MonST3R指出的"动态带真值数据稀缺"痛点。

**（5）DynaSCARED**（DyEndoVO, IJCARS 2025）：贴片合成法——真实SCARED背景+器械/组织贴片几何变换（单应/薄板样条），带运动掩码；**训练于合成者优于训练于真实者**。

**（6）EndoSLAM合成部分**：GI-tract合成胶囊内镜帧（带深度+位姿标注），明确用于sim-to-real迁移研究。

**（7）EndoMapper**（arXiv 2204.14240, MICCAI 2022）：24+小时完整真实内窥镜流程（含标定视频/标定、模拟序列带真值、同患者多流程）——真实数据的补充。

**（8）SurgCUT3R的伪GT管线**（2026）：FoundationStereo从真实立体数据反推度量深度作训练信号——介于真实与合成之间的第三条路线。

**仿真平台小结**：Unity（SimCol-to-3D/EndoVR）、Unreal（TartanAir）、Blender类（C3VD渲染真值）、视频扩散sim-to-real翻译（C3VD-DEFCOL）为主流四条技术路线。用户提到的"MIScnn"实为医学图像分割CNN框架而非仿真器；"VR-Renderers"未找到独立平台（SimCol-to-3D论文中的EndoVR即其VR渲染器）。

---

## 第五部分：常用评测协议

### 5.1 SCARED的评测划分
- **常用划分**（AF-SfMLearner提出，EndoDAC等沿用）：22,950帧中15351/1705/551作训练/验证/测试。
- **5帧位姿协议**（EndoDAC、Endo-FASt3r）：固定两条轨迹报ATE-T1/T2。
- **全序列协议**（Endo3R、SurgCUT3R）：全部测试序列在线推理报平均ATE。
- **剔除规则**：SurgCUT3R剔除标定有误的Dataset 4/5；SCARED-C指出非关键帧深度真值不可靠（用COLMAP重估修正）。
- **DROID-SLAM式协议**：SCARED常用关键帧+序列子集评测（多见于未公开评测设置，具体数字本报告未找到权威表格）。

### 5.2 指标定义
- **ATE RMSE**（绝对轨迹误差）：对齐后预测轨迹与真值轨迹的帧间欧氏距离RMSE。单目方法尺度不确定→需**Sim(3)对齐（7-DoF：旋转3+平移3+尺度1）**，即**Umeyama/Kabsch对齐+尺度**；立体/RGB-D/有度量尺度输出（CUT3R系）时可用**6-DoF（SE(3)）对齐**。DPV-SLAM回环即用Umeyama估计Sim(3)漂移。
- **RPE**（相对位姿误差）：固定时间间隔的相对位姿差（平移mm + 旋转deg）；DyEndoVO报告RPE mm/RPE deg/ATE mm三元组；SimCol3D用RTE（相对平移误差）+ROT（旋转误差）+ATE。
- **深度指标**：Abs Rel、Sq Rel、RMSE、RMSE log、δ<1.25阈值比例；SimCol3D深度按轨迹尺度对齐后评L1/RMSE/Rel中位数。

### 5.3 对齐细节实践
- 单目自监督方法（Monodepth2系）：Sim3对齐+常做中值尺度缩放；
- 前馈基础模型（DUSt3R系pointmap）：全局对齐时同样采用Umeyama；CUT3R/Endo3R输出本征度量尺度pointmap，理论上可直接6-DoF对齐（SurgCUT3R的毫米级评测即在此设定下）；
- **不同论文协议差异是SCARED数值混乱的主因**：0.07级（5帧协议）、0.112级（全轨迹归一化）、5.514mm级（全轨迹毫米）三类数值不可互相换算比较。

---

## 第六部分：总结表格

| 方法 | 年份/Venue | 类别 | SCARED关键指标 | 开源 |
|---|---|---|---|---|
| EndoSLAM/Endo-SfMLearner | 2020 MICCAI | 内窥镜自监督VO | 不适用（自建数据集） | 是 |
| Endo-Depth-and-Motion | 2022 ICRA | 内窥镜VO+重建 | 未找到（Hamlyn评测） | 是 |
| SimCol-to-3D/EndoVR | 2022 MICCAI | 合成数据+挑战赛 | 不适用（合成结肠） | 是 |
| DeDepth | 2023 MICCAI(?) | 自监督深度 | 未找到（记录未验证） | 未确认 |
| EndoDAC | 2024 MICCAI | 基础模型域适配 | ATE 0.0741/0.0512；AbsRel 0.051 | 是 |
| DyEndoVO | 2025 IJCARS | 动态场景VO | 未在SCARED评测；StereoMIS ATE 1.444mm | 声称（仓库空） |
| Endo-FASt3r | 2025 arXiv | 基础模型位姿适配 | **ATE 0.0702/0.0438（5帧协议最优）** | 未确认 |
| Endo3R | 2025 arXiv | 前馈统一重建 | ATE 0.112（全轨迹协议最优级） | 未确认 |
| EndoGS | 2024 arXiv | 可变形3DGS重建 | 渲染指标为主 | 是 |
| LumenGSLAM | 2025 | PBR+3DGS SLAM | 未找到 | 是 |
| SurgCUT3R | 2026 ICRA(?) | CUT3R手术域适配 | **ATE 5.514mm（前馈最优）** | 未确认 |
| EndoVGGT | 2026 arXiv | VGGT内窥镜变体 | PSNR +24.6% | 未确认 |
| SCARED-C | 2026 arXiv | 数据集修正 | 可靠RGB-D 35→17,135对 | 是 |
| DUSt3R | 2024 CVPR | pointmap回归 | —（通用） | 是 |
| MASt3R/MASt3R-SfM | 2024/2025 | 3D匹配/SfM | —（通用） | 是 |
| MonST3R | 2025 ICLR | 动态场景几何 | SCARED深度AbsRel 0.198（Endo3R引） | 是 |
| CUT3R | 2025 CVPR | 持续状态模型 | SCARED ATE 9.361mm（SurgCUT3R引） | 是 |
| VGGT | 2025 CVPR | 前馈全属性预测 | —（通用SOTA） | 是 |
| π³ (Pi3) | 2026 ICLR | 排列等变、无参照视图 | 通用 Sintel ATE 0.074（优于 VGGT 0.167） | 是（权重 NC） |
| Spann3R | 2025 3DV | 空间记忆 | SCARED ATE 10.258mm（SurgCUT3R引） | 是 |
| Fast3R | 2025 CVPR | 多视图并行 | —（通用） | 是 |
| DROID-SLAM | 2021 NeurIPS | 稠密BA | 未找到官方数字 | 是 |
| DPVO/DPV-SLAM | 2023/2024 | 稀疏patch VO | arXiv版无SCARED实验 | 是 |
| TartanVO | 2021 ICRA | 合成训练VO | 不适用 | 是 |
| MonoGS/SplaTAM/GS-SLAM | 2024 CVPR | 3DGS-SLAM | 未找到（通用基准） | 是 |
| NeRF-VO | 2024 RA-L | VO+NeRF | 未找到 | 是 |
| MegaSaM | 2025 CVPR | 优化类 | **ATE 2.002mm（全场最优）** | 未确认 |

---

## 第七部分：对仿真数据集构建的启示

基于以上调研，若要构建面向内窥镜egomotion训练的仿真数据集，SOTA方法的实际需求可归纳如下：

**1. 必需模态（按方法谱系）**
- **DUSt3R/MASt3R/Reloc3r谱系**（含Endo-FASt3r、Endo3R、SurgCUT3R）：图像序列/图像对 + **6-DoF相对位姿（4×4矩阵或q+t）** + 内参。Reloc3r以位姿为监督（这就是Endo-FASt3r能直接用的原因）。
- **MonST3R/CUT3R/VGGT谱系**：视频流 + 位姿 + **稠密逐帧深度图（pointmap可由位姿+深度+内参合成）**；MonST3R明确指出"动态、带位姿、带深度标签的视频"稀缺是其最大瓶颈——仿真数据正好零成本提供。
- **自监督方法**（EndoDAC/EndoSfM3D）：仅需单目视频+（可选）内参——仿真数据即使只给RGB也能用，但给深度真值可做监督混合（SurgCUT3R证明伪GT+几何自校正混合监督优于纯自监督）。
- **经典VO**（TartanVO/DROID/DPVO）：视频+位姿GT（DROID还需流/深度伪标签可选）。

**2. 强烈建议提供的附加真值**（已有工作证明价值）
- **逐帧稠密深度（度量尺度）**：CUT3R/Endo3R输出度量pointmap是当前方向，SCARED-C专门修正尺度问题；SurgCUT3R用FoundationStereo造度量深度伪GT。
- **光流/场景流**：EndoFlow-SLAM（光流约束）、DyEndoVO（RAFT流驱动运动检测）均依赖；C3VD-DEFCOL逐帧渲染光流。
- **运动/变形掩码**：DyEndoVO训练数据带运动掩码（虽然其网络可端到端仅用位姿训练，掩码可用于评测分层）。
- **器械/组织分割**：EndoGS需器械掩码输入。
- **非刚性变形参数化真值**：C3VD-DEFCOL开创——逐帧时戳3D网格+参数化变形（蠕动波、中心线运动），直接评测非刚性场景。
- **法线**：C3VD系提供。

**3. 外观真实性策略（sim-to-real gap的教训）**
- 纯渲染纹理在位姿任务上有明显域差（SimCol3D结论：合成深度可解、位姿难）；三条已验证的缓解路线：
  (a) **物理模型+真实相机拍摄**（C3VD：真实光学但轨迹受物理限制）；
  (b) **真实背景+合成动态贴片**（DynaSCARED：零渲染成本解决动态真值）；
  (c) **渲染几何+扩散模型sim-to-real纹理翻译**（C3VD-DEFCOL的LTX-2.3路线：保几何换外观，当前最前沿）；
  (d) 域随机化+测试时CycleGAN/风格迁移（SimCol3D任务3、ColonCrafter）。
- 仿真中应显式建模**内窥镜特有光学**：近场点光源（NFL-Depth证明shading是有效先验）、湿润镜面高光、烟雾、渐晕、非朗伯湿润组织反射。

**4. 轨迹与场景多样性设计**
- SimCol-to-3D用中心路径+随机扰动；TartanAir强调"物理平台难以实现的多样视角与运动模式"——内窥镜仿真应覆盖：近距离大基线、快速退镜/进镜、旋转镜身、回退重访（回环）、静止观察（SCARED的静止相机场景）、弱纹理长直肠段。
- 器械交互（牵拉组织、器械进出视野）与组织变形叠加（DynaSCARED的8类场景分类法可直接借鉴：动/静相机×器械运动/组织变形/静态组合）。
- DyEndoVO的发现——**合成训练≥真实训练**——说明在动态真值无法真实获取的前提下，合成动态数据不是妥协而是最优解。

**5. 评测协议配套建议**
- 数据集应同时提供：逐帧位姿（非仅关键帧，SCARED的教训）、每序列内参、深度（16bit PNG度量）、变形掩码/网格序列；
- 支持三种对齐评测：SE(3)（6-DoF，给度量尺度方法）、Sim(3)（7-DoF Umeyama，给单目方法）、以及关键帧级与全序列两种粒度；
- 提供与SCARED/C3VD/StereoMIS相同的目录结构约定（C3VD的"time + 列优先齐次位姿"逐行pose.txt格式已是事实标准之一），便于社区直接接入。

**核心结论**：2025-2026的趋势是"**前馈基础模型（VGGT/CUT3R谱系）+ 手术域适配（伪GT/LoRA/双记忆）**"取代传统自监督VO成为SCARED榜新王（SurgCUT3R 5.514mm vs EndoDAC 10.225mm），而这些基础模型全部是**数据饥渴型**——对带位姿+深度+（动态）真值的大规模视频的需求比以往任何时候都大，这正是高质量内窥镜仿真数据集的历史性机会窗口。
