# 方法

## 1. 技术基础

给定当前观测 $O_t$、本体状态 $q_t$ 和任务指令 $\ell$，世界动作模型联合预测未来视频 $V$、几何运动 $T$ 和长度为 $H$ 的动作序列 $A$。Metis4D 以本体—物体交互轨迹 Track4D 作为几何运动表示，学习条件分布 $p_\theta(V,T,A\mid O_t,q_t,\ell)$。三种模态覆盖同一物理预测窗口 $[t,t+H]$，但允许使用不同的采样频率与 token 数量。

**Flow matching.** Video 和 Track 在 VAE 潜空间生成，Action 在归一化动作空间生成。对任一模态的干净样本 $x^m$，构造线性加噪路径：

$$
x^m_{\sigma_m}=(1-\sigma_m)x^m+\sigma_m\epsilon^m,
\qquad \epsilon^m\sim\mathcal N(0,I),
$$

其中 $m\in\{V,T,A\}$，$\sigma_m=0$ 表示干净数据，$\sigma_m=1$ 表示纯噪声。网络回归速度场 $v_\theta^m$，监督目标为 $\epsilon^m-x^m$；推理时从 $\sigma_m=1$ 积分至 $0$。这一参数化沿用 [Flow Matching](https://arxiv.org/abs/2210.02747)。下文以 $\sigma$ 表示噪声水平，以 $t$ 表示物理时间，二者相互独立。

**Mixture of Transformers.** 三种模态分别使用独立的 Transformer 专家，各自保留输入映射、时间嵌入、注意力投影、前馈网络和输出头。专家将 Q、K、V 投影到兼容的注意力空间，通过跨模态注意力交换信息。视频先验、几何动力学和机器人控制因此可以使用不同的参数规模与去噪时钟，而不必共享同一套 Transformer 权重。

## 2. 以 Track4D 为核心的三专家世界动作模型

### 2.1 本体—物体交互的 Track4D 表示

Track4D 描述机械臂、夹爪及交互物体的三维运动。它在同一相机坐标系与物理时间轴下同时包含动作直接驱动的本体运动和物体对交互的响应。

**逐帧位移场。** 记时刻 $t$ 的前景像素集合为 $\Omega_t=\Omega_t^{\mathrm{body}}\cup\Omega_t^{\mathrm{object}}$。对 $u\in\Omega_t$，记其对应表面点在时刻 $t$ 相机坐标系 $C_t$ 下的位置为 $P_t(u)$，同一表面点在 $t+\Delta$ 的位置变换到 $C_t$ 后记为 $P^{C_t}_{t+\Delta}(u)$，定义

$$
D_t(u)=P^{C_t}_{t+\Delta}(u)-P_t(u),
\qquad u\in\Omega_t .
$$

$\Delta$ 是预处理固定的采样间隔，$D_t$ 表示该间隔内的总位移。每帧位移图定义在该帧自身的像素网格与相机坐标系上，因此 Track4D 是前景的三维场景流序列，一条表面点轨迹由相邻帧的对应关系逐段连接。相较于以固定锚帧像素索引的累积位移，逐帧定义避免长时程遮挡与出画造成的大面积失效，并使每一帧 Track 与同时刻的 Video 帧在空间上对齐。

机器人部分由关节状态、连杆几何与正向运动学在渲染的可见表面上建立对应，因此在给定动作下由运动学唯一确定；物体部分由几何对应或三维运动估计提供，并保留有效性标记。前者描述动作的执行主体，后者描述动作作用于环境的后果，无法由机器人运动学推出。这一区分决定了后文的损失归一化与评测均按本体与物体分列。

**RGB 编码。** 将 $D_t$ 的三个分量分别编码为 RGB 三通道。对坐标轴 $c\in\{x,y,z\}$，使用训练集统计得到并冻结的尺度 $s_c$，采用有符号 $\mu$-law 编码：

$$
z_c=\operatorname{clip}(D_{t,c}/s_c,-1,1),
\qquad
R_{t,c}=
\operatorname{round}
\left[
127.5\left(
1+\operatorname{sgn}(z_c)
\frac{\log(1+\mu|z_c|)}{\log(1+\mu)}
\right)
\right],
$$

其中 $\mu=31$。固定的全局尺度使相同颜色在不同样本中对应相同的物理位移；非线性编码提高小幅运动的量化分辨率。有效区域中的零位移编码为中性灰，背景及无效位置置零，并由独立的有效性掩码参与监督。

**编码与条件。** 一段 Track4D 由当前时刻的零位移锚帧及其后的位移图组成，经冻结的视频 VAE 编码为 Track latent；锚帧与当前观测对应，使 Track 与 Video 在 VAE 的时间结构上对齐。当前 RGB、米制深度和语义掩码分别编码为干净条件：RGB 提供外观，深度提供当前位置，语义掩码区分本体与物体，Track 则描述它们如何运动。未来帧的前景支持区与位移均由模型生成，推理时不输入未来掩码或未来 Track。

### 2.2 Video–Track–Action 联合建模

Video 专家预测可见的场景变化，Track 专家预测本体与物体的三维位移，Action 专家预测机器人动作块。三者覆盖同一物理预测窗口，允许使用不同的采样频率和 token 数量。以 $N_f$ 记世界模态在该窗口内的未来帧数，Video 与 Track 经 VAE 时间压缩后的 latent 帧数为 $N_f/\kappa$，$\kappa$ 为 VAE 的时间压缩率；Action 包含 $H$ 个动作步。

三专家接收各自的含噪输入与噪声嵌入，并在对应层交换特征；不同深度的专家按归一化层位置建立交互层映射。Track 在其中承担显式几何接口：Video 读取运动信息以生成与几何过程相符的画面，Action 读取交互后果以生成控制，Track 读取动作以建模动作条件的物体响应。第 3 节将 Action 对未来 Video、Track 的稠密读取替换为按交互显著度与耦合转变组织的紧凑读取；Action 对当前观测、本体状态与任务条件的读取保持不变。

当前观测和锚帧始终保持干净，不计入未来预测损失；条件 token 不读取含噪未来，以避免条件表示随预测内容改变。基础训练目标为

$$
\mathcal L_{\mathrm{WAM}}
=
\sum_{m\in\{V,T,A\}}
\lambda_m
\mathbb E
\left[
\left\|
v_\theta^m(x^V_{\sigma_V},x^T_{\sigma_T},x^A_{\sigma_A};O_t,q_t,\ell)
-(\epsilon^m-x^m)
\right\|_{\mathcal M_m}^{2}
\right],
$$

其中 $\mathcal M_m$ 表示该模态的监督区域。Action 排除填充维度；Video 排除干净锚帧；Track 将本体有效区域、物体有效区域与零背景分别归一化计损，避免大量零背景或由运动学决定的本体运动淹没物体响应的误差。

### 2.3 异步噪声与条件组合训练

训练不仅覆盖三模态共同生成，还覆盖"已知动作预测后果"和"缺少部分世界模态生成动作"两类条件。它们通过同一网络的噪声配置与注意力掩码实现。

**异步联合生成。** 三个专家分别接收各自的噪声水平，不强制 $\sigma_V=\sigma_T=\sigma_A$。训练混合两种采样方式：一部分样本独立采样三个噪声水平，覆盖不同可信度的模态组合；另一部分从推理使用的异步轨迹中采样，提高训练与推理的一致性。令 $u\in[0,1]$ 为整体去噪进度，$r_m$ 为模态 $m$ 完成去噪的进度位置，定义

$$
\sigma_m(u)=\max\left(1-\frac{u}{r_m},0\right),
\qquad r_A<r_T\leq r_V=1 .
$$

训练由此包含"动作仍含噪声、世界预测正在形成"和"动作已经干净、Video 和 Track 继续去噪"两类状态。按推理轨迹采样噪声组合的做法与 [X-WAM 的异步噪声训练](https://sharinka0715.github.io/X-WAM/) 一致，此处扩展到独立的 Video、Track 和 Action 三专家。

**干净动作条件。** 随机选取一部分样本，将完整真实动作块作为干净输入，置 $\sigma_A=0$，只计算 Video 和 Track 的生成损失。此时 Action token 仅编码动作与当前条件，不读取待生成的未来；Video 和 Track 则可读取这些动作特征。该分支直接训练 $p_\theta(V,T\mid A,O_t,q_t,\ell)$，使 Track 学习动作导致的交互后果，而非仅与视频共现的运动模式。异步轨迹中已到达零噪声的模态同样作为条件，不再计算其速度回归损失。

**随机模态丢弃。** 在动作生成样本中，随机屏蔽未来 Video、Track 中的一种或两种。丢弃作用于被屏蔽分支向其他专家提供的 K/V，并在所有交互层保持一致；该分支仍保留自身生成损失。当前观测和本体状态始终可见；被丢弃的未来模态不参与第 3 节读取模块的显著度、聚合与锚点计算，缺失位置由当前条件提供；丢弃 Track 时耦合场不可用，CSIA 退化为仅由内容得分驱动的注意力。模型因此同时学习三模态协同、仅依赖几何预见、仅依赖视觉预见以及仅依赖当前观测的控制方式。

### 2.4 多速率去噪与执行

推理时，三个模态从各自的噪声初始化，并沿上述异步轨迹更新。每一轮先根据当前三模态状态计算速度，再分别更新尚未完成去噪的模态：

$$
x^m_{\mathrm{next}}
=
x^m_{\sigma_m}
+
(\sigma_m^{\mathrm{next}}-\sigma_m)v_\theta^m .
$$

例如，在 16 轮世界预测预算下，可设置 Action 在第 4 轮完成、Track 在第 12 轮完成、Video 在第 16 轮完成。Action 达到零噪声后数值保持不变，转为 Video 和 Track 的条件；后两者继续细化所选动作对应的交互轨迹与视觉后果。

控制端在动作完成去噪时即可取得动作块，执行其前缀，并在下一次规划时使用新观测重新生成。后续世界细化对应已选定的动作，不回写已下发的控制。动作下发因此不必等待世界模态全部完成去噪。

Action 的完成位置 $r_A$ 决定动作生成时所读取的世界特征处于何种去噪状态：$r_A$ 越小，首包延迟越低，但 Action 读取的世界特征来自噪声更高的中间状态。这些中间状态仍以干净的当前观测、本体状态与任务为条件，其对控制的实际贡献由第 3 节的读取接口和实验中的依赖度诊断共同界定；$r_A$ 是需要在验证集上确定的设计变量，而非固定常数。

## 3. 交互显著的世界读取：CSIA 与 TAA

Track4D 已排除无关背景，但前景内部的信息对当前动作仍有不同价值。决定这一价值的不是运动幅度——机械臂自身总是运动最大的实体——而是本体与物体之间的运动关系是否正在改变：抓取是物体由静止转为随夹爪运动，释放是物体脱离随动，推动与滑脱介于二者之间。Track4D 将本体与物体的度量位移置于同一坐标系与时间轴，使这一关系可以作为一个几何量直接计算。我们据此定义本体—物体耦合场及其转变，并以其导出的交互显著度组织动作专家对世界预测的读取：空间上由 **Coupling-Salient Interaction Attention（CSIA）** 在显著度偏置下读取交互显著的本体—物体局部对，时间上由 **Transition-Anchored Aggregation（TAA）** 将逐帧交互 token 聚合到锚定于预测耦合转变的少数时刻。完整的 Video 与 Track 预测保持不变，压缩的只是供动作使用的未来信息。

### 3.1 子帧时间展开

世界模态经 VAE 时间压缩后，预测窗口内的 $N_f$ 个未来帧被折叠为 $N_f/\kappa$ 个 latent 帧；逐帧的运动信息保留在 latent 的通道维中，而非序列维。耦合转变是帧级事件，直接在 latent 帧上无法定位。因此读取端先将每个 latent 帧展开为 $\kappa$ 个子帧槽，而不改变骨干的 token 数与 VAE。

记 $h^m_{j,i}$ 为模态 $m\in\{V,T\}$ 在 latent 帧 $j$、空间位置 $i$ 的隐藏特征。对 $\tau\in\{1,\dots,\kappa\}$，定义子帧特征

$$
\hat h^m_{n,i}=W^m_\tau h^m_{j,i}+p_\tau,
\qquad n=\kappa(j-1)+\tau,
$$

其中 $W^m_\tau$ 为逐槽线性映射，$p_\tau$ 为槽位置嵌入；$n\in\{1,\dots,N_f\}$ 为帧级时间索引，与 Track 的位移行和 Action 的动作步在物理时间上对齐。Video 与 Track 使用相同的 VAE 与空间分辨率，因此共享 token 网格 $(n,i)$。

对 Track 的每个子帧槽附加一个轻量解码头 $g$，回归该帧在 token 网格上下采样后的位移：

$$
\tilde D_{n,i}=g(\hat h^T_{n,i}),
\qquad
\mathcal L_{\mathrm{unf}}
=
\frac{1}{|\mathcal V|}\sum_{(n,i)\in\mathcal V}
\left\|\tilde D_{n,i}-\bar D_{n,i}\right\|_1 ,
$$

$\bar D_{n,i}$ 为真实位移在 token 网格上的下采样，$\mathcal V$ 为有效前景位置集合。该损失使第 $\tau$ 个槽对应第 $\tau$ 个帧，并使 $\tilde D$ 成为当前去噪状态下可微的 token 级位移估计；下文的耦合场即由 $\tilde D$ 计算，不需要在每个去噪步解码 VAE。角色分类头同样作用于子帧特征，输出本体、物体、背景概率 $\rho_{n,i,r}$，在有效帧与位置上以交叉熵监督；真实掩码只作为标签。

### 3.2 本体—物体耦合场与交互显著度

对位置 $i$，以其空间邻域 $\mathcal N(i)$ 内本体 token 的位移加权平均作为该处的本体参考运动：

$$
\tilde D^{\mathrm{body}}_{n}(i)
=
\frac{\sum_{i'\in\mathcal N(i)}\omega_{i,i'}\,\rho_{n,i',\mathrm{body}}\,\tilde D_{n,i'}}
{\sum_{i'\in\mathcal N(i)}\omega_{i,i'}\,\rho_{n,i',\mathrm{body}}+\varepsilon},
$$

$\omega_{i,i'}$ 为固定的空间核。定义**本体—物体耦合场**

$$
r_{n,i}=\tilde D_{n,i}-\tilde D^{\mathrm{body}}_{n}(i),
\qquad
c_{n,i}=\frac{\|r_{n,i}\|}{\|\tilde D_{n,i}\|+\|\tilde D^{\mathrm{body}}_{n}(i)\|+\varepsilon}\in[0,1].
$$

$c_{n,i}$ 为归一化耦合状态：物体随本体共同运动时 $c\to 0$；物体静止而本体运动、或二者独立运动时 $c\to 1$。**耦合转变**定义为 $c$ 的时间变化 $\Delta_n c_{n,i}=c_{n,i}-c_{n-1,i}$：抓取对应 $\Delta c<0$，释放对应 $\Delta c>0$。

交互显著度由耦合状态、耦合转变与本体—物体邻近度三个物理量组合得到：

$$
s_{n,i}
=
\rho_{n,i,\mathrm{object}}\;
\phi\!\left([\,c_{n,i},\ |\Delta_n c_{n,i}|,\ \pi_{n,i},\ \|\tilde D^{\mathrm{body}}_{n}(i)\|\,];\ \ell,q_t\right),
\qquad
\pi_{n,i}=\sum_{i'\in\mathcal N(i)}\omega_{i,i'}\rho_{n,i',\mathrm{body}},
$$

其中 $\phi$ 为以任务指令与本体状态为条件的小型 MLP，输出非负标量。$\phi$ 决定当前任务更看重哪种耦合状态——抓取阶段看重转变，搬运阶段看重维持耦合，放置阶段看重物体与本体的解耦——但它的输入只有上述物理量，不能从外观或位置直接学出显著度。本体侧的显著度由物体侧传播：$s^{\mathrm{body}}_{n,i'}=\sum_i\omega_{i,i'}\rho_{n,i,\mathrm{object}}s_{n,i}$，使与显著物体相邻的本体部位同样显著。帧级转变强度取 $s_n=\sum_i s_{n,i}$。

显著度由模型自身的 Track 预测计算，随去噪进程更新，不依赖外部检测器或标注；仿真中的接触真值只在分析中用于检验 $s_n$ 的峰值是否落在接触与释放帧。

### 3.3 CSIA：耦合显著交互注意力

将任务指令、当前观测的池化特征与本体状态经 MLP 投影，加到 $K$ 个可学习向量上得到查询 $q_k$。在每个子帧 $n$，查询分别对物体侧与本体侧 token 做注意力，显著度作为注意力的对数偏置：

$$
\alpha^{m}_{k,n,i,r}
=
\operatorname{softmax}_{i}
\left[
\frac{(q_k+e_r)^\top W_K^m\hat h^m_{n,i}}{\sqrt d}
+\gamma_m\log\!\left(s^{r}_{n,i}+\varepsilon\right)
\right],
\qquad
a^{m}_{k,n,r}=\sum_i\alpha^{m}_{k,n,i,r}W_V^m\hat h^m_{n,i},
$$

其中 $r\in\{\mathrm{object},\mathrm{body}\}$，$s^{\mathrm{object}}_{n,i}=s_{n,i}$，$e_r$ 为角色嵌入，$\gamma_m\ge 0$ 为可学习的显著度偏置强度。显著度决定"哪些本体—物体局部对值得读"，内容得分决定"在这些局部对中读哪一部分"。$\gamma_m$ 可学习使模型在去噪早期 $\tilde D$ 不可靠时降低显著度的权重。

将两侧注意力输出与显著度加权的耦合向量合成为该帧的交互 token：

$$
\bar r_{k,n}=\sum_i\alpha^{T}_{k,n,i,\mathrm{object}}\,r_{n,i},
\qquad
e^m_{k,n}
=
\operatorname{MLP}\!\left([\,a^{m}_{k,n,\mathrm{object}};\ a^{m}_{k,n,\mathrm{body}};\ W_r\bar r_{k,n}\,]\right).
$$

耦合向量 $\bar r_{k,n}$ 显式携带该局部对的相对运动，使交互 token 既包含外观与几何特征，也包含度量意义下的耦合状态。

### 3.4 TAA：转变锚定聚合

每个查询在 CSIA 之后对应一条长度为 $N_f$ 的交互 token 序列 $\{e^m_{k,n}\}$。TAA 将该序列聚合为 $K$ 个 token，聚合中心锚定在帧级转变强度 $\{s_n\}$ 上：以归一化累积转变强度 $F(u)=\sum_{n\le uN_f}s_n/\sum_n s_n$ 的逆映射确定有序锚点，

$$
\bar c_k=F^{-1}\!\left(\frac{k-1/2}{K}\right),
\qquad
c_k=\operatorname{clip}\!\left(\bar c_k+\delta_k,\,0,\,1\right),
\qquad
w_k=w_{\min}+(w_{\max}-w_{\min})\operatorname{sigmoid}(b_k),
$$

其中 $F^{-1}$ 以线性插值实现并可微，$\delta_k$、$b_k$ 由任务条件与池化后的 Track 子帧特征预测。锚点按累积转变强度等分，因此转变集中的帧获得更多、更密的中心；当窗口内没有转变（$s_n$ 近似均匀）时，锚点退化为均匀采样。聚合权重为

$$
\beta^m_{k,n}
=
\operatorname{softmax}_{n}
\left[
\frac{q_k^\top W_\tau^m e^m_{k,n}}{\sqrt d}
-
\frac{(u_n-c_k)^2}{2w_k^2}
\right],
\qquad
z^m_k=\sum_n\beta^m_{k,n}e^m_{k,n},
\qquad u_n=n/N_f .
$$

Video 与 Track 共用锚点 $c_k$，分别计算内容权重。在 1–2 秒的预测窗口内通常只包含一次耦合转变，此时 TAA 的作用是以窄窗口集中聚合转变前后的帧并抑制无关时段；其定位精度通过 $c_k$ 与仿真接触事件时刻的误差检验。

### 3.5 读取接口与信息边界

经 CSIA 与 TAA，每个世界模态向 Action 提供 $K$ 个交互 token。Action 保留自身注意力和对当前条件的读取，将原先对全部未来 Video、Track token 的读取替换为对 $\{z_k^V,z_k^T\}_{k=1}^{K}$ 的读取；不保留可以绕过读取接口的稠密未来路径。Video 与 Track 的完整生成路径以及二者之间的读取保持不变。

可选地，动作 token 读取交互 token 时加入按物理时间对齐的偏置：第 $\tau$ 个动作步对中心 $c_k$ 靠近 $u_{\lceil\tau N_f/H\rceil}$ 的 token 施加可学习偏好。该偏置作为消融项检验，不改变接口定义。

压缩的对象是用于决策的未来信息，而非被预测的 Track4D 本身。子帧展开、显著度组合 $\phi$、聚合权重、锚点偏移与窗口宽度均通过动作 flow-matching 损失端到端学习；$\mathcal L_{\mathrm{unf}}$ 与角色分类损失分别提供帧级位移与本体—物体结构的监督。耦合场与显著度不受直接监督，其物理含义来自定义而非拟合。读取在各交互层的专家特征更新后执行，输入为当前去噪状态下的世界隐藏特征，结果送入 Action 的跨模态注意力。该模块不额外构造轨迹生成器，也不需要阶段标注。

## 4. 扩展：基于交互时刻的异步视频条件

TAA 将世界信息聚合为锚定在耦合转变上的少量有序时刻，因此也提供了读取不同节奏视频的接口。给定额外示范视频 $D$，我们保持目标机器人的 Track 与 Action 处于同一执行时间轴，仅为示范视频学习独立的读取位置。

$D$ 是完整可访问的条件视频，与需要生成的机器人未来视频 $V$ 使用独立时间轴。我们复用 Video 编码模块，以零噪声编码 $D$，缓存其时空特征；示范特征不读取机器人的含噪未来。随后沿用第 3.1–3.3 节的子帧展开、显著度与 CSIA，将每个示范帧表示为本体—物体交互特征；示范侧的位移估计由同一解码头 $g$ 在 Video 编码特征上给出，属于二维近似，仅用于显著度与聚合。

### 4.1 从执行时刻到示范位置

机器人预测窗口中的第 $k$ 个交互时刻由 Track token $z_k^T$ 表示。将该 token 与当前状态、任务查询共同输入一个轻量对齐模块，使其读取示范的交互特征及归一化位置编码，预测该时刻在示范中的位置 $d_k$。

对齐模块由一层交叉注意力和一个两层 MLP 构成，一次前向同时输出当前示范进度 $b$ 与 $K$ 个增量：

$$
d_k
=
b+(1-b)
\frac{\sum_{r=1}^{k}\operatorname{softplus}(\delta_r)}
{1+\sum_{r=1}^{K}\operatorname{softplus}(\delta_r)},
\qquad b\in[0,1].
$$

$b$ 由当前机器人观测与示范内容共同估计，$\delta_r$ 决定相邻时刻在示范中的间距。间距可以不均匀，因此能够表达局部节奏差异。

在第 3.4 节的时间聚合中，将示范分支的中心替换为 $d_k$，得到示范交互 token $z_k^D$。Track 时刻 $z_k^T$ 与对应示范时刻 $z_k^D$ 经线性投影和可学习门控融合，供 Track 后续层与 Action 读取；机器人未来 Video、Track 和 Action 的物理时间定义均保持不变。

每次闭环重规划根据新观测重新估计 $b$，使读取位置由实际交互状态决定。一次预测窗口内的时刻保持有序；不同重规划之间允许 $b$ 回退，以处理执行停顿或操作重试。

### 4.2 通过可控时间变换学习对齐

训练时，从配对机器人轨迹构造示范条件，对示范视频随机施加全局变速、分段变速和重复帧；机器人观测、Track 目标与 Action 目标保持不变。记已知时间变换为 $\psi$：示范中的位置 $u$ 对应原始轨迹中的物理时刻 $\psi(u)$。

对机器人预测窗口中的时间中心 $c_k$，其原始物理时刻为 $t_k^\star=t+Hc_k$。据此构造示范时间位置上的软目标分布并监督示范分支的聚合权重：

$$
\pi_{k,j}
=
\frac{\exp\!\left[-(\psi(u_j^D)-\operatorname{sg}(t_k^\star))^2/(2\eta^2)\right]}
{\sum_l\exp\!\left[-(\psi(u_l^D)-\operatorname{sg}(t_k^\star))^2/(2\eta^2)\right]},
\qquad
\mathcal L_{\mathrm{align}}
=
\frac{1}{K}\sum_k
\operatorname{KL}(\pi_k\|\beta_k^D)
+
\operatorname{Huber}(\psi(b)-t).
$$

所有时刻在计算前统一归一化到原始轨迹范围，$\eta$ 控制标注容差，$\operatorname{sg}$ 表示停止梯度。分布监督使注意力集中到对应时段，而非仅让多个不相关时段的平均时间接近目标；第二项监督当前进度。由于 $\psi$ 来自可控时间变换，该监督无需阶段标注，也允许重复帧对应相同物理时刻。

此外，在动作生成样本上，对原视频和变速视频使用相同的含噪动作、噪声水平及其他条件，以两次 Action 速度场预测的均方差构成节奏一致性损失 $\mathcal L_{\mathrm{rate}}$。它要求播放节奏的变化只影响示范的读取位置，不改变机器人应执行的动作。

最终训练目标为

$$
\mathcal L
=
\mathcal L_{\mathrm{WAM}}
+
\lambda_{\mathrm{role}}\mathcal L_{\mathrm{role}}
+
\lambda_{\mathrm{unf}}\mathcal L_{\mathrm{unf}}
+
\lambda_{\mathrm{align}}\mathcal L_{\mathrm{align}}
+
\lambda_{\mathrm{rate}}\mathcal L_{\mathrm{rate}} .
$$

无示范样本仅使用前三项并关闭示范读取分支；有示范样本增加时间对齐与节奏一致性监督。异步适配因此复用耦合转变及其注意力接口，扩展视频条件的可用方式，而不改变以几何运动预测和机器人控制为核心的训练任务。
