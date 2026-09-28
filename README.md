# GridWorld 强化学习：DQN、Actor–Critic 与 PPO
以下都是AI写的，这个仓库主要是一个学习用途
本目录使用同一个 5×5 GridWorld 学习和比较三类强化学习算法：DQN、Actor–Critic（AC）和 PPO。每套实现都包含环境、网络、采样、训练、确定性基线评估和结果可视化；主要入口是各目录下的 `main.ipynb`。

> 三个子项目的环境语义和训练方式并不完全相同，实验结果不能只按 loss 或训练轮数直接横向比较。尤其是 DQN 中目标格是非终止格，而 AC/PPO 到达目标后终止。

## 项目结构

```text
code/
├── README.md                 # 三种算法的统一说明
├── DQN/
│   ├── config.py             # 环境、DQN、回放缓冲区和网络参数
│   ├── env.py                # 连续任务 GridWorld
│   ├── net.py                # 主 Q 网络和目标 Q 网络
│   ├── rollout.py            # 状态编码、ReplayBuffer、随机数据采集
│   ├── runner.py             # DQN 动作选择与 TD 更新
│   ├── evaluate.py           # 值迭代基线和确定性策略精确评估
│   ├── visualize.py          # loss、策略及 Q 值可视化
│   └── main.ipynb            # DQN 完整实验入口
├── AC/
│   ├── config.py             # 环境、Actor–Critic 和收敛验收参数
│   ├── env.py                # 终止型 GridWorld 与批量转移
│   ├── net.py                # Actor–Critic 网络
│   ├── rollout.py            # 状态编码和采样辅助
│   ├── runner.py             # 一步 TD Actor–Critic 更新
│   ├── evaluate.py           # 最优基线、随机/贪心策略精确评估
│   ├── visualize.py          # 训练、收敛、策略和价值图
│   ├── train.py              # 可直接运行的训练与产物保存脚本
│   ├── main.ipynb            # AC 完整实验入口
│   └── README.md             # AC 的详细实验和验收说明
└── PPO/
    ├── config.py             # 环境、网络、rollout、PPO 和熵系数参数
    ├── env.py                # 支持并行 batch 的终止型 GridWorld
    ├── net.py                # 独立 Actor 与 Critic MLP
    ├── rollout.py            # 并行 on-policy 轨迹采集
    ├── runner.py             # PPO 裁剪目标和 Critic 更新
    ├── evaluate.py           # 值迭代 baseline 与确定性策略精确评估
    ├── vizualize.py          # 训练、评估、策略和 baseline 对比图
    └── main.ipynb            # PPO 完整实验入口
```

`PPO/vizualize.py` 的文件名沿用项目当前拼写。导入时必须写作 `from vizualize import ...`。

## 公共环境

网格中的单元类型为：

- `0`：普通格；
- `1`：可进入的惩罚格，不是不可穿越的墙；
- `2`：目标格；
- 动作 `0/1/2/3/4`：上、右、下、左、停留。

越界时智能体留在原位置并获得边界奖励。地图、奖励和折扣率以各 notebook 创建的配置对象为准，而不是只看 `config.py` 的默认值。

环境语义存在一项重要差异：

| 实现 | 目标格 | 状态表示 | 采样方式 |
|---|---|---|---|
| DQN | 非终止，可继续获得后续奖励 | 二维归一化坐标 | 一次随机采集后放入经验回放 |
| AC | 到达即终止，未来价值为 0 | 25 维 one-hot | 从全部非终止状态均衡采样 |
| PPO | 到达即终止，未来价值为 0 | 25 维 one-hot | 多环境并行 on-policy rollout |

## 三种算法

### DQN

DQN 使用主 Q 网络估计动作价值，并使用周期同步的目标网络构造 TD 目标：

```text
随机策略采集 → ReplayBuffer → 主网络更新 → 定期同步目标网络
```

当前实现的特点：

- 状态编码为归一化后的 `(row, column)`；
- 五个网络输出分别对应五个动作的 Q 值；
- 训练数据由 `uniform_rollout` 随机采集并保存在固定容量回放缓冲区中；
- 使用目标网络最大 Q 值构造一步 TD 目标；
- `evaluate.py` 对完整有限 MDP 做值迭代，并通过线性方程精确计算学习策略价值；
- 目标格在 DQN 环境中不是终止状态，因此 baseline 也是无限时域折扣连续任务基线。

这是一套教学型 DQN。当前动作选择接口使用贪心动作，训练数据主要来自预先采集的随机回放，并未实现训练过程中持续变化的 epsilon-greedy 在线探索。

### Actor–Critic

AC 同时训练策略 Actor 和状态价值 Critic：

```text
全部非终止状态 → 按当前策略采样动作 → 一步 TD 优势 → 更新 Actor/Critic
```

当前实现的特点：

- one-hot 状态表示，相当于对固定有限状态空间进行表格化神经参数建模；
- 每次更新都从全部非终止状态均衡采样，避免无限循环轨迹垄断训练数据；
- Actor 使用一步 TD advantage，Critic 回归一步 TD target；
- 包含熵系数衰减、学习率衰减、奖励尺度归一化和梯度裁剪；
- 使用值迭代得到最优参考，并精确评估当前随机策略和贪心策略；
- `train.py` 支持连续多次通过验收后提前停止，并保存完整诊断产物。

更详细的收敛口径、历史结果和输出字段见 `AC/README.md`。

### PPO

PPO 在当前策略采集的数据上，使用旧策略 log probability 与新策略 log probability 的比值构造裁剪代理目标：

```text
并行环境 rollout → 保存 old_log_prob → 随机小批量 → clipped surrogate → Actor/Critic 更新
```

当前实现的特点：

- 独立的 Actor 和 Critic MLP，输入为状态编号的 one-hot 表示；
- 每轮采集 `steps_per_env × rollout_batch` 条转移；
- 同一批 rollout 数据执行 `update_per_iter` 次随机小批量更新；
- Actor 使用 PPO clipped surrogate，Critic 使用一步 TD target；
- advantage 在每个小批量内标准化；
- 熵系数从 `entropy_coef_start` 线性衰减到 `entropy_coef_end`，每轮 rollout 内保持不变；
- `evaluate.py` 用值迭代生成确定性 baseline，并排除终止格后精确计算策略价值差；
- 当前版本尚未实现多步回报或 GAE。

PPO notebook 当前的奖励、熵系数和训练轮数可能与 `config.py` 默认值不同，应以 notebook 的集中配置单元格为准。零奖励停留、稀疏目标奖励和过早降低探索都可能使策略陷入“原地停留”的局部最优。

## 数据流

```text
配置
  ├──> GridWorld 环境 ──> transition / reward / done
  ├──> 网络结构
  └──> 训练与采样参数

环境交互 ──> rollout 或 ReplayBuffer ──> Runner 更新网络
                                         │
完整环境模型 ──> 值迭代 baseline ─────────┤
                                         v
                              精确指标、策略图和价值图
```

动态规划 baseline 只用于评估和收敛判定，不会作为 Actor、Critic 或 Q 网络的监督标签。

## 环境依赖

核心依赖：

- Python 3.10 或更高版本；
- NumPy；
- PyTorch；
- Matplotlib；
- Jupyter Notebook 或 JupyterLab；
- IPython（用于 notebook 内图片显示）。

项目目前没有锁定依赖版本的 `requirements.txt` 或 `environment.yml`。可在独立虚拟环境中安装：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install numpy torch matplotlib jupyter ipython
```

如需 CUDA，请根据本机驱动和 CUDA 环境安装对应的 PyTorch 构建；AC 和 PPO 当前实现默认按 CPU 路径使用，DQN 支持 `cpu`、`cuda` 或 `auto` 设备配置。

PPO 的 `env.py` 仍使用了旧 NumPy 类型别名，`main.ipynb` 已在导入前提供 `np.int64`/`np.bool_` 兼容映射。若绕过 notebook 直接复用 PPO 环境，应先完成同等兼容处理，或将环境源码中的旧别名替换为新类型。

## 运行方法

建议在仓库根目录启动 Jupyter，以保证相对路径和输出目录一致：

```powershell
jupyter lab
```

随后分别打开并从上到下运行：

- `code/DQN/main.ipynb`
- `code/AC/main.ipynb`
- `code/PPO/main.ipynb`

AC 也可以直接通过脚本运行：

```powershell
python code/AC/train.py
```

只计算动态规划 baseline、不训练网络：

```powershell
python code/DQN/evaluate.py
python code/PPO/evaluate.py
```

修改 Python 模块后，应重启 notebook kernel 再执行“Run All”，避免内存中继续使用旧类定义。运行前还应确认 notebook 选择的 kernel 正是安装了上述依赖的环境。

## 输出

不同 notebook 默认会将图片或实验产物写入工作区的 `output/`：

- DQN：训练 loss、贪心策略及 baseline 对比图；
- AC：模型、配置、训练/评估 CSV、逐状态诊断、价值数组和多张收敛图；
- PPO：训练曲线、精确评估曲线、策略/Critic 图和 baseline 对比图。

图中的移动平均只用于展示，原始训练数据仍应保留。最优动作命中率、价值差和 Bellman residual 的含义不同，不能用低 loss 代替策略最优性判断。

## 可复现性与结果边界

- 三套实现均提供随机种子；AC 还显式配置确定性 PyTorch 算法和 CPU 线程数。
- 确定性设置不保证不同操作系统、PyTorch 版本和硬件得到逐位一致的结果。
- 值迭代只证明给定有限 GridWorld 和奖励配置下的基准，不证明神经网络对其他地图具有泛化能力。
- DQN、AC、PPO 的环境终止规则、状态编码、采样量和“训练一步”含义不同，训练轮数不可直接比较。
- Critic loss 或 TD loss 接近零只表示满足当前数据分布下的目标，不等价于策略达到最优。
- PPO 仍是一份简化教学实现；若用于更一般的任务，应考虑 GAE、多步回报、梯度裁剪、设备管理、检查点和更完整的实验记录。

## AI 辅助情况

本项目开发过程中使用了 OpenAI Codex 作为编程辅助工具。已知的辅助范围包括：

- 阅读和梳理现有 DQN、AC、PPO 模块及其数据流；
- 辅助实现 PPO 的 `evaluate.py`、`vizualize.py` 和 `main.ipynb` 组织结构；
- 辅助设计值迭代 baseline、确定性策略价值比较和训练结果可视化；
- 检查 PPO 熵系数衰减逻辑、训练循环参数传递和结果解释；
- 整理本 README 的项目结构、依赖、运行说明和局限性。

项目维护者负责确定任务、地图、奖励、超参数和实验口径，执行训练并对结果作最终判断。AI 生成或建议的代码不被视为实验结论本身；baseline 数值、训练指标和图片均由本地程序计算。提交、引用或继续开发前，应由维护者复核代码、配置、日志和输出文件。

为避免误读，AI 辅助不代表：

- 三种算法在完全相同的 MDP 上进行了公平基准比较；
- 当前超参数已经过系统搜索；
- notebook 中一次运行的结果可以推广到其他随机种子或环境；
- 自动生成的代码无需人工审查或测试。
