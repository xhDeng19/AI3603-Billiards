# NewAgent 开发总结

## 目标与约束
- 目标：在八球台球规则与带噪物理环境下，输出合法、稳健、具竞争力的击球动作参数（V0、phi、theta、a、b）。
- 约束：不得修改环境与裁判逻辑；动作范围需严格裁剪；首球必须先碰己方目标球；清台后黑8为唯一目标；白球+黑8同杆入袋判负。
- 噪声：环境会对所有动作加入高斯噪声（V0、phi、theta、a、b），策略需具鲁棒性。

## 架构设计
- 两阶段决策：
  - 候选生成：基于“幽灵球”几何为每个目标球与每个球袋生成理想击球方向与力度，并加入微扰增强鲁棒性。
  - 带噪仿真评分：复制当前状态，在仿真中注入噪声并依据与裁判一致的规则打分，选平均分最高的动作。
  - 回退策略：若评分低于阈值，优先选择“安全球”而非完全随机动作，降低白球落袋与首球犯规风险。

## 关键算法
- 幽灵球几何
  - 思路：将目标球沿着通往袋口的方向回退一个“两个球半径”的距离，得到“幽灵球”位置；白球指向幽灵球中心的连线即理想击打方向。
  - 位置：new_agent.py 的 ghost 计算函数 [new_agent.py:_ghost_target](file:///d:/SJTU/2025.9/Principles_and_Applications_of_Artificial_Intelligence/Final_Project/agents/new_agent.py#L136-L146)。
- 遮挡判定
  - 思路：对“白球→幽灵球”线段与“目标球→袋口”线段分别做直线-圆最近距离判定，若小于球半径则视为遮挡，剔除该路线；判定阈值为 1.25×球半径，偏保守以降低误判。
  - 位置：白球路径 [new_agent.py:_has_clear_path](file:///d:/SJTU/2025.9/Principles_and_Applications_of_Artificial_Intelligence/Final_Project/agents/new_agent.py#L148-L165)，目标球路径 [new_agent.py:_has_clear_path_obj_to_pocket](file:///d:/SJTU/2025.9/Principles_and_Applications_of_Artificial_Intelligence/Final_Project/agents/new_agent.py#L166-L182)。
- 首碰判定（射线近似）
  - 思路：沿给定 φ 方向从白球做射线，计算与各球的垂距与前向投影，选择满足垂距≤球半径且投影t>0的最小t作为“首个可接触球”；并进行小扰动一致性校验（φ±0.3°范围内多数样本首碰一致）后才保留候选。
  - 位置： [new_agent.py:_first_ball_along_phi](file:///d:/SJTU/2025.9/Principles_and_Applications_of_Artificial_Intelligence/Final_Project/agents/new_agent.py#L183-L197)。
- 候选生成
  - 对每个未进袋的己方目标球与最近的 2 个袋口，计算理想角度与距离，映射力度 v_base（上限 5.8）。近距离时略微收敛力度避免过冲；对理想方向强制首碰校验，若首球不是该目标球则丢弃该路线。
  - 微扰版本：角度微调 ±0.5° 与 ±1.0°，结合轻回旋 b≈-0.02；引入“落位侧旋”（a≈±0.03），根据目标球到己方目标质心的相对方向选择左右侧旋以改善下一杆连贯性。
  - 中心击打候选：若白球至目标球中心的直线路径可行，加入“直接击打目标球中心”的候选（含 ±0.8° 微调与轻回旋），降低“未接触任何球”的犯规概率。
  - 保证首碰候选：为每个目标球计算与中心方向的“切线偏移”角 δ≈arcsin(R/dist)，尝试 φ±δ、±2δ 等，使用“首碰判定”仅保留首球为该目标球的候选。
  - 若无进攻路线，生成安全球（白球指向最近库边，θ=0，轻回旋 b≈-0.02，力度按距离映射）；对安全球同样采用首碰一致性校验以提升合法性鲁棒性。
  - 安全球（保证接触版）：优先扫描最近的若干球，围绕中心方向尝试多组 φ 微偏，使用“首碰判定”只保留首碰为该球的候选；若仍无，则退化到库边安全球。
  - 位置：候选生成入口 [new_agent.py:_generate_candidates](file:///d:/SJTU/2025.9/Principles_and_Applications_of_Artificial_Intelligence/Final_Project/agents/new_agent.py#L184-L264)，安全球生成 [new_agent.py:_generate_safety_candidates](file:///d:/SJTU/2025.9/Principles_and_Applications_of_Artificial_Intelligence/Final_Project/agents/new_agent.py#L293-L322)。
- 带噪仿真
  - 在候选仿真中对动作参数注入噪声并裁剪至合法范围，模拟真实环境扰动。
  - 位置：仿真函数 [new_agent.py:_simulate_action](file:///d:/SJTU/2025.9/Principles_and_Applications_of_Artificial_Intelligence/Final_Project/agents/new_agent.py#L294-L309)。
- 评分函数（对齐裁判）
  - 加分：己方进球（+50/球）、合法打进黑8（+100）、合法无进球（+10）。
  - 扣分：白球入袋（-100）、非法黑8或同杆白+黑8（-150）、首球犯规（-30）、无进球且未碰库（-30）、对方球入袋（-20/球）、未接触任何球额外惩罚（-70，总计约-100）。
  - 位置：评分函数 [new_agent.py:_analyze_shot_for_reward](file:///d:/SJTU/2025.9/Principles_and_Applications_of_Artificial_Intelligence/Final_Project/agents/new_agent.py#L376-L434)。
 - 下一杆与对手可打性评分（轻量估计）
  - 思路：在仿真结果上，选最近的 3 个己方目标球与各自 2 个最近袋口，检查“白球→幽灵球”和“目标球→袋口”的直线可行性，统计可行比例作为下一杆可打性率。
  - 对手可打性：以相同方法评估对手目标球的可行比例，作为对手可打性率，用负权重纳入鲁棒评分以降低“给对手留好球”的风险。
  - 白球落袋风险：依据白球到各袋中心的最小距离 dmin，用 exp(-dmin/0.25) 归一化为 [0,1] 作为风险度量，加入负权重。

## 决策流程
 - 清台判定：若 my_targets 全进袋，目标切换为 ['8']（必须首碰黑8）。见 [decision](file:///d:/SJTU/2025.9/Principles_and_Applications_of_Artificial_Intelligence/Final_Project/agents/new_agent.py#L19-L130) 中的剩余目标判断与切换。
- 两阶段评分：
  - 第一阶段对全部候选进行 1 次带噪仿真，选出 Top-4 候选。
  - 第二阶段对 Top-4 每个候选进行 3–4 次带噪仿真（目标4，预算紧张时降至3），以“鲁棒评分”选最优：平均分 + Wc×首碰己方率 − Wq×白球入袋率 + Wn×下一杆可打性率 − Wo×对手可打性率 − Wr×白球落袋风险。
  - 阶段权重：常规阶段 Wc=26、Wq=60、Wn=8、Wo=10、Wr=8；黑8阶段 Wc=20、Wq=80、Wn=4、Wo=14、Wr=12。
  - 局部微调：仅评估 4–5 个轻量变体（φ±0.25°、V0+0.2、b=-0.03），每个 2 次仿真；在预算充裕时启用，整体开销约 10 次仿真。
  - 回退策略：当最佳动作平均分低于阈值（如 <10）时，评估安全球候选（3 次仿真），同样使用阶段化鲁棒评分（含对手可打性与白球风险）选取最佳，不再使用完全随机动作。
  - 思考时长控制：为每杆设置约 10s 的时间预算，超出预算时直接返回当前最优或随机兜底动作，避免单局累计超时。

## 重要参数与范围
- 球半径：ball_radius = 0.028575。见 [new_agent.py:L15-L17](file:///d:/SJTU/2025.9/Principles_and_Applications_of_Artificial_Intelligence/Final_Project/agents/new_agent.py#L15-L17)。
- 仿真噪声（Agent 侧）：V0=0.1、phi=0.12、theta=0.08、a/b=0.004。见 [new_agent.py:L15-L17](file:///d:/SJTU/2025.9/Principles_and_Applications_of_Artificial_Intelligence/Final_Project/agents/new_agent.py#L15-L17)。
- 力度映射（更稳健）：v_base ≈ 1.2 + 1.2 × 距离，裁剪至 [1.0, 5.8]；冗余力度 v_base+0.6（上限 6.0），并加入轻微回旋 b≈-0.02 以降低白球入袋风险。
 - 动作范围：V0∈[0.5,8.0]、phi∈[0,360)、theta∈[0,90]、a/b∈[-0.5,0.5]；最终返回时统一裁剪。见 [decision](file:///d:/SJTU/2025.9/Principles_and_Applications_of_Artificial_Intelligence/Final_Project/agents/new_agent.py#L125-L130)。
- 环境噪声与裁剪：参考 [poolenv.py:L259-L277](file:///d:/SJTU/2025.9/Principles_and_Applications_of_Artificial_Intelligence/Final_Project/poolenv.py#L259-L277) 的噪声与裁剪逻辑。

## 评估与调试
- 评估与调试
  - 评估脚本：公平轮换先后手与球型；统计胜负与得分。见 [evaluate.py](file:///d:/SJTU/2025.9/Principles_and_Applications_of_Artificial_Intelligence/Final_Project/evaluate.py)。
  - 调试建议：对战局数可临时下调至 40 以加速迭代，正式评估恢复默认设置。
- 运行：在 Conda 环境中执行 `python evaluate.py`，终端输出包含犯规、入袋与黑8判定日志，末尾打印“最终结果”统计。
- 随机控制：调试阶段可使用 [utils.py:set_random_seed](file:///d:/SJTU/2025.9/Principles_and_Applications_of_Artificial_Intelligence/Final_Project/utils.py#L9-L38) 固定种子；正式评估保持默认随机。

## 设计权衡
- 合规优先：评分函数与裁判完全对齐，避免“评分很好但实际犯规”的偏差。
- 鲁棒优先：
  - 候选限定：仅选距离白球最近的最多 4 个目标球；每球优先 2 个最近袋口。
  - φ与力度的微扰（±0.4° 与 +0.6 力度），加入小幅侧旋（|a|≈0.02）降低白球入袋风险。
 - 时间权衡：候选数量上限约 12、两阶段总仿真数受控（约 20 次/杆），局部微调约 10 次，总计约 30 次/杆；安全球约 12 次以内。整体思考时长与此前相当但犯规风险更低。

## 可扩展优化
- 搜索强化：将候选池接入小规模 MCTS（回传平均回报选“鲁棒子”），或最优附近做贝叶斯优化微调。
- 布局评分：安全球中加入“目标球靠袋”“白球停库边/障碍后”等布局改良项。
- 路线扩展：考虑一库/两库反弹路线的可行性判定与容错窗口。

## 代码参考索引
 - 决策主逻辑 [decision](file:///d:/SJTU/2025.9/Principles_and_Applications_of_Artificial_Intelligence/Final_Project/agents/new_agent.py#L19-L130)
 - 幽灵球计算 [_ghost_target](file:///d:/SJTU/2025.9/Principles_and_Applications_of_Artificial_Intelligence/Final_Project/agents/new_agent.py#L136-L146)
 - 直线遮挡判定 [_has_clear_path](file:///d:/SJTU/2025.9/Principles_and_Applications_of_Artificial_Intelligence/Final_Project/agents/new_agent.py#L148-L165)
 - 候选生成 [_generate_candidates](file:///d:/SJTU/2025.9/Principles_and_Applications_of_Artificial_Intelligence/Final_Project/agents/new_agent.py#L184-L264)
 - 安全球生成 [_generate_safety_candidates](file:///d:/SJTU/2025.9/Principles_and_Applications_of_Artificial_Intelligence/Final_Project/agents/new_agent.py#L266-L292)
 - 带噪仿真 [_simulate_action](file:///d:/SJTU/2025.9/Principles_and_Applications_of_Artificial_Intelligence/Final_Project/agents/new_agent.py#L294-L309)
 - 局部微调 [_local_refine](file:///d:/SJTU/2025.9/Principles_and_Applications_of_Artificial_Intelligence/Final_Project/agents/new_agent.py#L311-L312)
 - 规则一致评分 [_analyze_shot_for_reward](file:///d:/SJTU/2025.9/Principles_and_Applications_of_Artificial_Intelligence/Final_Project/agents/new_agent.py#L376-L434)
