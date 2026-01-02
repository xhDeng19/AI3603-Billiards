"""
evaluate.py - Agent 评估脚本

功能：
- 让两个 Agent 进行多局对战
- 统计胜负和得分
- 支持切换先后手和球型分配

使用方式：
1. 修改 agent_b 为你设计的待测试的 Agent， 与课程提供的BasicAgent对打
2. 调整 n_games 设置对战局数（评分时设置为120局来计算胜率）
3. 运行脚本查看结果
"""

# 导入必要的模块
from utils import set_random_seed
from poolenv import PoolEnv
from agents import BasicAgent, BasicAgentPro, NewAgent
import os
import sys
from datetime import datetime
import time

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

log_dir = os.path.join(os.path.dirname(__file__), 'log')
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.log')
_log_file = open(log_path, 'w', encoding='utf-8')
sys.stdout = Tee(sys.stdout, _log_file)
sys.stderr = Tee(sys.stderr, _log_file)

# 设置随机种子，enable=True 时使用固定种子，enable=False 时使用完全随机
# 根据需求，我们在这里统一设置随机种子，确保 agent 双方的全局击球扰动使用相同的随机状态
set_random_seed(enable=False, seed=42)

env = PoolEnv()
results = {'AGENT_A_WIN': 0, 'AGENT_B_WIN': 0, 'SAME': 0}
n_games = 6  # 对战局数

## 选择对打的对手
agent_a, agent_b = NewAgent(), NewAgent() # NewAgent 对战 NewAgent

players = [agent_a, agent_b]  # 用于切换先后手
target_ball_choice = ['solid', 'solid', 'stripe', 'stripe']  # 轮换球型

for i in range(n_games): 
    print()
    print(f"------- 第 {i} 局比赛开始 -------")
    env.reset(target_ball=target_ball_choice[i % 4])
    agent_A_idx = i % 2
    agent_B_idx = (i + 1) % 2
    player_class = players[agent_A_idx].__class__.__name__
    ball_type = target_ball_choice[i % 4]
    print(f"本局 Player A: {player_class}, 目标球型: {ball_type}")
    game_think_time_A = 0.0
    game_think_time_B = 0.0
    agent_A_name = players[agent_A_idx].__class__.__name__
    agent_B_name = players[agent_B_idx].__class__.__name__
    # 每局统计
    game_stats = {
        'A': {'shots': 0, 'pockets': 0, 'fouls': 0, 'fo_first': 0, 'no_hit': 0, 'no_rail': 0, 'white_in': 0, 'max_streak': 0, 'curr_streak': 0},
        'B': {'shots': 0, 'pockets': 0, 'fouls': 0, 'fo_first': 0, 'no_hit': 0, 'no_rail': 0, 'white_in': 0, 'max_streak': 0, 'curr_streak': 0}
    }
    while True:
        player = env.get_curr_player()
        print(f"[第{env.hit_count}次击球] player: {player}")
        obs = env.get_observation(player)
        if player == 'A':
            _ts = time.perf_counter()
            action = players[agent_A_idx].decision(*obs)
            game_think_time_A += time.perf_counter() - _ts
        else:
            _ts = time.perf_counter()
            action = players[agent_B_idx].decision(*obs)
            game_think_time_B += time.perf_counter() - _ts
        step_info = env.take_shot(action)
        # 统计当前杆
        st = game_stats[player]
        st['shots'] += 1
        own_pockets = step_info.get('ME_INTO_POCKET', []) or []
        st['pockets'] += len(own_pockets)
        if len(own_pockets) > 0:
            st['curr_streak'] += 1
            if st['curr_streak'] > st['max_streak']:
                st['max_streak'] = st['curr_streak']
        else:
            st['curr_streak'] = 0
        if step_info.get('FOUL_FIRST_HIT'):
            st['fouls'] += 1
            st['fo_first'] += 1
        if step_info.get('NO_HIT'):
            st['fouls'] += 1
            st['no_hit'] += 1
        if step_info.get('NO_POCKET_NO_RAIL'):
            st['fouls'] += 1
            st['no_rail'] += 1
        if step_info.get('WHITE_BALL_INTO_POCKET'):
            st['fouls'] += 1
            st['white_in'] += 1
        
        done, info = env.get_done()
        if not done:
            # poolenv中已有打印，无需再输出
            # if step_info.get('FOUL_FIRST_HIT'):
            #     print("本杆判罚：首次接触对方球或黑8，直接交换球权。")
            # if step_info.get('NO_POCKET_NO_RAIL'):
            #     print("本杆判罚：无进球且母球或目标球未碰库，直接交换球权。")
            # if step_info.get('NO_HIT'):
            #     print("本杆判罚：白球未接触任何球，直接交换球权。")
            # if step_info.get('ME_INTO_POCKET'):
            #     print(f"我方球入袋：{step_info['ME_INTO_POCKET']}")
            if step_info.get('ENEMY_INTO_POCKET'):
                print(f"对方球入袋：{step_info['ENEMY_INTO_POCKET']}")
        if done:
            # 统计结果（player A/B 转换为 agent A/B） 
            if info['winner'] == 'SAME':
                results['SAME'] += 1
            elif info['winner'] == 'A':
                results[['AGENT_A_WIN', 'AGENT_B_WIN'][i % 2]] += 1
            else:
                results[['AGENT_A_WIN', 'AGENT_B_WIN'][(i+1) % 2]] += 1
            print(f"本局思考总时长：AGENT_A({agent_A_name})={game_think_time_A:.2f}s, AGENT_B({agent_B_name})={game_think_time_B:.2f}s")
            if 'THINK_TIMES' not in results:
                results['THINK_TIMES'] = []
            results['THINK_TIMES'].append({'AGENT_A': round(game_think_time_A, 2), 'AGENT_B': round(game_think_time_B, 2)})
            # 存储本局详细统计
            if 'DETAILS' not in results:
                results['DETAILS'] = []
            results['DETAILS'].append({
                'game': i,
                'winner': info['winner'],
                'A': game_stats['A'],
                'B': game_stats['B']
            })
            break

# 计算分数：胜1分，负0分，平局0.5
results['AGENT_A_SCORE'] = results['AGENT_A_WIN'] * 1 + results['SAME'] * 0.5
results['AGENT_B_SCORE'] = results['AGENT_B_WIN'] * 1 + results['SAME'] * 0.5

print("\n最终结果：", results)
print("日志文件:", log_path)

# 策略总结（基于统计）
if 'DETAILS' in results and len(results['DETAILS']) > 0:
    total = len(results['DETAILS'])
    winners = {'fouls': 0, 'white_in': 0, 'fo_first': 0, 'no_hit': 0, 'max_streak': 0, 'pockets_per_shot': 0.0}
    losers = {'fouls': 0, 'white_in': 0, 'fo_first': 0, 'no_hit': 0, 'max_streak': 0, 'pockets_per_shot': 0.0}
    for d in results['DETAILS']:
        w = d['winner']
        A = d['A']; B = d['B']
        if w == 'A':
            winners['fouls'] += A['fouls']; winners['white_in'] += A['white_in']; winners['fo_first'] += A['fo_first']; winners['no_hit'] += A['no_hit']; winners['max_streak'] += A['max_streak']; winners['pockets_per_shot'] += (A['pockets'] / max(1, A['shots']))
            losers['fouls'] += B['fouls']; losers['white_in'] += B['white_in']; losers['fo_first'] += B['fo_first']; losers['no_hit'] += B['no_hit']; losers['max_streak'] += B['max_streak']; losers['pockets_per_shot'] += (B['pockets'] / max(1, B['shots']))
        elif w == 'B':
            winners['fouls'] += B['fouls']; winners['white_in'] += B['white_in']; winners['fo_first'] += B['fo_first']; winners['no_hit'] += B['no_hit']; winners['max_streak'] += B['max_streak']; winners['pockets_per_shot'] += (B['pockets'] / max(1, B['shots']))
            losers['fouls'] += A['fouls']; losers['white_in'] += A['white_in']; losers['fo_first'] += A['fo_first']; losers['no_hit'] += A['no_hit']; losers['max_streak'] += A['max_streak']; losers['pockets_per_shot'] += (A['pockets'] / max(1, A['shots']))
        else:
            # 平局同时计入胜负均值避免偏斜
            winners['fouls'] += (A['fouls'] + B['fouls'])/2; winners['white_in'] += (A['white_in'] + B['white_in'])/2; winners['fo_first'] += (A['fo_first'] + B['fo_first'])/2; winners['no_hit'] += (A['no_hit'] + B['no_hit'])/2; winners['max_streak'] += (A['max_streak'] + B['max_streak'])/2; winners['pockets_per_shot'] += ((A['pockets']/max(1,A['shots'])) + (B['pockets']/max(1,B['shots'])))/2
            losers['fouls'] += (A['fouls'] + B['fouls'])/2; losers['white_in'] += (A['white_in'] + B['white_in'])/2; losers['fo_first'] += (A['fo_first'] + B['fo_first'])/2; losers['no_hit'] += (A['no_hit'] + B['no_hit'])/2; losers['max_streak'] += (A['max_streak'] + B['max_streak'])/2; losers['pockets_per_shot'] += ((A['pockets']/max(1,A['shots'])) + (B['pockets']/max(1,B['shots'])))/2
    for k in winners:
        winners[k] = winners[k] / total
        losers[k] = losers[k] / total
    print("\n策略总结（平均值，基于", total, "局）：")
    print(f"- 胜方平均犯规数更低：{winners['fouls']:.2f} vs 负方 {losers['fouls']:.2f}")
    print(f"- 胜方白球落袋更少：{winners['white_in']:.2f} vs 负方 {losers['white_in']:.2f}")
    print(f"- 胜方首碰犯规更少：{winners['fo_first']:.2f} vs 负方 {losers['fo_first']:.2f}")
    print(f"- 胜方最大连杆更长：{winners['max_streak']:.2f} vs 负方 {losers['max_streak']:.2f}")
    print(f"- 胜方每杆进球率更高：{winners['pockets_per_shot']:.3f} vs 负方 {losers['pockets_per_shot']:.3f}")
