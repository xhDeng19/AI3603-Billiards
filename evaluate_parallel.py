"""
evaluate_parallel.py - 并行化 Agent 评估脚本

功能：
- 使用多进程并行运行多局对战，显著加快测试速度
- 保持与 evaluate.py 一致的逻辑和参数
- 自动统计胜负和得分
"""

import time
import collections
import multiprocessing
import sys
import os

# 添加 pooltool 路径到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pooltool')))

from utils import set_random_seed
from poolenv import PoolEnv
from agents import BasicAgent, BasicAgentPro, NewAgent

def run_single_game(game_idx, agent_a_cls, agent_b_cls):
    """
    运行单局比赛
    :param game_idx: 比赛编号
    :param agent_a_cls: Agent A 的类
    :param agent_b_cls: Agent B 的类
    :return: (winner, reason, agent_a_name, agent_b_name, game_idx)
    """
    # 每个进程设置不同的随机种子，确保随机性
    # 使用 game_idx + 固定偏移量作为种子，保证可复现性
    set_random_seed(enable=False, seed=42 + game_idx)
    
    env = PoolEnv()
    
    # 实例化 Agent
    agent_a = agent_a_cls()
    agent_b = agent_b_cls()
    
    players = [agent_a, agent_b]
    target_ball_choice = ['solid', 'solid', 'stripe', 'stripe']
    
    # 确定本局参数
    target_ball = target_ball_choice[game_idx % 4]
    
    # 重置环境
    env.reset(target_ball=target_ball)
    
    while True:
        player = env.get_curr_player()
        obs = env.get_observation(player)
        
        if player == 'A':
            action = players[game_idx % 2].decision(*obs)
        else:
            action = players[(game_idx + 1) % 2].decision(*obs)
            
        env.take_shot(action)
        done, info = env.get_done()
        
        if done:
            return (info['winner'], info.get('reason', 'UNKNOWN'), 
                    agent_a.__class__.__name__, agent_b.__class__.__name__, game_idx)

def main():
    # 参数设置
    n_games = 120  # 对战局数
    n_processes = min(multiprocessing.cpu_count(), n_games) # 使用 CPU 核心数
    
    agent_a_cls = BasicAgent
    agent_b_cls = NewAgent
    
    print(f"开始并行评估: {agent_a_cls.__name__} vs {agent_b_cls.__name__}")
    print(f"总局数: {n_games}, 并行进程数: {n_processes}")
    
    start_time = time.time()
    
    # 准备参数
    tasks = [(i, agent_a_cls, agent_b_cls) for i in range(n_games)]
    
    # 并行执行
    with multiprocessing.Pool(processes=n_processes) as pool:
        results_list = pool.starmap(run_single_game, tasks)
        
    end_time = time.time()
    
    # 统计结果
    results = {'AGENT_A_WIN': 0, 'AGENT_B_WIN': 0, 'SAME': 0}
    loss_reasons = collections.defaultdict(lambda: collections.defaultdict(int))
    
    for winner, reason, agent_a_name, agent_b_name, i in results_list:
        # 记录胜负
        if winner == 'SAME':
            results['SAME'] += 1
        elif winner == 'A':
            results[['AGENT_A_WIN', 'AGENT_B_WIN'][i % 2]] += 1
        else:
            results[['AGENT_A_WIN', 'AGENT_B_WIN'][(i+1) % 2]] += 1
            
        # 记录详细原因
        current_player_a_is_agent_a = (i % 2 == 0)
        
        winner_agent_name = "DRAW"
        if winner == 'A':
            winner_agent_name = agent_a_name if current_player_a_is_agent_a else agent_b_name
        elif winner == 'B':
            winner_agent_name = agent_b_name if current_player_a_is_agent_a else agent_a_name
            
        loss_reasons[winner_agent_name][reason] += 1
        if winner != 'SAME':
            loser_agent_name = agent_b_name if winner_agent_name == agent_a_name else agent_a_name
            loss_reasons[loser_agent_name][f"LOST_BY_{reason}"] += 1

    # 计算分数
    results['AGENT_A_SCORE'] = results['AGENT_A_WIN'] * 1 + results['SAME'] * 0.5
    results['AGENT_B_SCORE'] = results['AGENT_B_WIN'] * 1 + results['SAME'] * 0.5

    print(f"\n评估完成！耗时: {end_time - start_time:.2f} 秒")
    print("\n最终结果：", results)
    print("\n详细结束原因统计:")
    for agent_name, reasons in loss_reasons.items():
        print(f"\n{agent_name}:")
        for reason, count in reasons.items():
            print(f"  {reason}: {count}")

if __name__ == '__main__':
    # Windows 下必须在 if __name__ == '__main__': 块中调用 multiprocessing
    multiprocessing.freeze_support()
    main()
