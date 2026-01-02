import math
import pooltool as pt
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
from datetime import datetime
import copy
import random
import time

from .agent import Agent

class NewAgent(Agent):
    """自定义 Agent 模板（待学生实现）"""
    
    def __init__(self):
        super().__init__()
        self.ball_radius = 0.028575
        self.sim_noise = {'V0': 0.1, 'phi': 0.12, 'theta': 0.08, 'a': 0.004, 'b': 0.004}
    
    def decision(self, balls=None, my_targets=None, table=None):
        """决策方法
        
        参数：
            observation: (balls, my_targets, table)
        
        返回：
            dict: {'V0', 'phi', 'theta', 'a', 'b'}
        """
        if balls is None:
            return self._random_action()
        remaining = [bid for bid in my_targets if balls[bid].state.s != 4]
        if len(remaining) == 0:
            my_targets = ['8']
        last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        candidates = self._generate_candidates(balls, my_targets, table)
        start_time = time.perf_counter()
        budget = 10
        prelim_scores = []
        for action in candidates:
            agg = 0.0
            for _ in range(1):
                shot = self._simulate_action(balls, table, action)
                agg += -500.0 if shot is None else self._analyze_shot_for_reward(shot, last_state_snapshot, my_targets)
            prelim_scores.append(agg / 1.0)
            if time.perf_counter() - start_time > budget:
                break
        k = min(4, len(prelim_scores))
        top_indices = np.argsort(prelim_scores)[-k:]
        best_action = None
        best_score = -1e9
        best_robust = -1e9
        for idx in top_indices:
            action = candidates[int(idx)]
            agg = 0.0
            contact = 0
            cue_pocket = 0
            sims = 4
            count = 0
            viability_sum = 0.0
            opp_viability_sum = 0.0
            risk_sum = 0.0
            for _ in range(sims):
                shot = self._simulate_action(balls, table, action)
                if shot is None:
                    agg += -500.0
                else:
                    agg += self._analyze_shot_for_reward(shot, last_state_snapshot, my_targets)
                    ids_first = None
                    valid_ids = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'}
                    for e in shot.events:
                        et = str(e.event_type).lower()
                        ids = list(e.ids) if hasattr(e, 'ids') else []
                        if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
                            others = [i for i in ids if i != 'cue' and i in valid_ids]
                            if others:
                                ids_first = others[0]
                                break
                    if ids_first in my_targets:
                        contact += 1
                    new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state_snapshot[bid].state.s != 4]
                    if 'cue' in new_pocketed:
                        cue_pocket += 1
                    viability_sum += self._next_shot_viability_rate(shot, my_targets)
                    opp_viability_sum += self._opponent_viability_rate(shot, my_targets)
                    if ('cue' in shot.balls) and (shot.balls['cue'].state.s != 4):
                        risk_sum += self._white_ball_risk(shot.balls['cue'].state.rvw[0], shot.table)
                count += 1
                if time.perf_counter() - start_time > budget:
                    break
            avg = agg / max(1, count)
            if my_targets == ['8']:
                fc_w, cue_w, next_w, opp_w, risk_w = 20.0, 80.0, 4.0, 14.0, 12.0
            else:
                fc_w, cue_w, next_w, opp_w, risk_w = 26.0, 60.0, 8.0, 10.0, 8.0
            robust = (
                avg
                + fc_w * (contact / max(1, count))
                - cue_w * (cue_pocket / max(1, count))
                + next_w * (viability_sum / max(1, count))
                - opp_w * (opp_viability_sum / max(1, count))
                - risk_w * (risk_sum / max(1, count))
            )
            if robust > best_robust:
                best_robust = robust
                best_score = avg
                best_action = action
        if best_action is not None:
            if time.perf_counter() - start_time < budget * 0.6:
                best_action = self._local_refine(balls, table, best_action, last_state_snapshot, my_targets)
        if best_action is None or best_score < 10:
            if time.perf_counter() - start_time > budget:
                if best_action is None:
                    return self._random_action()
                return {
                    'V0': float(np.clip(best_action['V0'], 0.5, 8.0)),
                    'phi': float(best_action['phi'] % 360),
                    'theta': float(np.clip(best_action['theta'], 0, 90)),
                    'a': float(np.clip(best_action['a'], -0.5, 0.5)),
                    'b': float(np.clip(best_action['b'], -0.5, 0.5))
                }
            safety_actions = self._generate_safety_candidates(balls, table)
            if len(safety_actions) > 0:
                best_safe = None
                best_safe_score = -1e9
                for sa in safety_actions:
                    agg = 0.0
                    sims = 3
                    contact = 0
                    cue_pocket = 0
                    count = 0
                    viability_sum = 0.0
                    opp_viability_sum = 0.0
                    risk_sum = 0.0
                    for _ in range(sims):
                        shot = self._simulate_action(balls, table, sa)
                        if shot is None:
                            agg += -500.0
                        else:
                            agg += self._analyze_shot_for_reward(shot, last_state_snapshot, my_targets)
                            ids_first = None
                            valid_ids = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'}
                            for e in shot.events:
                                et = str(e.event_type).lower()
                                ids = list(e.ids) if hasattr(e, 'ids') else []
                                if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
                                    others = [i for i in ids if i != 'cue' and i in valid_ids]
                                    if others:
                                        ids_first = others[0]
                                        break
                            if ids_first in my_targets:
                                contact += 1
                            new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state_snapshot[bid].state.s != 4]
                            if 'cue' in new_pocketed:
                                cue_pocket += 1
                            viability_sum += self._next_shot_viability_rate(shot, my_targets)
                            opp_viability_sum += self._opponent_viability_rate(shot, my_targets)
                            if ('cue' in shot.balls) and (shot.balls['cue'].state.s != 4):
                                risk_sum += self._white_ball_risk(shot.balls['cue'].state.rvw[0], shot.table)
                        count += 1
                        if time.perf_counter() - start_time > budget:
                            break
                    avg = agg / max(1, count)
                    if my_targets == ['8']:
                        fc_w, cue_w, next_w, opp_w, risk_w = 20.0, 80.0, 4.0, 14.0, 12.0
                    else:
                        fc_w, cue_w, next_w, opp_w, risk_w = 26.0, 60.0, 8.0, 10.0, 8.0
                    robust = (
                        avg
                        + fc_w * (contact / max(1, count))
                        - cue_w * (cue_pocket / max(1, count))
                        + next_w * (viability_sum / max(1, count))
                        - opp_w * (opp_viability_sum / max(1, count))
                        - risk_w * (risk_sum / max(1, count))
                    )
                    if robust > best_safe_score:
                        best_safe_score = robust
                        best_safe = sa
                if best_safe is not None:
                    best_action = best_safe
            if best_action is None:
                return self._random_action()
        return {
            'V0': float(np.clip(best_action['V0'], 0.5, 8.0)),
            'phi': float(best_action['phi'] % 360),
            'theta': float(np.clip(best_action['theta'], 0, 90)),
            'a': float(np.clip(best_action['a'], -0.5, 0.5)),
            'b': float(np.clip(best_action['b'], -0.5, 0.5))
        }

    def _calc_angle(self, v):
        a = math.degrees(math.atan2(v[1], v[0]))
        return a % 360

    def _ghost_target(self, cue_pos, obj_pos, pocket_pos):
        vec_obj_to_pocket = np.array(pocket_pos) - np.array(obj_pos)
        d = np.linalg.norm(vec_obj_to_pocket)
        if d == 0:
            return 0.0, 0.0, np.array(obj_pos)
        unit = vec_obj_to_pocket / d
        ghost_pos = np.array(obj_pos) - unit * (2 * self.ball_radius)
        vec_cue_to_ghost = ghost_pos - np.array(cue_pos)
        dist = np.linalg.norm(vec_cue_to_ghost)
        phi = self._calc_angle(vec_cue_to_ghost)
        return phi, dist, ghost_pos

    def _has_clear_path(self, cue_pos, ghost_pos, balls, ignore_ids):
        p = np.array(cue_pos[:2])
        q = np.array(ghost_pos[:2])
        v = q - p
        vv = np.dot(v, v)
        for bid, ball in balls.items():
            if bid in ignore_ids:
                continue
            c = np.array(ball.state.rvw[0][:2])
            w = c - p
            if vv == 0:
                continue
            t = np.clip(np.dot(w, v) / vv, 0.0, 1.0)
            proj = p + t * v
            if np.linalg.norm(proj - c) <= self.ball_radius * 1.25:
                return False
        return True

    def _has_clear_path_obj_to_pocket(self, obj_pos, pocket_pos, balls, ignore_ids):
        p = np.array(obj_pos[:2])
        q = np.array(pocket_pos[:2])
        v = q - p
        vv = np.dot(v, v)
        for bid, ball in balls.items():
            if bid in ignore_ids:
                continue
            c = np.array(ball.state.rvw[0][:2])
            w = c - p
            if vv == 0:
                continue
            t = np.clip(np.dot(w, v) / vv, 0.0, 1.0)
            proj = p + t * v
            if np.linalg.norm(proj - c) <= self.ball_radius * 1.25:
                return False
        return True

    def _first_ball_along_phi(self, cue_pos, phi, balls, ignore_ids):
        p = np.array(cue_pos[:2])
        ang = math.radians(phi % 360)
        v = np.array([math.cos(ang), math.sin(ang)])
        best_id = None
        best_t = 1e9
        for bid, ball in balls.items():
            if bid in ignore_ids:
                continue
            c = np.array(ball.state.rvw[0][:2])
            w = c - p
            t = np.dot(w, v)
            if t <= 0:
                continue
            perp = np.linalg.norm(w - t * v)
            if perp <= self.ball_radius * 1.05 and t < best_t:
                best_t = t
                best_id = bid
        return best_id

    def _is_first_contact_consistent(self, cue_pos, phi, target_id, balls, ignore_ids):
        ok = 0
        total = 0
        for d in [0.0, 0.3, -0.3, 0.15, -0.15]:
            total += 1
            first_id = self._first_ball_along_phi(cue_pos, (phi + d) % 360, balls, ignore_ids)
            if first_id == target_id:
                ok += 1
        return ok >= 3

    def _next_shot_viability_rate(self, shot, player_targets):
        try:
            balls = shot.balls
            table = shot.table
            if 'cue' not in balls:
                return 0.0
            cue_pos = balls['cue'].state.rvw[0]
            remaining = [bid for bid in player_targets if bid in balls and balls[bid].state.s != 4]
            if len(remaining) == 0:
                remaining = ['8']
            dlist = []
            for bid in remaining:
                if bid not in balls:
                    continue
                d = np.linalg.norm(np.array(balls[bid].state.rvw[0][:2]) - np.array(cue_pos[:2]))
                dlist.append((bid, d))
            dlist.sort(key=lambda x: x[1])
            targets = [bid for bid, _ in dlist[:3]]
            pocket_items = list(table.pockets.items())
            total = 0
            viable = 0
            for tid in targets:
                obj_pos = balls[tid].state.rvw[0]
                pockets_sorted = sorted(pocket_items, key=lambda it: np.linalg.norm(np.array(it[1].center[:2]) - np.array(obj_pos[:2])))
                for _, pocket in pockets_sorted[:2]:
                    total += 1
                    phi_ideal, _, ghost_pos = self._ghost_target(cue_pos, obj_pos, pocket.center)
                    if not self._has_clear_path(cue_pos, ghost_pos, balls, ignore_ids={'cue', tid}):
                        continue
                    if not self._has_clear_path_obj_to_pocket(obj_pos, pocket.center, balls, ignore_ids={'cue', tid}):
                        continue
                    viable += 1
            return float(viable) / float(max(1, total))
        except Exception:
            return 0.0

    def _opponent_viability_rate(self, shot, player_targets):
        try:
            balls = shot.balls
            table = shot.table
            if 'cue' not in balls:
                return 0.0
            cue_pos = balls['cue'].state.rvw[0]
            valid_ids = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'}
            opp_remaining = []
            for bid, b in balls.items():
                if (bid in valid_ids) and (b.state.s != 4) and (bid not in player_targets) and (bid != '8'):
                    opp_remaining.append(bid)
            if len(opp_remaining) == 0:
                if ('8' in balls) and (balls['8'].state.s != 4):
                    opp_remaining = ['8']
            dlist = []
            for bid in opp_remaining:
                d = np.linalg.norm(np.array(balls[bid].state.rvw[0][:2]) - np.array(cue_pos[:2]))
                dlist.append((bid, d))
            dlist.sort(key=lambda x: x[1])
            targets = [bid for bid, _ in dlist[:3]]
            pocket_items = list(table.pockets.items())
            total = 0
            viable = 0
            for tid in targets:
                obj_pos = balls[tid].state.rvw[0]
                pockets_sorted = sorted(pocket_items, key=lambda it: np.linalg.norm(np.array(it[1].center[:2]) - np.array(obj_pos[:2])))
                for _, pocket in pockets_sorted[:2]:
                    total += 1
                    phi_ideal, _, ghost_pos = self._ghost_target(cue_pos, obj_pos, pocket.center)
                    if not self._has_clear_path(cue_pos, ghost_pos, balls, ignore_ids={'cue', tid}):
                        continue
                    if not self._has_clear_path_obj_to_pocket(obj_pos, pocket.center, balls, ignore_ids={'cue', tid}):
                        continue
                    viable += 1
            return float(viable) / float(max(1, total))
        except Exception:
            return 0.0

    def _white_ball_risk(self, cue_pos, table):
        try:
            p = np.array(cue_pos[:2])
            dmin = 1e9
            for _, pocket in table.pockets.items():
                c = np.array(pocket.center[:2])
                d = float(np.linalg.norm(c - p))
                if d < dmin:
                    dmin = d
            scale = 0.25
            return float(math.exp(-dmin / max(scale, 1e-6)))
        except Exception:
            return 0.0

    def _generate_candidates(self, balls, my_targets, table):
        actions = []
        cue_ball = balls.get('cue')
        if cue_ball is None:
            return actions
        cue_pos = cue_ball.state.rvw[0]
        target_ids_all = [bid for bid in my_targets if balls[bid].state.s != 4]
        target_centroid = None
        if len(target_ids_all) > 0:
            tps = np.array([balls[bid].state.rvw[0][:2] for bid in target_ids_all])
            target_centroid = np.mean(tps, axis=0)
        # 仅选择距离白球最近的若干目标，提高稳健性
        if len(target_ids_all) == 0:
            target_ids = ['8']
        else:
            dists = []
            for bid in target_ids_all:
                d = np.linalg.norm(balls[bid].state.rvw[0][:2] - cue_pos[:2])
                dists.append((bid, d))
            dists.sort(key=lambda x: x[1])
            target_ids = [bid for bid, _ in dists[:4]]
        pocket_items = list(table.pockets.items())
        for tid in target_ids:
            obj = balls.get(tid)
            if obj is None:
                continue
            obj_pos = obj.state.rvw[0]
            pockets_sorted = sorted(pocket_items, key=lambda it: np.linalg.norm(np.array(it[1].center[:2]) - np.array(obj_pos[:2])))
            for _, pocket in pockets_sorted[:2]:
                pocket_pos = pocket.center
                phi_ideal, dist, ghost_pos = self._ghost_target(cue_pos, obj_pos, pocket_pos)
                if not self._has_clear_path(cue_pos, ghost_pos, balls, ignore_ids={'cue', tid}):
                    continue
                if not self._has_clear_path_obj_to_pocket(obj_pos, pocket_pos, balls, ignore_ids={'cue', tid}):
                    continue
                first_id_ideal = self._first_ball_along_phi(cue_pos, phi_ideal, balls, ignore_ids={'cue'})
                if first_id_ideal != tid:
                    continue
                if not self._is_first_contact_consistent(cue_pos, phi_ideal, tid, balls, ignore_ids={'cue'}):
                    continue
                v_base = 1.2 + dist * 1.2
                v_base = float(np.clip(v_base, 1.0, 5.8))
                if dist < 0.35:
                    v_base = float(max(1.0, v_base * 0.92))
                pos_a = 0.0
                if target_centroid is not None:
                    shot_dir = np.array([math.cos(math.radians(phi_ideal)), math.sin(math.radians(phi_ideal))])
                    vec_to_centroid = target_centroid - np.array(obj_pos[:2])
                    cross = shot_dir[0] * vec_to_centroid[1] - shot_dir[1] * vec_to_centroid[0]
                    pos_a = 0.03 if cross > 0 else -0.03
                actions.append({'V0': v_base, 'phi': phi_ideal, 'theta': 0.0, 'a': 0.0, 'b': 0.0})
                actions.append({'V0': min(v_base + 0.6, 6.0), 'phi': phi_ideal, 'theta': 0.0, 'a': 0.0, 'b': -0.02})
                actions.append({'V0': v_base, 'phi': (phi_ideal + 0.5) % 360, 'theta': 0.0, 'a': 0.02, 'b': -0.02})
                actions.append({'V0': v_base, 'phi': (phi_ideal - 0.5) % 360, 'theta': 0.0, 'a': -0.02, 'b': -0.02})
                actions.append({'V0': v_base, 'phi': (phi_ideal + 1.0) % 360, 'theta': 0.0, 'a': pos_a, 'b': -0.02})
                actions.append({'V0': v_base, 'phi': (phi_ideal - 1.0) % 360, 'theta': 0.0, 'a': -pos_a, 'b': -0.02})
                phi_center = self._calc_angle(np.array(obj_pos[:2]) - np.array(cue_pos[:2]))
                dist_center = np.linalg.norm(np.array(obj_pos[:2]) - np.array(cue_pos[:2]))
                if self._has_clear_path(cue_pos, obj_pos, balls, ignore_ids={'cue', tid}):
                    first_id_center = self._first_ball_along_phi(cue_pos, phi_center, balls, ignore_ids={'cue'})
                    if first_id_center == tid and self._is_first_contact_consistent(cue_pos, phi_center, tid, balls, ignore_ids={'cue'}):
                        v_center = float(np.clip(1.1 + dist_center * 1.1, 1.0, 6.0))
                        actions.append({'V0': v_center, 'phi': phi_center, 'theta': 0.0, 'a': 0.0, 'b': -0.02})
                        actions.append({'V0': v_center, 'phi': (phi_center + 0.8) % 360, 'theta': 0.0, 'a': 0.01, 'b': -0.02})
                        actions.append({'V0': v_center, 'phi': (phi_center - 0.8) % 360, 'theta': 0.0, 'a': -0.01, 'b': -0.02})
                if dist_center > self.ball_radius * 1.2:
                    try:
                        delta = math.degrees(math.asin(min(0.95, self.ball_radius / max(dist_center, 1e-6))))
                    except Exception:
                        delta = 0.0
                    for dphi in [0.0, delta, -delta, 2*delta, -2*delta]:
                        phi_tan = (phi_center + dphi) % 360
                        first_id = self._first_ball_along_phi(cue_pos, phi_tan, balls, ignore_ids={'cue'})
                        if first_id == tid and self._is_first_contact_consistent(cue_pos, phi_tan, tid, balls, ignore_ids={'cue'}):
                            v_tan = float(np.clip(1.0 + dist_center * 1.0, 0.8, 5.5))
                            actions.append({'V0': v_tan, 'phi': phi_tan, 'theta': 0.0, 'a': 0.0, 'b': -0.02})
        if len(actions) == 0:
            cp = np.array(cue_pos)
            target_ids_scan = target_ids if len(target_ids) > 0 else [tid]
            for sid in target_ids_scan:
                if sid not in balls:
                    continue
                sp = np.array(balls[sid].state.rvw[0][:2])
                dist_s = np.linalg.norm(sp - cp[:2])
                phi_s = self._calc_angle(sp - cp[:2])
                for dphi in [0.0, 0.6, -0.6, 1.2, -1.2, 2.0, -2.0]:
                    phi_try = (phi_s + dphi) % 360
                    first_id = self._first_ball_along_phi(cue_pos, phi_try, balls, ignore_ids={'cue'})
                    if first_id == sid and self._is_first_contact_consistent(cue_pos, phi_try, sid, balls, ignore_ids={'cue'}):
                        v_s = float(np.clip(1.1 + dist_s * 0.9, 0.8, 5.0))
                        actions.append({'V0': v_s, 'phi': phi_try, 'theta': 0.0, 'a': 0.0, 'b': -0.02})
            if len(actions) == 0:
                cp = np.array(cue_pos)
                xs = [0.0, table.l]
                ys = [0.0, table.w]
                targets = [np.array([xs[0], cp[1], cp[2]]), np.array([xs[1], cp[1], cp[2]]),
                           np.array([cp[0], ys[0], cp[2]]), np.array([cp[0], ys[1], cp[2]])]
                best = None
                bestd = 1e9
                for t in targets:
                    d = np.linalg.norm(t[:2] - cp[:2])
                    if d < bestd:
                        bestd = d
                        best = t
                v_base = float(np.clip(1.2 + bestd * 0.8, 0.8, 5.0))
                phi_ideal = self._calc_angle(best[:2] - cp[:2])
                actions.append({'V0': v_base, 'phi': phi_ideal, 'theta': 0.0, 'a': 0.0, 'b': -0.02})
                actions.append({'V0': min(v_base + 0.8, 6.0), 'phi': phi_ideal, 'theta': 0.0, 'a': 0.0, 'b': -0.02})
                actions.append({'V0': v_base, 'phi': (phi_ideal + 0.7) % 360, 'theta': 0.0, 'a': 0.02, 'b': -0.02})
        random.shuffle(actions)
        return actions[:12]

    def _generate_safety_candidates(self, balls, table):
        actions = []
        cue_ball = balls.get('cue')
        if cue_ball is None:
            return actions
        cue_pos = cue_ball.state.rvw[0]
        cp = np.array(cue_pos)
        target_ids = []
        for bid, ball in balls.items():
            if bid == 'cue' or ball.state.s == 4:
                continue
            target_ids.append(bid)
        target_ids_sorted = sorted(target_ids, key=lambda bid: np.linalg.norm(np.array(balls[bid].state.rvw[0][:2]) - cp[:2]))
        for sid in target_ids_sorted[:2]:
            sp = np.array(balls[sid].state.rvw[0][:2])
            dist_s = np.linalg.norm(sp - cp[:2])
            phi_s = self._calc_angle(sp - cp[:2])
            for dphi in [0.0, 0.6, -0.6]:
                phi_try = (phi_s + dphi) % 360
                first_id = self._first_ball_along_phi(cue_pos, phi_try, balls, ignore_ids={'cue'})
                if first_id == sid:
                    v_s = float(np.clip(1.0 + dist_s * 0.9, 0.8, 5.0))
                    actions.append({'V0': v_s, 'phi': phi_try, 'theta': 0.0, 'a': 0.0, 'b': -0.02})
        if len(actions) == 0:
            xs = [0.0, table.l]
            ys = [0.0, table.w]
            targets = [np.array([xs[0], cp[1], cp[2]]), np.array([xs[1], cp[1], cp[2]]),
                       np.array([cp[0], ys[0], cp[2]]), np.array([cp[0], ys[1], cp[2]])]
            best = None
            bestd = 1e9
            for t in targets:
                d = np.linalg.norm(t[:2] - cp[:2])
                if d < bestd:
                    bestd = d
                    best = t
            v_base = float(np.clip(1.2 + bestd * 0.8, 0.8, 5.0))
            phi_ideal = self._calc_angle(best[:2] - cp[:2])
            actions.append({'V0': v_base, 'phi': phi_ideal, 'theta': 0.0, 'a': 0.0, 'b': -0.02})
        return actions

    def _simulate_action(self, balls, table, action):
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")
        shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
        try:
            V0 = float(np.clip(action['V0'] + np.random.normal(0, self.sim_noise['V0']), 0.5, 8.0))
            phi = float((action['phi'] + np.random.normal(0, self.sim_noise['phi'])) % 360)
            theta = float(np.clip(action['theta'] + np.random.normal(0, self.sim_noise['theta']), 0, 90))
            a = float(np.clip(action['a'] + np.random.normal(0, self.sim_noise['a']), -0.5, 0.5))
            b = float(np.clip(action['b'] + np.random.normal(0, self.sim_noise['b']), -0.5, 0.5))
            cue.set_state(V0=V0, phi=phi, theta=theta, a=a, b=b)
            pt.simulate(shot, inplace=True)
            return shot
        except Exception:
            return None

    def _local_refine(self, balls, table, action, last_state_snapshot, my_targets):
        base = {'V0': float(action['V0']), 'phi': float(action['phi']), 'theta': float(action['theta']), 'a': float(action['a']), 'b': float(action['b'])}
        variants = [
            base,
            {'V0': float(np.clip(base['V0'] + 0.2, 0.5, 8.0)), 'phi': base['phi'], 'theta': base['theta'], 'a': base['a'], 'b': base['b']},
            {'V0': base['V0'], 'phi': float((base['phi'] + 0.25) % 360), 'theta': base['theta'], 'a': base['a'], 'b': base['b']},
            {'V0': base['V0'], 'phi': float((base['phi'] - 0.25) % 360), 'theta': base['theta'], 'a': base['a'], 'b': base['b']},
            {'V0': base['V0'], 'phi': base['phi'], 'theta': base['theta'], 'a': base['a'], 'b': float(np.clip(-0.03, -0.5, 0.5))}
        ]
        best = base
        best_robust = -1e9
        for v in variants:
            agg = 0.0
            contact = 0
            cue_pocket = 0
            sims = 2
            viability_sum = 0.0
            opp_viability_sum = 0.0
            risk_sum = 0.0
            for _ in range(sims):
                shot = self._simulate_action(balls, table, v)
                if shot is None:
                    agg += -500.0
                else:
                    agg += self._analyze_shot_for_reward(shot, last_state_snapshot, my_targets)
                    ids_first = None
                    valid_ids = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'}
                    for e in shot.events:
                        et = str(e.event_type).lower()
                        ids = list(e.ids) if hasattr(e, 'ids') else []
                        if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
                            others = [i for i in ids if i != 'cue' and i in valid_ids]
                            if others:
                                ids_first = others[0]
                                break
                    if ids_first in my_targets:
                        contact += 1
                    new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state_snapshot[bid].state.s != 4]
                    if 'cue' in new_pocketed:
                        cue_pocket += 1
                    viability_sum += self._next_shot_viability_rate(shot, my_targets)
                    opp_viability_sum += self._opponent_viability_rate(shot, my_targets)
                    if ('cue' in shot.balls) and (shot.balls['cue'].state.s != 4):
                        risk_sum += self._white_ball_risk(shot.balls['cue'].state.rvw[0], shot.table)
            avg = agg / sims
            if my_targets == ['8']:
                fc_w, cue_w, next_w, opp_w, risk_w = 20.0, 80.0, 4.0, 14.0, 12.0
            else:
                fc_w, cue_w, next_w, opp_w, risk_w = 26.0, 60.0, 8.0, 10.0, 8.0
            robust = (
                avg
                + fc_w * (contact / sims)
                - cue_w * (cue_pocket / sims)
                + next_w * (viability_sum / sims)
                - opp_w * (opp_viability_sum / sims)
                - risk_w * (risk_sum / sims)
            )
            if robust > best_robust:
                best_robust = robust
                best = v
        return best

    def _random_action(self):
        V0 = round(random.uniform(0.8, 5.0), 2)
        phi = round(random.uniform(0, 360), 2)
        theta = round(random.uniform(0, 5.0), 2)
        a = round(random.uniform(-0.08, 0.08), 3)
        b = round(random.uniform(-0.10, 0.02), 3)
        return {'V0': V0, 'phi': phi, 'theta': theta, 'a': a, 'b': b}

    def _analyze_shot_for_reward(self, shot: pt.System, last_state: dict, player_targets: list):
        new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
        own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
        enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]
        cue_pocketed = "cue" in new_pocketed
        eight_pocketed = "8" in new_pocketed
        first_contact_ball_id = None
        valid_ball_ids = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'}
        for e in shot.events:
            et = str(e.event_type).lower()
            ids = list(e.ids) if hasattr(e, 'ids') else []
            if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
                other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
                if other_ids:
                    first_contact_ball_id = other_ids[0]
                    break
        cue_hit_cushion = False
        target_hit_cushion = False
        for e in shot.events:
            et = str(e.event_type).lower()
            ids = list(e.ids) if hasattr(e, 'ids') else []
            if 'cushion' in et:
                if 'cue' in ids:
                    cue_hit_cushion = True
                if first_contact_ball_id is not None and first_contact_ball_id in ids:
                    target_hit_cushion = True
        foul_first_hit = False
        did_no_hit = False
        if first_contact_ball_id is None:
            if len(last_state) > 2 or player_targets != ['8']:
                foul_first_hit = True
                did_no_hit = True
        else:
            if first_contact_ball_id not in player_targets:
                foul_first_hit = True
        foul_no_rail = False
        if len(new_pocketed) == 0 and first_contact_ball_id is not None and (not cue_hit_cushion) and (not target_hit_cushion):
            foul_no_rail = True
        score = 0.0
        if cue_pocketed and eight_pocketed:
            score -= 150.0
        elif cue_pocketed:
            score -= 100.0
        elif eight_pocketed:
            if player_targets == ['8']:
                score += 100.0
            else:
                score -= 150.0
        if foul_first_hit:
            score -= 30.0
        if did_no_hit:
            score -= 70.0
        if foul_no_rail:
            score -= 30.0
        score += float(len(own_pocketed)) * 50.0
        score -= float(len(enemy_pocketed)) * 20.0
        if score == 0.0 and not cue_pocketed and not eight_pocketed and not foul_first_hit and not foul_no_rail:
            score = 10.0
        return score
