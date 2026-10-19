# -*- coding: utf-8 -*-
"""
闭环策略 V3 —— 黑客松"图纸"（你的策略部分）
=====================================================
本文件是环路上的"决策大脑"：每批结束后读质检数据，决定补偿多少。
产线物理在 line_simulator.py 中；本文件是你调优的战场。

基线策略（保守的带死区比例控制）：
    |e| < DEADZONE  →  不动（不与噪声搏斗）
    否则            →  补偿 -e * GAIN（把均值拉回目标）
基线故意调得保守（GAIN = 0.5）：能收敛，但慢、保持期 CPK 勉强。
你的任务是找到更好的组合——注意执行延迟会放大激进增益的震荡。

成绩口径（黑客松统一）：
    收敛批次（越小越好） + 保持期 CPK 均值（越大越好） + 全程超差件数（越少越好）
"""

from line_simulator import (
    LineSimulator, CPK_TARGET, CONSEC_PASS, HOLD_BATCHES, MAX_BATCHES,
)

# ---------------- 闭环控制参数（黑客松调优的"工程旋钮"） ----------------
DEADZONE = 0.002         # 死区：|e| 小于此值不调整
GAIN = 0.5               # 增益：补偿步长 = -e * GAIN（基线故意保守）
COMPETITION_SEED = 8     # 竞赛统一种子：同一张考卷，比的是答题方法


def decide(e):
    """根据均值偏移 e 给出本批补偿步长。"""
    if abs(e) < DEADZONE:
        return 0.0
    return -e * GAIN


def run_closed_loop(seed=COMPETITION_SEED, verbose=True):
    """闭环主循环：出批 -> 评估 -> 决策 -> 执行（下批生效）。
    收敛后进入保持赛：继续跑 HOLD_BATCHES 批，统计保持期 CPK 均值。"""
    sim = LineSimulator(seed=seed)
    conv_batch = None
    hold_cpk = []
    while sim.batch_index < MAX_BATCHES:
        rec = sim.run_batch()
        step = decide(rec["mean_err"])
        status = sim.apply_compensation(step)
        rec["step"] = round(step, 5)
        rec["decision"] = status
        if verbose:
            tag = " ⚠护栏拦截" if status == "guardrail" else ""
            phase = "" if conv_batch is None else " [保持赛]"
            print(f"批 {rec['batch']:>2} | e={rec['mean_err']:+.4f} | "
                  f"σ={rec['sigma']:.4f} | CPK={rec['cpk']:.2f} | "
                  f"超差={rec['oot']} | 补偿={rec['compensation']:+.4f}{tag}{phase}")
        if conv_batch is None and sim.is_converged():
            conv_batch = sim.batch_index
            if verbose:
                print(f"── ✅ 第 {conv_batch} 批收敛，进入保持赛 ──")
        elif conv_batch is not None:
            hold_cpk.append(rec["cpk"])
            if len(hold_cpk) >= HOLD_BATCHES:
                break
    if conv_batch is None:
        if verbose:
            print(f"\n❌ 未收敛：{MAX_BATCHES} 批内未达到连续 {CONSEC_PASS} 批 CPK ≥ {CPK_TARGET}")
        return sim, None
    hold_mean = sum(hold_cpk) / len(hold_cpk) if hold_cpk else 0.0
    if verbose:
        print(f"\n成绩：第 {conv_batch} 批收敛 | 保持期 CPK 均值 {hold_mean:.2f} | "
              f"全程超差 {sim.total_oot()} 件 | 调整 {sim.adjust_count} 次")
    return sim, {"conv_batch": conv_batch, "hold_cpk": round(hold_mean, 3),
                 "oot": sim.total_oot(), "adjusts": sim.adjust_count}


if __name__ == "__main__":
    run_closed_loop()
