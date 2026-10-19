# -*- coding: utf-8 -*-
"""
虚拟产线仿真器 V3 —— 黑客松"图纸"（产线物理部分）
=====================================================
本文件不需要在学生电脑上运行。它是产线物理规律的权威定义：
你的任务是借助 AI 工具读懂它，并生成行为一致的 HTML 交互端。

产线背景
--------
一台数控磨床连续加工轴类零件，目标外径 20.000 mm，公差 ±0.020 mm。
加工中存在三类扰动：
  1. 刀具磨损：每加工一件，尺寸均值缓慢正向漂移；
  2. 温度漂移：环境温度使均值缓慢随机游走；
  3. 测量噪声：每件测量值带有随机误差。
每 10 件为一批。每批结束后，质检台用最近 30 件的测量数据
计算均值偏移 e 与过程能力指数 CPK。当 CPK 连续 3 批 >= 1.33 时，
判定产线"收敛"；收敛后不停机，继续跑 8 批"保持赛"，
以保持期 CPK 均值论英雄。赛程上限 40 批。
（换刀剧本下，只有统计窗口完全由阶跃后的零件组成时判定才有效
 ——官方窗口口径，混着换刀前旧件的批次不算数。）

闭环策略（死区、增益）在 agent_loop.py 中；本文件定义物理与执行机构。
执行机构有四个关键特性——前三条是硬约束，第四条是本套件的灵魂：
  * 速率限制：单步最多补偿 ±0.006 mm（伺服爬不了更快）；
  * 安全护栏：策略给出的单步补偿超过 ±0.020 mm 直接拦截，转人工审批；
  * 行程上限：累计补偿量不得超过 ±0.150 mm；
  * 执行延迟：本批决定的补偿量，下一批才真正生效（PLC 扫描周期）。
    增益越大越容易震荡——滞后环节放大一切激进。
"""

import random

# ---------------- 产线物理参数（黑客松档案卡会指定修改其中某几行） ----------------
TARGET = 20.000          # 目标尺寸 (mm)
TOL = 0.020              # 公差半宽 (mm)，合格区间 [TARGET-TOL, TARGET+TOL]
BATCH_SIZE = 10          # 每批件数
MAX_BATCHES = 40         # 赛程上限（批数）
NOISE_SIGMA = 0.0035     # 测量噪声标准差 (mm)     <- B 组档案卡修改此行
WEAR_PER_PART = (0.0001, 0.0004)     # 每件刀具磨损量均匀分布区间 (mm)  <- A 组档案卡修改此行
TEMP_PER_PART = (-0.00005, 0.00015)  # 每件温度漂移均匀分布区间 (mm)
INITIAL_WEAR = 0.012     # 开线时刀具已有磨损 (mm)
STEP_SHIFT_BATCH = None  # 在第 N 批开始时换刀产生阶跃偏移，None=不启用  <- C 组档案卡修改此行
STEP_SHIFT_MM = 0.006    # 换刀阶跃偏移量 (mm)

# ---------------- 执行机构硬约束（护栏在机器里，不在提示词里） ----------------
RATE_LIMIT = 0.006       # 速率限制：单步补偿的执行上限 (mm/批)
GUARDRAIL = 0.020        # 安全护栏：策略输出超过此值将被拦截
COMP_CAP = 0.150         # 累计补偿行程上限 (mm)
ACT_DELAY = 1            # 执行延迟：补偿决策多少批后生效

# ---------------- 统计与判定 ----------------
STATS_WINDOW = 30        # CPK 统计窗口（最近 N 件）
CPK_TARGET = 1.33        # 收敛判定阈值
CONSEC_PASS = 3          # 需连续达标的批数
HOLD_BATCHES = 8         # 保持赛：收敛后继续跑的批数


class LineSimulator:
    """虚拟产线：逐件出件，逐批接受闭环调整。"""

    def __init__(self, seed=8):
        self.rng = random.Random(seed)
        self.wear = INITIAL_WEAR
        self.temp = 0.0
        self.compensation = 0.0   # 机床当前补偿量
        self.pending = 0.0        # 已决策但尚未生效的补偿（执行延迟）
        self.measurements = []    # 全部测量值
        self.batch_log = []       # 每批统计记录
        self.part_index = 0
        self.batch_index = 0
        self.adjust_count = 0     # 实际生效的调整次数
        self.guardrail_hits = 0   # 护栏拦截次数

    def _true_mean(self):
        return TARGET + self.wear + self.temp + self.compensation

    def make_part(self):
        """加工一件：先累加扰动，再得到带噪声的测量值。"""
        self.part_index += 1
        self.wear += self.rng.uniform(*WEAR_PER_PART)
        self.temp += self.rng.uniform(*TEMP_PER_PART)
        if STEP_SHIFT_BATCH is not None and \
           self.part_index == (STEP_SHIFT_BATCH - 1) * BATCH_SIZE + 1:
            self.wear += STEP_SHIFT_MM   # 换刀后装夹偏移，表现为均值阶跃
        return self._true_mean() + self.rng.gauss(0, NOISE_SIGMA)

    def stats(self):
        """最近 STATS_WINDOW 件：均值偏移 e、标准差 σ、CPK、超差件数。"""
        w = self.measurements[-STATS_WINDOW:]
        mean = sum(w) / len(w)
        e = mean - TARGET
        var = sum((x - mean) ** 2 for x in w) / (len(w) - 1) if len(w) > 1 else 0.0
        sigma = var ** 0.5
        cpk = (TOL - abs(e)) / (3 * sigma) if sigma > 0 else float("inf")
        oot = sum(1 for x in w if abs(x - TARGET) > TOL)
        return e, sigma, cpk, oot

    def apply_compensation(self, step):
        """
        执行机构：写入一步补偿请求。护栏立即审查，通过后进入延迟队列，
        下一批加工前才真正生效。
        返回状态：
          "queued"    已入队（下批生效）
          "guardrail" 护栏拦截（作废，转人工审批）
        """
        if abs(step) > GUARDRAIL:
            self.guardrail_hits += 1
            return "guardrail"
        self.pending = max(-RATE_LIMIT, min(RATE_LIMIT, step))
        return "queued"

    def run_batch(self):
        """加工一批：先生效上批的补偿（执行延迟），再出件，再统计。"""
        self.batch_index += 1
        if abs(self.pending) > 1e-12:
            new_comp = max(-COMP_CAP, min(COMP_CAP, self.compensation + self.pending))
            self.compensation = new_comp
            self.adjust_count += 1
            self.pending = 0.0
        for _ in range(BATCH_SIZE):
            self.measurements.append(self.make_part())
        e, sigma, cpk, oot = self.stats()
        rec = {
            "batch": self.batch_index,
            "mean_err": round(e, 5),
            "sigma": round(sigma, 5),
            "cpk": round(cpk, 3),
            "oot": oot,
            "compensation": round(self.compensation, 5),
        }
        self.batch_log.append(rec)
        return rec

    def is_converged(self):
        """最近 CONSEC_PASS 批 CPK 是否连续达标。
        换刀剧本（C 组）下，统计窗口必须完全由阶跃后的零件组成，
        判定才有效——官方窗口口径，混着旧件的批次不算数。"""
        if len(self.batch_log) < CONSEC_PASS:
            return False
        if STEP_SHIFT_BATCH is not None:
            min_batch = STEP_SHIFT_BATCH + STATS_WINDOW // BATCH_SIZE - 1
            if self.batch_index < min_batch:
                return False
        return all(r["cpk"] >= CPK_TARGET for r in self.batch_log[-CONSEC_PASS:])

    def total_oot(self):
        """全程累计超差件数。"""
        return sum(1 for x in self.measurements if abs(x - TARGET) > TOL)
