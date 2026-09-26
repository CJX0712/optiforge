# OptiForge 架构文档

> 版本 0.1.0 · 作者 晨星 · 2026-09-27

## 1. 领域与选型

组合优化（Combinatorial Optimization）：在离散解空间中找目标最优解。
三个经典问题族：**TSP**（路径）、**0-1 背包**（子集选择）、**分配问题**（二分图完美匹配）。

| 层 | 技术 | 版本 | 选型理由 |
|---|---|---|---|
| SOTA 求解 | Google OR-Tools（CP-SAT + Routing） | 9.15.6755 | 工业级精确/元启发求解器，禁自研 SOTA 原则下最优复用 |
| 兜底求解 | numpy 2.5.3 + scipy 1.18.1 | pinned | 零下载可跑；scipy Hungarian 本身即精确算法 |
| HPO | Optuna 5.0.0（TPE） | pinned | 业界标准贝叶斯调参 |
| 测试 | pytest 9.1.1 | pinned | - |

## 2. 模块分层（调用单向无环）

```
optiforge/
├── core/                  # 零依赖基座
│   ├── types.py           # Problem / Solution dataclass（frozen）
│   ├── errors.py          # E100~E500 分段错误码
│   ├── config.py          # ENV_XXX_* 覆盖 → Config
│   └── interfaces.py      # Solver Protocol（supports + solve）
├── data/generators.py     # 合成实例（同 seed 确定性）+ JSON IO + 结构校验
├── solvers/
│   ├── registry.py        # backend 探测 + auto/ortools/fallback 路由
│   ├── tsp_ortools.py     # Routing + Guided Local Search
│   ├── tsp_heuristic.py   # 最近邻多起点 + 2-opt（HPO 参数可注入）
│   ├── knapsack.py        # CP-SAT exact / 精确 DP / 贪心
│   └── assignment.py      # CP-SAT exact / scipy Hungarian
├── eval/metrics.py        # optimality_gap + brute_force 参照（小规模）
├── hpo/tune.py            # Optuna 调 2-opt 超参（独立验证实例防泄漏）
├── pipeline/pipeline.py   # OptiPipeline.run / benchmark / save / format_table
└── cli.py                 # argparse 入口
```

依赖方向：`cli → pipeline → {data, hpo, solvers, eval} → core`，无环。
每个模块可独立验证（单测覆盖）且可组合（pipeline 串联）。

## 3. 接口契约

- `Solver.solve(problem, timeout_s=10, **hpo_params) -> Solution`
  **不抛异常**：内部失败转为 `status="error"` 的 Solution，保证 benchmark 循环健壮。
- `Solution.score` 统一「越小越好」：`min → objective`，`max → -objective`。
  跨问题、跨求解器公平比较全靠这一个字段。
- `Solution.status ∈ {optimal, feasible, heuristic, infeasible, error}`。
- 生成器确定性：同 `(kind, n, seed)` 产出逐位相同实例（单测断言）。

## 4. 关键设计决策

1. **目标值整数化**：TSP 距离 ×1000 取整，满足 OR-Tools 整型弧代价要求，且消除浮点比较噪声。
2. **DP 容量护栏**：`n*capacity > 400_000` 时 DP 拒解（返回 error），registry 层排序保证 ortools/greedy 兜住大规模。
3. **HPO 防泄漏**：调参用独立验证实例（tsp, n=40, seed=9042），与 benchmark 数据集（seed 101/42）不相交。
4. **双层 gap 参照**：小规模暴力精确参照；规模超限时以 best-known 为参照并在 `reference` 字段标注口径——数值诚实可溯源。
5. **solve 契约不抛异常**：错误信息进 `Solution.error`，一行 solver 失败不拖垮全表。

## 5. 评测协议

- 全部实例固定 seed（101 / 42 / 9042），`benchmark_out/benchmark.json` 可逐位复现。
- 计时口径：`time.perf_counter()` 包住单次 solve（含 OR-Tools GLS 用满 time limit 的语义）。
- HPO：Optuna TPE，`n_starts ∈ [1,8]`、`max_passes ∈ [1,40]`、`first_improve ∈ {T,F}`，
  目标 = 验证实例 tour length（minimize，与全局 score 语义一致）。

## 6. 复现命令

```bash
python -m pytest -q -W ignore::UserWarning       # 59 tests
python examples/run_demo.py                       # benchmark_out/benchmark.json
OPTIFORGE_BACKEND=fallback python -m optiforge.cli --problem all --size 12 --seed 7
```
