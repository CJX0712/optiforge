# OptiForge

**模块化组合优化（Combinatorial Optimization）基准系统** — TSP / 0-1 背包 / 分配问题。
复用 Google **OR-Tools**（CP-SAT + Routing GLS）、**scipy**（Hungarian）、**Optuna**（HPO），
全部求解器带 **numpy/scipy 离线兜底**，零下载即可跑通端到端。

> Author: 晨星 (CJX0712) · License: MIT · Python ≥ 3.10 · CPU-only

## 一键复现

```bash
git clone https://github.com/CJX0712/optiforge.git
cd optiforge
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt   # Windows
# Linux/macOS: python -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
.venv/Scripts/python -m pytest -q -W ignore::UserWarning        # 单测全绿
.venv/Scripts/python examples/run_demo.py                       # demo -> benchmark_out/benchmark.json
```

CLI：

```bash
python -m optiforge.cli --problem all --size 40 --seed 42 --backend auto --output benchmark.json
python -m optiforge.cli --problem tsp --hpo --hpo-trials 20   # 先 Optuna 调参再评测
python -m optiforge.cli --list-solvers                        # 探测可用后端
```

Docker：

```bash
docker build -t optiforge .
docker run --rm -v "$PWD/benchmark_out:/app/benchmark_out" optiforge
```

## 量化基线（seed=42，固定实例，benchmark.json 可复现）

| problem | solver | backend | objective | status | gap | time_s |
|---|---|---|---|---|---|---|
| tsp_n8_s101 | tsp_ortools | ortools | 263983 | feasible | 0.00% (vs brute force) | 3.0 |
| tsp_n15_s101 | tsp_heuristic | numpy | 383573 | heuristic | 1.30% (vs best-known) | 0.001 |
| tsp_n40_s42 | tsp_ortools | ortools | 511625 | feasible | 0.00% (best-known) | 3.0 |
| tsp_n40_s42 | tsp_heuristic | numpy | 514340 | heuristic | 0.53% | 0.038 |
| knapsack_n15_s101 | knapsack_dp | numpy | 454 | **optimal** | 0.00% | <0.001 |
| knapsack_n15_s101 | knapsack_greedy | numpy | 450 | heuristic | 0.88% | <0.001 |
| assignment_n40_s42 | assignment_scipy | scipy | 315 | **optimal** | 0.00% | <0.001 |

HPO（Optuna TPE, 12 trials, seed=42）: `{'n_starts': 3, 'max_passes': 39, 'first_improve': True}` → tour 518739.0（验证实例 tsp_n40_s9042）。

## 统一评测语义

`Solution.score` **恒为「越小越好」**：min 问题 `score = objective`；max 问题（背包）`score = -objective`。
gap 百分比：有暴力参照（n≤9 TSP / n≤22 背包 / n≤8 分配）用精确最优；否则以 best-known 为参照并标注 `reference` 字段。

## 架构

```
cli → pipeline → {data, hpo, solvers, eval} → core      （单向无环）
```

详见 [docs/architecture.md](docs/architecture.md)。

## 后端降级矩阵

| 问题 | SOTA 后端 | 离线兜底 | 兜底保证 |
|---|---|---|---|
| TSP | OR-Tools Routing (GLS) | 最近邻 + 2-opt（numpy） | heuristic, gap ≤1.5% |
| 0-1 背包 | OR-Tools CP-SAT (exact) | 精确 DP + 贪心（numpy） | DP 仍为 **optimal** |
| 分配 | OR-Tools CP-SAT (exact) | scipy Hungarian | 仍为 **optimal** |

`OPTIFORGE_BACKEND=fallback` 可强制纯兜底模式（实测上表右侧全通过）。

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `OPTIFORGE_SEED` | 42 | 全局随机种子 |
| `OPTIFORGE_BACKEND` | auto | auto / ortools / fallback |
| `OPTIFORGE_TIMEOUT_S` | 10 | 单次求解超时（秒） |
| `OPTIFORGE_TRIALS` | 20 | HPO trial 数 |
| `OPTIFORGE_OUTPUT_DIR` | benchmark_out | 输出目录 |
