"""OptiForge 端到端演示：合成实例 → 可选 HPO → 跨求解器 benchmark → benchmark.json。

零下载可跑：ortools 缺失时自动降级 numpy/scipy 兜底。
"""
from __future__ import annotations

import json
from pathlib import Path

from optiforge.core.config import get_config
from optiforge.data.generators import gen_problem
from optiforge.hpo.tune import tune
from optiforge.pipeline.pipeline import (OptiPipeline, format_table,
                                         save_benchmark)


def main() -> None:
    cfg = get_config()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== OptiForge demo (seed fixed, reproducible) ===")
    print(f"backend={cfg.backend} timeout={cfg.timeout_s}s seed={cfg.seed}\n")

    # 1) HPO（小 trial 数，演示可复现调参）
    print("[1/3] Optuna HPO on TSP heuristic (trials=12)...")
    hpo_res = tune(trials=12, seed=cfg.seed)
    print(f"      best_params={hpo_res['best_params']} best_value={hpo_res['best_value']:.1f}\n")

    # 2) 固定 seed 数据集（三域 × 多规模，保证可复现；n=8 TSP 可暴力校验 gap）
    print("[2/3] benchmark over fixed-seed instances...")
    problems = [gen_problem("tsp", 8, 101)]  # 暴力可解，提供精确 gap 参照
    for kind in ("tsp", "knapsack", "assignment"):
        for n, seed in ((15, 101), (40, cfg.seed)):
            if kind == "knapsack" and n > 25:
                continue  # DP 容量保护：n=40 背包由 ortools/greedy 覆盖
            problems.append(gen_problem(kind, n, seed))
    pipeline = OptiPipeline(backend=cfg.backend, timeout_s=min(cfg.timeout_s, 3.0),
                            hpo_params={"tsp": hpo_res["best_params"]})
    result = pipeline.benchmark(problems)

    # 3) 表格 + 落盘
    print("[3/3] results:")
    print(format_table(result["rows"]))
    bench_path = out_dir / "benchmark.json"
    save_benchmark(result, bench_path)
    (out_dir / "hpo.json").write_text(json.dumps(hpo_res, indent=2), encoding="utf-8")
    print(f"\n[ok] benchmark -> {bench_path}")
    print(f"[ok] hpo       -> {out_dir / 'hpo.json'}")


if __name__ == "__main__":
    main()
