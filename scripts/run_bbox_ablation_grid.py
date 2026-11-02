"""Ten-seed bbox forecaster ablations with checkpointed held-out inference."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_bbox_forecast_experiments import (  # noqa: E402
    run_detailed_inference,
    summarize_forecast_diversity,
    summarize_inference,
)
from src.training.trainer import TrainConf, TrainingPipeline  # noqa: E402
from src.utils.logging import JSONLogger  # noqa: E402


OUTPUT = ROOT / "src" / "forecast_ablation_experiments"
SEEDS = tuple(range(1, 11))
TRAIN_EPISODES = 12
EPISODE_STEPS = 200
INFERENCE_EPISODES = 5
VARIANTS = {
    "full": {},
    "no_adversary": {"mx_bad_agents": 0},
    "no_kaiser": {"kaiser_effects": False},
    "no_forecast_reward": {"forecast_reward_weight": 0.0},
    "frozen_gru": {"train_forecaster": False},
}


def run_one(variant, seed, overrides):
    folder = OUTPUT / variant / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = folder / "checkpoints"
    logger = JSONLogger(str(folder / "logger"), log_file="training.jsonl")
    config_values = dict(
        n_episodes=TRAIN_EPISODES,
        episode_len=EPISODE_STEPS,
        n_agents=5,
        mx_bad_agents=1,
        seed=seed,
        use_trainer=False,
        eval_freq=0,
        save_freq=6,
        log_freq=1,
        terminal_log_freq=1,
        eval_episodes=INFERENCE_EPISODES,
        checkpoint_dir=str(checkpoint_dir),
        log_dir=str(folder / "tracker"),
        black_box_frequency=20,
        black_box_duration=5,
        adversial_train=True,
        kaiser_effects=True,
        use_robust_ensemble=True,
        forecast_reward_weight=0.5,
        train_forecaster=True,
    )
    config_values.update(overrides)
    pipeline = TrainingPipeline(TrainConf(**config_values), logger=logger)
    pipeline.train()
    logger.flush()
    (folder / "training_log_full.json").write_text(json.dumps(pipeline.train_log, indent=2, allow_nan=False))
    detailed = run_detailed_inference(pipeline, INFERENCE_EPISODES)
    (folder / "inference_detailed.json").write_text(json.dumps(detailed, indent=2, allow_nan=False))

    bbox_rows = [row for row in pipeline.train_log if row["bbox"]["rounds"]]
    models = {aid: agent for aid, agent in pipeline.env.agent_objects.items() if hasattr(agent, "forecaster")}
    summary = {
        "variant": variant,
        "seed": seed,
        "configuration_overrides": overrides,
        "episodes_completed": pipeline.curr_ep,
        "training_mean_episode_reward": float(np.mean(pipeline.ep_reward)),
        "training_std_episode_reward": float(np.std(pipeline.ep_reward)),
        "training_mean_episode_length": float(np.mean(pipeline.ep_len)),
        "training_bbox_rounds": int(sum(row["bbox"]["rounds"] for row in pipeline.train_log)),
        "training_mean_bbox_accuracy": float(np.mean([row["bbox"]["mean_accuracy"] for row in bbox_rows])) if bbox_rows else None,
        "training_mean_ensemble_accuracy": float(np.mean([row["bbox"]["mean_ensemble_accuracy"] for row in bbox_rows if row["bbox"]["mean_ensemble_accuracy"] is not None])) if bbox_rows else None,
        "training_mean_forecast_reward": float(np.mean([row["bbox"].get("mean_forecast_reward", 0.0) for row in bbox_rows])) if bbox_rows else None,
        "forecast_updates_by_agent": {aid: agent.forecaster.updates for aid, agent in models.items()},
        "forecast_mean_loss_by_agent": {aid: agent.forecaster.loss_sum / max(1, agent.forecaster.updates) for aid, agent in models.items()},
        "training_reward_components_sum": {
            key: float(sum(row["reward_components"][aid][key] for row in pipeline.train_log for aid in row["agent_rewards"]))
            for key in ("pnl", "forecast", "preference", "drawdown", "damage", "kaiser_damage")
        },
        "training_termination_counts": {
            reason: sum(row["termination_reason"] == reason for row in pipeline.train_log)
            for reason in sorted({row["termination_reason"] for row in pipeline.train_log})
        },
        "inference_summary": summarize_inference(detailed),
        "forecast_diversity": summarize_forecast_diversity(detailed),
        "official_pipeline_evaluation": detailed["official_pipeline_evaluation"],
        "checkpoint": str(checkpoint_dir / f"policies_ep{pipeline.curr_ep}.pt"),
    }
    (folder / "run_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    return summary, str(pipeline.exp_tracker.exp_dir)


def aggregate_variant(rows):
    result = {"runs": len(rows)}
    for field in ("training_mean_episode_reward", "training_mean_bbox_accuracy", "training_mean_ensemble_accuracy", "training_mean_forecast_reward"):
        values = [row[field] for row in rows if row[field] is not None]
        result[field] = {
            "mean": float(np.mean(values)) if values else None,
            "std_across_seeds": float(np.std(values)) if values else None,
            "per_seed": values,
        }
    result["inference_by_agent_type"] = {}
    types = sorted({kind for row in rows for kind in row["inference_summary"]["by_agent_type"]})
    for kind in types:
        values = [row["inference_summary"]["by_agent_type"][kind]["mean_portfolio_return"] for row in rows if kind in row["inference_summary"]["by_agent_type"]]
        result["inference_by_agent_type"][kind] = {
            "mean_return": float(np.mean(values)),
            "std_across_seeds": float(np.std(values)),
            "per_seed": values,
        }
    result["forecast_updates_mean"] = {
        aid: float(np.mean([row["forecast_updates_by_agent"][aid] for row in rows if aid in row["forecast_updates_by_agent"]]))
        for aid in sorted({aid for row in rows for aid in row["forecast_updates_by_agent"]})
    }
    diversity = [row["forecast_diversity"] for row in rows]
    result["inference_bbox_rounds"] = int(sum(value["bbox_round_groups"] for value in diversity))
    result["all_inference_paths_distinct"] = all(value["all_traders_submitted_distinct_paths_in_every_round"] for value in diversity)
    return result


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "seeds": list(SEEDS),
        "training_episodes_per_run": TRAIN_EPISODES,
        "steps_per_episode": EPISODE_STEPS,
        "held_out_inference_episodes_per_run": INFERENCE_EPISODES,
        "bbox_frequency": 20,
        "bbox_duration": 5,
        "variants": VARIANTS,
        "runs": [],
    }
    all_rows = []
    for variant, overrides in VARIANTS.items():
        for seed in SEEDS:
            summary, tracker_directory = run_one(variant, seed, overrides)
            all_rows.append(summary)
            manifest["runs"].append({
                "variant": variant,
                "seed": seed,
                "directory": str(OUTPUT / variant / f"seed_{seed}"),
                "tracker_directory": tracker_directory,
                "logger": str(OUTPUT / variant / f"seed_{seed}" / "logger" / "training.jsonl"),
                "training_log": str(OUTPUT / variant / f"seed_{seed}" / "training_log_full.json"),
                "inference_details": str(OUTPUT / variant / f"seed_{seed}" / "inference_detailed.json"),
                "run_summary": str(OUTPUT / variant / f"seed_{seed}" / "run_summary.json"),
                "checkpoint": summary["checkpoint"],
            })
            (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
            print(json.dumps({
                "variant": variant,
                "seed": seed,
                "training_bbox_accuracy": summary["training_mean_bbox_accuracy"],
                "forecast_updates": summary["forecast_updates_by_agent"],
                "inference_return": summary["inference_summary"]["by_agent_type"],
            }, allow_nan=False), flush=True)

    aggregate = {
        "seeds": list(SEEDS),
        "training_episodes_per_run": TRAIN_EPISODES,
        "steps_per_episode": EPISODE_STEPS,
        "held_out_inference_episodes_per_run": INFERENCE_EPISODES,
        "runs": all_rows,
        "by_variant": {
            variant: aggregate_variant([row for row in all_rows if row["variant"] == variant])
            for variant in VARIANTS
        },
    }
    (OUTPUT / "all_runs_summary.json").write_text(json.dumps(aggregate, indent=2, allow_nan=False))
    report = [
        "# Ten-seed bbox forecaster ablation grid",
        "",
        f"Generated: {manifest['created_utc']}",
        f"Each variant uses {len(SEEDS)} seeds, {TRAIN_EPISODES} training episodes of {EPISODE_STEPS} steps, and {INFERENCE_EPISODES} held-out inference episodes.",
        "The same held-out episodes are evaluated by the standard pipeline and replayed once to capture individual forecast traces. Bbox opens every 20 steps and lasts 5 steps. Seeds are paired across variants.",
        "",
        "| Variant | Training reward | Bbox accuracy | Forecast reward | RLAIF held-out return | Adversarial held-out return | Mean GRU updates / trader |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, result in aggregate["by_variant"].items():
        types = result["inference_by_agent_type"]
        rlaif = types.get("rlaif", {}).get("mean_return")
        adversary = types.get("adversarial", {}).get("mean_return")
        rlaif_text = f"{rlaif:.3%}" if rlaif is not None else "n/a"
        adversary_text = f"{adversary:.3%}" if adversary is not None else "n/a"
        updates = result["forecast_updates_mean"]
        update_text = f"{np.mean(list(updates.values())):.1f}" if updates else "n/a"
        reward = result["training_mean_episode_reward"]["mean"]
        accuracy = result["training_mean_bbox_accuracy"]["mean"]
        forecast_reward = result["training_mean_forecast_reward"]["mean"]
        report.append(f"| {variant} | {reward:.4f} | {accuracy:.6f} | {forecast_reward:.4f} | {rlaif_text} | {adversary_text} | {update_text} |")
    report.extend([
        "",
        "## Interpretation notes",
        "",
        "Forecast accuracy is measured on price levels and naturally stays near one for small price movements. Compare variants using returns, forecast rewards, and per-seed variation together.",
        "All returns use held-out synthetic episodes. They are not evidence of real-market profitability.",
        "The no_forecast_reward variant still trains the GRU; it removes the forecast score from policy reward. The frozen_gru variant leaves the probabilistic forecaster active but stops its parameter updates.",
        "",
        "## Artifact guide",
        "",
        "Every variant/seed directory contains tracker hyperparameters, metrics and summary, raw JSON logger, intermediate/final checkpoints, full training logs, per-prediction held-out inference records, and a run summary.",
        "",
        "## Across-seed variation",
        "",
        "| Variant | RLAIF return, mean ± SD | Adversarial return, mean ± SD | Training reward, mean ± SD | Bbox accuracy, mean ± SD |",
        "|---|---:|---:|---:|---:|",
    ])
    for variant, result in aggregate["by_variant"].items():
        rlaif = result["inference_by_agent_type"].get("rlaif")
        adversary = result["inference_by_agent_type"].get("adversarial")
        rlaif_text = f"{rlaif['mean_return']:.3%} ± {rlaif['std_across_seeds']:.3%}" if rlaif else "n/a"
        adversary_text = f"{adversary['mean_return']:.3%} ± {adversary['std_across_seeds']:.3%}" if adversary else "n/a"
        reward = result["training_mean_episode_reward"]
        accuracy = result["training_mean_bbox_accuracy"]
        report.append(f"| {variant} | {rlaif_text} | {adversary_text} | {reward['mean']:.3f} ± {reward['std_across_seeds']:.3f} | {accuracy['mean']:.6f} ± {accuracy['std_across_seeds']:.6f} |")
    report.append("\nStandard deviations describe variation across the 10 seeds; they are not confidence intervals or significance tests.")
    (OUTPUT / "report.md").write_text("\n".join(report) + "\n")
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["aggregate_results"] = str(OUTPUT / "all_runs_summary.json")
    manifest["report"] = str(OUTPUT / "report.md")
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    print(f"Ablation artifacts written to {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
