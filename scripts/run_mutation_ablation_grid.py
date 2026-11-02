"""Twenty-seed, 500-step ablations of evolved trader trait initialization."""
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


OUTPUT = ROOT / "src" / "mutation_ablation_experiments"
SEEDS = tuple(range(101, 121))
TRAIN_EPISODES = 5
EPISODE_STEPS = 500
INFERENCE_EPISODES = 3
VARIANTS = {
    "evolved_full": {"trait_initialization": "evolutionary"},
    "homogeneous_full": {"trait_initialization": "homogeneous"},
    "evolved_no_adversary": {"trait_initialization": "evolutionary", "mx_bad_agents": 0},
    "evolved_no_kaiser": {"trait_initialization": "evolutionary", "kaiser_effects": False},
    "evolved_no_forecast_reward": {"trait_initialization": "evolutionary", "forecast_reward_weight": 0.0},
    "evolved_frozen_gru": {"trait_initialization": "evolutionary", "train_forecaster": False},
}


def run_one(variant, seed, overrides):
    folder = OUTPUT / variant / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = folder / "checkpoints"
    logger = JSONLogger(str(folder / "logger"), log_file="training.jsonl")
    config = dict(
        n_episodes=TRAIN_EPISODES,
        episode_len=EPISODE_STEPS,
        n_agents=5,
        mx_bad_agents=1,
        seed=seed,
        use_trainer=False,
        eval_freq=0,
        save_freq=5,
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
        trait_initialization="evolutionary",
    )
    config.update(overrides)
    pipeline = TrainingPipeline(TrainConf(**config), logger=logger)
    initial_traits = {aid: dict(agent.traits.__dict__) for aid, agent in pipeline.env.agent_objects.items()}
    pipeline.train()
    logger.flush()
    (folder / "training_log_full.json").write_text(json.dumps(pipeline.train_log, indent=2, allow_nan=False))
    detailed = run_detailed_inference(pipeline, INFERENCE_EPISODES, include_official=False)
    (folder / "inference_detailed.json").write_text(json.dumps(detailed, indent=2, allow_nan=False))
    bbox_rows = [row for row in pipeline.train_log if row["bbox"]["rounds"]]
    models = {aid: agent for aid, agent in pipeline.env.agent_objects.items() if hasattr(agent, "forecaster")}
    termination_counts = {}
    for row in pipeline.train_log:
        reason = row["termination_reason"]
        termination_counts[reason] = termination_counts.get(reason, 0) + 1
    summary = {
        "variant": variant,
        "seed": seed,
        "configuration_overrides": overrides,
        "trait_initialization": config["trait_initialization"],
        "initial_agent_traits": initial_traits,
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
        "termination_counts": termination_counts,
        "inference_summary": summarize_inference(detailed),
        "forecast_diversity": summarize_forecast_diversity(detailed),
        "checkpoint": str(checkpoint_dir / f"policies_ep{pipeline.curr_ep}.pt"),
    }
    (folder / "run_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    return summary, str(pipeline.exp_tracker.exp_dir)


def summarize_variant(rows):
    summary = {"runs": len(rows)}
    for field in ("training_mean_episode_reward", "training_mean_episode_length", "training_mean_bbox_accuracy", "training_mean_ensemble_accuracy", "training_mean_forecast_reward"):
        values = [row[field] for row in rows if row[field] is not None]
        summary[field] = {
            "mean": float(np.mean(values)) if values else None,
            "std_across_seeds": float(np.std(values)) if values else None,
            "per_seed": values,
        }
    summary["inference_by_agent_type"] = {}
    agent_types = sorted({kind for row in rows for kind in row["inference_summary"]["by_agent_type"]})
    for kind in agent_types:
        returns = [row["inference_summary"]["by_agent_type"][kind]["mean_portfolio_return"] for row in rows if kind in row["inference_summary"]["by_agent_type"]]
        summary["inference_by_agent_type"][kind] = {
            "mean_return": float(np.mean(returns)),
            "std_across_seeds": float(np.std(returns)),
            "per_seed": returns,
        }
    summary["forecast_updates_mean"] = {
        aid: float(np.mean([row["forecast_updates_by_agent"][aid] for row in rows if aid in row["forecast_updates_by_agent"]]))
        for aid in sorted({aid for row in rows for aid in row["forecast_updates_by_agent"]})
    }
    diversity = [row["forecast_diversity"] for row in rows]
    summary["inference_bbox_rounds"] = int(sum(row["bbox_round_groups"] for row in diversity))
    summary["all_traders_distinct_forecasts_each_round"] = all(row["all_traders_submitted_distinct_paths_in_every_round"] for row in diversity)
    return summary


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "seeds": list(SEEDS),
        "episodes_per_run": TRAIN_EPISODES,
        "steps_per_episode": EPISODE_STEPS,
        "held_out_inference_episodes_per_run": INFERENCE_EPISODES,
        "bbox_frequency": 20,
        "bbox_duration": 5,
        "variants": VARIANTS,
        "runs": [],
    }
    rows = []
    for variant, overrides in VARIANTS.items():
        for seed in SEEDS:
            result, tracker_path = run_one(variant, seed, overrides)
            rows.append(result)
            run_folder = OUTPUT / variant / f"seed_{seed}"
            manifest["runs"].append({
                "variant": variant,
                "seed": seed,
                "directory": str(run_folder),
                "tracker_directory": tracker_path,
                "logger": str(run_folder / "logger" / "training.jsonl"),
                "training_log": str(run_folder / "training_log_full.json"),
                "inference_details": str(run_folder / "inference_detailed.json"),
                "run_summary": str(run_folder / "run_summary.json"),
                "checkpoint": result["checkpoint"],
            })
            (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
            print(json.dumps({
                "variant": variant,
                "seed": seed,
                "training_bbox_accuracy": result["training_mean_bbox_accuracy"],
                "forecast_updates": result["forecast_updates_by_agent"],
                "inference_returns": result["inference_summary"]["by_agent_type"],
            }, allow_nan=False), flush=True)

    aggregate = {
        "seeds": list(SEEDS),
        "episodes_per_run": TRAIN_EPISODES,
        "steps_per_episode": EPISODE_STEPS,
        "held_out_inference_episodes_per_run": INFERENCE_EPISODES,
        "runs": rows,
        "by_variant": {name: summarize_variant([row for row in rows if row["variant"] == name]) for name in VARIANTS},
    }
    (OUTPUT / "all_runs_summary.json").write_text(json.dumps(aggregate, indent=2, allow_nan=False))
    report = [
        "# Twenty-seed agent-trait mutation and crossover ablations",
        "",
        f"Generated: {manifest['created_utc']}",
        f"Each variant uses {len(SEEDS)} seeds, {TRAIN_EPISODES} training episodes of {EPISODE_STEPS} steps, and {INFERENCE_EPISODES} held-out inference episodes.",
        "The same held-out episodes are replayed to capture individual forecasts. Bbox opens every 20 steps and lasts 5 steps.",
        "The evolutionary rule leaves agent 0 at its type-specific defaults, mutates each odd-indexed agent from the prior agent, and crosses each even-indexed agent from the previous two. The homogeneous control retains constructor defaults.",
        "",
        "| Variant | Train reward | Bbox accuracy | Forecast score | RLAIF inference return | Adversarial inference return | GRU updates / trader |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, result in aggregate["by_variant"].items():
        types = result["inference_by_agent_type"]
        rlaif = types.get("rlaif")
        adversarial = types.get("adversarial")
        rtext = f"{rlaif['mean_return']:.3%} ± {rlaif['std_across_seeds']:.3%}" if rlaif else "n/a"
        atext = f"{adversarial['mean_return']:.3%} ± {adversarial['std_across_seeds']:.3%}" if adversarial else "n/a"
        updates = result["forecast_updates_mean"]
        update_text = f"{np.mean(list(updates.values())):.1f}" if updates else "n/a"
        reward = result["training_mean_episode_reward"]
        accuracy = result["training_mean_bbox_accuracy"]
        forecast = result["training_mean_forecast_reward"]
        report.append(
            f"| {variant} | {reward['mean']:.3f} ± {reward['std_across_seeds']:.3f} | "
            f"{accuracy['mean']:.6f} ± {accuracy['std_across_seeds']:.6f} | "
            f"{forecast['mean']:.4f} ± {forecast['std_across_seeds']:.4f} | "
            f"{rtext} | {atext} | {update_text} |"
        )
    report.extend([
        "",
        "## Across-seed standard deviations",
        "",
        "The all_runs_summary.json records standard deviations and per-seed observations for every metric. Standard deviations are descriptive, not confidence intervals or significance tests.",
        "",
        "## Interpretation",
        "",
        "The homogeneous_full arm isolates the mutation/crossover initialization. Other arms isolate adversary presence, Kaiser effects, forecast reward, or GRU updates while retaining evolved traits.",
        "All returns use held-out synthetic episodes and do not establish real-market profitability.",
        "The no-forecast-reward arm still trains its GRU; frozen-GRU keeps stochastic forecasting/rewards active but disables forecaster parameter updates.",
        "",
        "## Files per run",
        "",
        "Each seed directory contains initial agent trait values, tracker files, raw logger, full training log, checkpoint and detailed per-prediction inference records.",
    ])
    (OUTPUT / "report.md").write_text("\n".join(report) + "\n")
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["aggregate_results"] = str(OUTPUT / "all_runs_summary.json")
    manifest["report"] = str(OUTPUT / "report.md")
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    print(f"Mutation ablation artifacts written to {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
