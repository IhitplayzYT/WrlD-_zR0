"""Paired-seed training/inference experiment for BBOX enabled versus disabled."""
import json
import platform
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


OUTPUT = ROOT / "src" / "bbox_presence_ablation_experiments"
SEEDS = tuple(range(101, 121))
TRAIN_EPISODES = 5
EPISODE_STEPS = 500
INFERENCE_EPISODES = 3
VARIANTS = {"bbox_enabled": True, "bbox_disabled": False}


def run_one(variant, seed, enabled):
    folder = OUTPUT / variant / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    logger = JSONLogger(str(folder / "logger"), log_file="training.jsonl")
    config = TrainConf(
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
        checkpoint_dir=str(folder / "checkpoints"),
        log_dir=str(folder / "tracker"),
        black_box_frequency=20,
        black_box_duration=5,
        bbox_enabled=enabled,
        adversial_train=True,
        kaiser_effects=True,
        use_robust_ensemble=True,
        forecast_reward_weight=0.5,
        train_forecaster=True,
        trait_initialization="evolutionary",
    )
    pipeline = TrainingPipeline(config, logger=logger)
    pipeline.train()
    logger.flush()
    (folder / "training_log_full.json").write_text(
        json.dumps(pipeline.train_log, indent=2, allow_nan=False)
    )
    detailed = run_detailed_inference(pipeline, INFERENCE_EPISODES)
    (folder / "inference_detailed.json").write_text(
        json.dumps(detailed, indent=2, allow_nan=False)
    )

    bbox_rows = [row for row in pipeline.train_log if row["bbox"]["rounds"]]
    forecasters = {
        aid: agent.forecaster
        for aid, agent in pipeline.env.agent_objects.items()
        if hasattr(agent, "forecaster")
    }
    summary = {
        "variant": variant,
        "seed": seed,
        "bbox_enabled": enabled,
        "configuration": config.__dict__,
        "episodes_completed": pipeline.curr_ep,
        "training_mean_episode_reward": float(np.mean(pipeline.ep_reward)),
        "training_std_episode_reward": float(np.std(pipeline.ep_reward)),
        "training_mean_episode_length": float(np.mean(pipeline.ep_len)),
        "training_bbox_rounds": int(sum(row["bbox"]["rounds"] for row in pipeline.train_log)),
        "training_mean_bbox_accuracy": (
            float(np.mean([row["bbox"]["mean_accuracy"] for row in bbox_rows]))
            if bbox_rows else None
        ),
        "training_mean_ensemble_accuracy": (
            float(np.mean([
                row["bbox"]["mean_ensemble_accuracy"]
                for row in bbox_rows
                if row["bbox"]["mean_ensemble_accuracy"] is not None
            ])) if bbox_rows else None
        ),
        "training_mean_forecast_reward": (
            float(np.mean([row["bbox"].get("mean_forecast_reward", 0.0) for row in bbox_rows]))
            if bbox_rows else 0.0
        ),
        "forecast_updates_by_agent": {aid: model.updates for aid, model in forecasters.items()},
        "training_reward_components_sum": {
            key: float(sum(
                row["reward_components"][aid][key]
                for row in pipeline.train_log
                for aid in row["reward_components"]
            ))
            for key in ("pnl", "forecast", "preference", "drawdown", "damage", "kaiser_damage")
        },
        "termination_counts": {
            reason: sum(row["termination_reason"] == reason for row in pipeline.train_log)
            for reason in sorted({row["termination_reason"] for row in pipeline.train_log})
        },
        "inference_summary": summarize_inference(detailed),
        "forecast_diversity": summarize_forecast_diversity(detailed),
        "checkpoint": str(folder / "checkpoints" / f"policies_ep{pipeline.curr_ep}.pt"),
    }
    (folder / "run_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    return summary, folder


def metric_summary(rows, selector):
    values = [selector(row) for row in rows]
    values = [float(value) for value in values if value is not None]
    return {
        "n": len(values),
        "mean": float(np.mean(values)) if values else None,
        "std_across_seeds": float(np.std(values)) if values else None,
        "per_seed": values,
    }


def aggregate_variant(rows):
    return {
        "runs": len(rows),
        "training_mean_episode_reward": metric_summary(rows, lambda row: row["training_mean_episode_reward"]),
        "training_mean_episode_length": metric_summary(rows, lambda row: row["training_mean_episode_length"]),
        "training_bbox_rounds": metric_summary(rows, lambda row: row["training_bbox_rounds"]),
        "training_mean_bbox_accuracy": metric_summary(rows, lambda row: row["training_mean_bbox_accuracy"]),
        "training_mean_ensemble_accuracy": metric_summary(rows, lambda row: row["training_mean_ensemble_accuracy"]),
        "training_mean_forecast_reward": metric_summary(rows, lambda row: row["training_mean_forecast_reward"]),
        "rlaif_inference_return": metric_summary(
            rows,
            lambda row: row["inference_summary"]["by_agent_type"].get("rlaif", {}).get("mean_portfolio_return"),
        ),
        "adversarial_inference_return": metric_summary(
            rows,
            lambda row: row["inference_summary"]["by_agent_type"].get("adversarial", {}).get("mean_portfolio_return"),
        ),
        "forecast_updates_per_trader": metric_summary(
            rows,
            lambda row: np.mean(list(row["forecast_updates_by_agent"].values()))
            if row["forecast_updates_by_agent"] else 0.0,
        ),
        "heldout_bbox_round_groups": int(sum(row["forecast_diversity"]["bbox_round_groups"] for row in rows)),
    }


def paired_delta(enabled_rows, disabled_rows, metric):
    enabled = {row["seed"]: metric(row) for row in enabled_rows}
    disabled = {row["seed"]: metric(row) for row in disabled_rows}
    pairs = [(enabled[seed], disabled[seed]) for seed in sorted(enabled.keys() & disabled.keys())]
    differences = [left - right for left, right in pairs if left is not None and right is not None]
    return {
        "n_paired_seeds": len(differences),
        "enabled_minus_disabled_mean": float(np.mean(differences)) if differences else None,
        "paired_delta_std": float(np.std(differences, ddof=1)) if len(differences) > 1 else None,
        "enabled_higher_count": sum(value > 0 for value in differences),
        "disabled_higher_count": sum(value < 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "per_seed_delta": {
            seed: enabled[seed] - disabled[seed]
            for seed in sorted(enabled.keys() & disabled.keys())
            if enabled[seed] is not None and disabled[seed] is not None
        },
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository": str(ROOT),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "seeds": list(SEEDS),
        "training_episodes_per_seed": TRAIN_EPISODES,
        "steps_per_episode_max": EPISODE_STEPS,
        "heldout_inference_episodes_per_seed": INFERENCE_EPISODES,
        "matched_configuration": {
            "n_agents": 5,
            "adversarial_agents": 1,
            "black_box_frequency": 20,
            "black_box_duration": 5,
            "trait_initialization": "evolutionary",
            "forecast_reward_weight_configured": 0.5,
            "evaluation_seeds": "same fixed held-out seeds in both conditions",
        },
        "conditions": {
            "bbox_enabled": {"bbox_enabled": True},
            "bbox_disabled": {"bbox_enabled": False},
        },
        "runs": [],
    }
    all_rows = []
    for variant, enabled in VARIANTS.items():
        for seed in SEEDS:
            result, folder = run_one(variant, seed, enabled)
            all_rows.append(result)
            manifest["runs"].append({
                "variant": variant,
                "seed": seed,
                "directory": str(folder),
                "tracker_directory": str(next((folder / "tracker").glob("training_*"))),
                "logger": str(folder / "logger" / "training.jsonl"),
                "training_log": str(folder / "training_log_full.json"),
                "inference_details": str(folder / "inference_detailed.json"),
                "run_summary": str(folder / "run_summary.json"),
                "checkpoint": result["checkpoint"],
            })
            (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
            print(json.dumps({
                "variant": variant,
                "seed": seed,
                "bbox_rounds": result["training_bbox_rounds"],
                "forecast_updates": result["forecast_updates_by_agent"],
                "rlaif_inference": result["inference_summary"]["by_agent_type"].get("rlaif"),
            }, allow_nan=False), flush=True)

    by_variant = {
        name: aggregate_variant([row for row in all_rows if row["variant"] == name])
        for name in VARIANTS
    }
    enabled_rows = [row for row in all_rows if row["variant"] == "bbox_enabled"]
    disabled_rows = [row for row in all_rows if row["variant"] == "bbox_disabled"]
    paired = {
        "training_mean_episode_reward": paired_delta(enabled_rows, disabled_rows, lambda row: row["training_mean_episode_reward"]),
        "training_mean_episode_length": paired_delta(enabled_rows, disabled_rows, lambda row: row["training_mean_episode_length"]),
        "rlaif_inference_return": paired_delta(
            enabled_rows, disabled_rows,
            lambda row: row["inference_summary"]["by_agent_type"]["rlaif"]["mean_portfolio_return"],
        ),
        "adversarial_inference_return": paired_delta(
            enabled_rows, disabled_rows,
            lambda row: row["inference_summary"]["by_agent_type"]["adversarial"]["mean_portfolio_return"],
        ),
    }
    aggregate = {
        "seeds": list(SEEDS),
        "episodes_per_run": TRAIN_EPISODES,
        "maximum_steps_per_episode": EPISODE_STEPS,
        "heldout_inference_episodes_per_run": INFERENCE_EPISODES,
        "runs": all_rows,
        "by_variant": by_variant,
        "paired_deltas_enabled_minus_disabled": paired,
    }
    (OUTPUT / "all_runs_summary.json").write_text(json.dumps(aggregate, indent=2, allow_nan=False))
    report = [
        "# Paired BBOX enabled versus disabled experiment",
        "",
        f"Generated: {manifest['created_utc']}",
        f"Each condition uses {len(SEEDS)} matched seeds, {TRAIN_EPISODES} training episodes of up to {EPISODE_STEPS} steps, and {INFERENCE_EPISODES} held-out inference episodes.",
        "The two conditions use the same five-agent population, one adversarial agent, evolutionary trait initialization, and held-out seeds.",
        "When disabled, no contest opens, no forecast is submitted or trained, forecast reward weight is zero, and the BBOX/consensus/Kaiser policy feature slots are zero.",
        "",
        "| Condition | Train reward | BBOX accuracy | Ensemble accuracy | RLAIF inference return | Adversarial inference return | Training BBOX rounds | GRU updates / trader |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in by_variant.items():
        rlaif = result["rlaif_inference_return"]
        adversarial = result["adversarial_inference_return"]
        ret = lambda value: "n/a" if value["mean"] is None else f"{value['mean']:.3%} ± {value['std_across_seeds']:.3%}"
        accuracy = result["training_mean_bbox_accuracy"]
        bbox_accuracy = "n/a" if accuracy["mean"] is None else f"{accuracy['mean']:.6f} ± {accuracy['std_across_seeds']:.6f}"
        ensemble = result["training_mean_ensemble_accuracy"]
        ensemble_accuracy = "n/a" if ensemble["mean"] is None else f"{ensemble['mean']:.6f} ± {ensemble['std_across_seeds']:.6f}"
        reward = result["training_mean_episode_reward"]
        updates = result["forecast_updates_per_trader"]
        rounds = result["training_bbox_rounds"]
        report.append(
            f"| {name} | {reward['mean']:.3f} ± {reward['std_across_seeds']:.3f} | "
            f"{bbox_accuracy} | {ensemble_accuracy} | {ret(rlaif)} | {ret(adversarial)} | {rounds['mean']:.1f} | "
            f"{updates['mean']:.1f} ± {updates['std_across_seeds']:.1f} |"
        )
    report.extend([
        "",
        "## Paired-seed effects (enabled minus disabled)",
        "",
        "| Metric | Mean difference | Seeds enabled higher | Seeds disabled higher |",
        "|---|---:|---:|---:|",
    ])
    for key, label in (("training_mean_episode_reward", "Training reward"), ("rlaif_inference_return", "RLAIF held-out return"), ("adversarial_inference_return", "Adversarial held-out return")):
        diff = paired[key]
        if key.endswith("return"):
            value = f"{diff['enabled_minus_disabled_mean'] * 100:.3f} pp"
        else:
            value = f"{diff['enabled_minus_disabled_mean']:.3f}"
        report.append(f"| {label} | {value} | {diff['enabled_higher_count']} | {diff['disabled_higher_count']} |")
    report.extend([
        "",
        "The disabled condition has no BBOX accuracy value because no predictions are made. Standard deviations describe seed variation; they are not significance tests.",
        "Return differences are percentage points (pp).",
        "All runs use synthetic market data. Results are evidence about this simulator configuration, not real-market profitability.",
        "Raw per-seed tracker data, logs, checkpoints, training traces, and inference records are stored under the condition/seed folders.",
    ])
    (OUTPUT / "report.md").write_text("\n".join(report) + "\n")
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["aggregate_results"] = str(OUTPUT / "all_runs_summary.json")
    manifest["report"] = str(OUTPUT / "report.md")
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    print(f"BBOX presence ablation artifacts written to {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
