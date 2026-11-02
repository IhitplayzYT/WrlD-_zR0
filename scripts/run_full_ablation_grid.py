"""Twenty-seed, 700-step factorial and one-factor market ablation grid."""
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


OUTPUT = ROOT / "src" / "full_ablation_experiments"
SEEDS = tuple(range(101, 121))
TRAIN_EPISODES = 5
EPISODE_STEPS = 700
INFERENCE_EPISODES = 3

# The first six arms form the BBOX × ordinary-policy × GRU-learning core.
# The final six each alter one additional subsystem from the trained baseline.
VARIANTS = {
    "bbox_on_rlaif_on_gru_train": {},
    "bbox_off_rlaif_on": {"bbox_enabled": False},
    "bbox_on_rlaif_off_gru_train": {"rlaif_enabled": False},
    "bbox_off_rlaif_off": {"bbox_enabled": False, "rlaif_enabled": False},
    "bbox_on_rlaif_on_gru_frozen": {"train_forecaster": False},
    "bbox_on_rlaif_off_gru_frozen": {"rlaif_enabled": False, "train_forecaster": False},
    "full_no_adversary": {"mx_bad_agents": 0},
    "full_no_kaiser": {"kaiser_effects": False},
    "full_no_forecast_reward": {"forecast_reward_weight": 0.0},
    "full_homogeneous_traits": {"trait_initialization": "homogeneous"},
    "full_weighted_average_ensemble": {"use_robust_ensemble": False},
    "full_no_villain_points": {"bbox_villain_points": False},
}


def run_one(variant, seed, overrides):
    folder = OUTPUT / variant / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    logger = JSONLogger(str(folder / "logger"), log_file="training.jsonl")
    values = {
        "n_episodes": TRAIN_EPISODES,
        "episode_len": EPISODE_STEPS,
        "n_agents": 5,
        "mx_bad_agents": 1,
        "seed": seed,
        "use_trainer": False,
        "eval_freq": 0,
        "save_freq": TRAIN_EPISODES,
        "log_freq": 1,
        "terminal_log_freq": 1,
        "eval_episodes": INFERENCE_EPISODES,
        "checkpoint_dir": str(folder / "checkpoints"),
        "log_dir": str(folder / "tracker"),
        "black_box_frequency": 20,
        "black_box_duration": 5,
        "bbox_enabled": True,
        "rlaif_enabled": True,
        "adversial_train": True,
        "kaiser_effects": True,
        "use_robust_ensemble": True,
        "forecast_reward_weight": 0.5,
        "train_forecaster": True,
        "trait_initialization": "evolutionary",
    }
    values.update(overrides)
    config = TrainConf(**values)
    pipeline = TrainingPipeline(config, logger=logger)
    initial_traits = {
        aid: dict(agent.traits.__dict__)
        for aid, agent in pipeline.env.agent_objects.items()
        if hasattr(agent, "traits")
    }
    pipeline.train()
    logger.flush()
    (folder / "training_log_full.json").write_text(
        json.dumps(pipeline.train_log, indent=2, allow_nan=False)
    )
    detailed = run_detailed_inference(pipeline, INFERENCE_EPISODES, include_official=False)
    (folder / "inference_detailed.json").write_text(
        json.dumps(detailed, indent=2, allow_nan=False)
    )

    bbox_rows = [row for row in pipeline.train_log if row["bbox"]["rounds"]]
    forecasters = {
        aid: agent.forecaster
        for aid, agent in pipeline.env.agent_objects.items()
        if hasattr(agent, "forecaster")
    }
    termination_counts = {}
    for row in pipeline.train_log:
        reason = row["termination_reason"]
        termination_counts[reason] = termination_counts.get(reason, 0) + 1
    inference_summary = summarize_inference(detailed)
    ordinary_role = "rlaif" if config.rlaif_enabled else "base"
    summary = {
        "variant": variant,
        "seed": seed,
        "configuration_overrides": overrides,
        "configuration": config.__dict__,
        "agent_types": {aid: agent.agent_type.value for aid, agent in pipeline.env.agent_objects.items()},
        "initial_agent_traits": initial_traits,
        "episodes_completed": pipeline.curr_ep,
        "training_mean_episode_reward": float(np.mean(pipeline.ep_reward)),
        "training_std_episode_reward": float(np.std(pipeline.ep_reward)),
        "training_mean_episode_length": float(np.mean(pipeline.ep_len)),
        "training_bbox_rounds": int(sum(row["bbox"]["rounds"] for row in pipeline.train_log)),
        "training_mean_bbox_accuracy": (
            float(np.mean([row["bbox"]["mean_accuracy"] for row in bbox_rows])) if bbox_rows else None
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
        "policy_updates_by_agent": {
            aid: getattr(getattr(agent, "policy", None), "updates", 0)
            for aid, agent in pipeline.env.agent_objects.items()
        },
        "forecast_updates_by_agent": {aid: model.updates for aid, model in forecasters.items()},
        "forecast_mean_loss_by_agent": {
            aid: model.loss_sum / max(1, model.updates) for aid, model in forecasters.items()
        },
        "training_reward_components_sum": {
            key: float(sum(
                row["reward_components"][aid][key]
                for row in pipeline.train_log
                for aid in row["reward_components"]
            ))
            for key in ("pnl", "forecast", "preference", "drawdown", "damage", "kaiser_damage")
        },
        "termination_counts": termination_counts,
        "inference_summary": inference_summary,
        "ordinary_agent_role": ordinary_role,
        "ordinary_heldout_return": inference_summary["by_agent_type"].get(ordinary_role, {}).get("mean_portfolio_return"),
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
    result = {"runs": len(rows)}
    for field in (
        "training_mean_episode_reward",
        "training_mean_episode_length",
        "training_bbox_rounds",
        "training_mean_bbox_accuracy",
        "training_mean_ensemble_accuracy",
        "training_mean_forecast_reward",
    ):
        result[field] = metric_summary(rows, lambda row, key=field: row[key])
    result["heldout_return_by_agent_type"] = {}
    kinds = sorted({
        kind
        for row in rows
        for kind in row["inference_summary"]["by_agent_type"]
    })
    for kind in kinds:
        result["heldout_return_by_agent_type"][kind] = metric_summary(
            rows,
            lambda row, role=kind: row["inference_summary"]["by_agent_type"].get(role, {}).get("mean_portfolio_return"),
        )
    result["policy_updates_per_agent"] = {
        aid: float(np.mean([row["policy_updates_by_agent"].get(aid, 0) for row in rows]))
        for aid in sorted({aid for row in rows for aid in row["policy_updates_by_agent"]})
    }
    result["forecast_updates_per_agent"] = {
        aid: float(np.mean([row["forecast_updates_by_agent"].get(aid, 0) for row in rows]))
        for aid in sorted({aid for row in rows for aid in row["forecast_updates_by_agent"]})
    }
    result["heldout_bbox_round_groups"] = int(sum(row["forecast_diversity"]["bbox_round_groups"] for row in rows))
    return result


def paired_effect(rows_by_variant, left, right, metric):
    a = {row["seed"]: metric(row) for row in rows_by_variant[left]}
    b = {row["seed"]: metric(row) for row in rows_by_variant[right]}
    pairs = [(a[seed], b[seed]) for seed in sorted(a.keys() & b.keys())]
    diffs = [x - y for x, y in pairs if x is not None and y is not None]
    return {
        "left": left,
        "right": right,
        "contrast": "left minus right",
        "n_paired": len(diffs),
        "mean_difference": float(np.mean(diffs)) if diffs else None,
        "std_paired_difference": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else None,
        "left_higher": sum(value > 0 for value in diffs),
        "right_higher": sum(value < 0 for value in diffs),
        "ties": sum(value == 0 for value in diffs),
        "per_seed_difference": {
            seed: a[seed] - b[seed]
            for seed in sorted(a.keys() & b.keys())
            if a[seed] is not None and b[seed] is not None
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
        "training_episodes_per_run": TRAIN_EPISODES,
        "maximum_steps_per_episode": EPISODE_STEPS,
        "heldout_inference_episodes_per_run": INFERENCE_EPISODES,
        "baseline_configuration": {
            "n_agents": 5,
            "adversarial_agents": 1,
            "black_box_frequency": 20,
            "black_box_duration": 5,
            "trait_initialization": "evolutionary",
            "forecast_reward_weight": 0.5,
            "robust_ensemble": True,
            "evaluation_seeds": "same fixed held-out seeds in every arm",
        },
        "variants": VARIANTS,
        "runs": [],
    }
    rows = []
    rows_by_variant = {name: [] for name in VARIANTS}
    for variant, overrides in VARIANTS.items():
        for seed in SEEDS:
            result, folder = run_one(variant, seed, overrides)
            rows.append(result)
            rows_by_variant[variant].append(result)
            tracker_dirs = list((folder / "tracker").glob("training_*"))
            manifest["runs"].append({
                "variant": variant,
                "seed": seed,
                "directory": str(folder),
                "tracker_directory": str(tracker_dirs[0]) if tracker_dirs else None,
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
                "agent_types": result["agent_types"],
                "bbox_rounds": result["training_bbox_rounds"],
                "policy_updates": result["policy_updates_by_agent"],
                "forecast_updates": result["forecast_updates_by_agent"],
                "heldout_returns": result["inference_summary"]["by_agent_type"],
            }, allow_nan=False), flush=True)

    by_variant = {name: aggregate_variant(values) for name, values in rows_by_variant.items()}
    contrasts = [
        ("bbox_on_rlaif_on_gru_train", "bbox_off_rlaif_on"),
        ("bbox_on_rlaif_off_gru_train", "bbox_off_rlaif_off"),
        ("bbox_on_rlaif_on_gru_train", "bbox_on_rlaif_off_gru_train"),
        ("bbox_on_rlaif_on_gru_train", "bbox_on_rlaif_on_gru_frozen"),
        ("bbox_on_rlaif_off_gru_train", "bbox_on_rlaif_off_gru_frozen"),
    ]
    paired = {}
    for left, right in contrasts:
        key = f"{left}_vs_{right}"
        paired[key] = {
            "training_reward": paired_effect(
                rows_by_variant, left, right, lambda row: row["training_mean_episode_reward"]
            ),
            "training_bbox_accuracy": paired_effect(
                rows_by_variant, left, right, lambda row: row["training_mean_bbox_accuracy"]
            ),
        }
        for role in sorted({
            kind
            for row in rows_by_variant[left] + rows_by_variant[right]
            for kind in row["inference_summary"]["by_agent_type"]
        }):
            paired[key][f"{role}_heldout_return"] = paired_effect(
                rows_by_variant,
                left,
                right,
                lambda row, agent_type=role: row["inference_summary"]["by_agent_type"].get(agent_type, {}).get("mean_portfolio_return"),
            )
        if right == "bbox_on_rlaif_off_gru_train":
            paired[key]["ordinary_heldout_return"] = paired_effect(
                rows_by_variant, left, right, lambda row: row["ordinary_heldout_return"]
            )
    full = "bbox_on_rlaif_on_gru_train"
    for variant in list(VARIANTS)[6:]:
        key = f"{full}_vs_{variant}"
        paired[key] = {
            "training_reward": paired_effect(
                rows_by_variant, full, variant, lambda row: row["training_mean_episode_reward"]
            ),
            "rlaif_heldout_return": paired_effect(
                rows_by_variant,
                full,
                variant,
                lambda row: row["inference_summary"]["by_agent_type"].get("rlaif", {}).get("mean_portfolio_return"),
            ),
            "adversarial_heldout_return": paired_effect(
                rows_by_variant,
                full,
                variant,
                lambda row: row["inference_summary"]["by_agent_type"].get("adversarial", {}).get("mean_portfolio_return"),
            ),
        }

    aggregate = {
        "seeds": list(SEEDS),
        "training_episodes_per_run": TRAIN_EPISODES,
        "maximum_steps_per_episode": EPISODE_STEPS,
        "heldout_inference_episodes_per_run": INFERENCE_EPISODES,
        "runs": rows,
        "by_variant": by_variant,
        "paired_effects": paired,
    }
    (OUTPUT / "all_runs_summary.json").write_text(json.dumps(aggregate, indent=2, allow_nan=False))

    report = [
        "# Twenty-seed full ablation grid",
        "",
        f"Generated: {manifest['created_utc']}",
        f"Each arm uses {len(SEEDS)} seeds, {TRAIN_EPISODES} training episodes of up to {EPISODE_STEPS} steps, and {INFERENCE_EPISODES} held-out inference episodes.",
        "Seeds, episode settings, population size and held-out episodes are matched across arms. The six core arms compare BBOX, RLAIF population, and GRU forecaster learning; six further arms ablate adversary, Kaiser, forecast reward, trait initializer, ensemble method, and villain points.",
        "The RLAIF-off condition replaces ordinary RLAIF agents with non-learning base agents; the adversarial agent remains unless the arm explicitly disables it.",
        "",
        "| Variant | Train reward | BBOX accuracy | Ensemble accuracy | RLAIF held-out return | Base held-out return | Adversarial held-out return | GRU updates / trader |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    fmt = lambda stat, digits=3: "n/a" if stat["mean"] is None else f"{stat['mean']:.{digits}f} ± {stat['std_across_seeds']:.{digits}f}"
    pct = lambda stat: "n/a" if stat["mean"] is None else f"{stat['mean']:.3%} ± {stat['std_across_seeds']:.3%}"
    for name, result in by_variant.items():
        role_returns = result["heldout_return_by_agent_type"]
        updates = list(result["forecast_updates_per_agent"].values())
        update_text = f"{np.mean(updates):.1f}" if updates else "0.0"
        report.append(
            f"| {name} | {fmt(result['training_mean_episode_reward'])} | "
            f"{fmt(result['training_mean_bbox_accuracy'], 6)} | "
            f"{fmt(result['training_mean_ensemble_accuracy'], 6)} | "
            f"{pct(role_returns.get('rlaif', {'mean': None}))} | "
            f"{pct(role_returns.get('base', {'mean': None}))} | "
            f"{pct(role_returns.get('adversarial', {'mean': None}))} | {update_text} |"
        )
    report.extend([
        "",
        "## Selected paired comparisons",
        "",
        "Differences are left minus right; return differences are percentage points. Seed standard deviations and per-seed values are in all_runs_summary.json.",
        "",
        "| Contrast | Metric | Mean difference | Left higher | Right higher |",
        "|---|---|---:|---:|---:|",
    ])
    selected = [
        ("bbox_on_rlaif_on_gru_train_vs_bbox_off_rlaif_on", "rlaif_heldout_return", "BBOX effect, RLAIF on"),
        ("bbox_on_rlaif_off_gru_train_vs_bbox_off_rlaif_off", "base_heldout_return", "BBOX effect, RLAIF off"),
        ("bbox_on_rlaif_on_gru_train_vs_bbox_on_rlaif_off_gru_train", "ordinary_heldout_return", "RLAIF vs base ordinary agents"),
        ("bbox_on_rlaif_on_gru_train_vs_bbox_on_rlaif_on_gru_frozen", "rlaif_heldout_return", "GRU training effect, RLAIF on"),
        ("bbox_on_rlaif_off_gru_train_vs_bbox_on_rlaif_off_gru_frozen", "adversarial_heldout_return", "GRU training effect, RLAIF off"),
    ]
    for contrast, metric, label in selected:
        value = paired.get(contrast, {}).get(metric)
        if value is None:
            continue
        difference = value["mean_difference"]
        shown = "n/a" if difference is None else f"{difference * 100:.3f} pp"
        report.append(f"| {label} | {metric} | {shown} | {value['left_higher']} | {value['right_higher']} |")
    report.extend([
        "",
        "BBOX accuracy is n/a in arms where BBOX is disabled. Differences are descriptive; no significance tests are implied.",
        "All runs use synthetic markets. Results are not evidence of real-market profitability.",
        "Every seed folder contains the tracker, raw logger, full training log, checkpoint, initial traits, run summary and detailed held-out inference record.",
    ])
    (OUTPUT / "report.md").write_text("\n".join(report) + "\n")
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["aggregate_results"] = str(OUTPUT / "all_runs_summary.json")
    manifest["report"] = str(OUTPUT / "report.md")
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    print(f"Full ablation artifacts written to {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
