"""Train independent bbox forecasters and record detailed held-out inference."""
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.training.trainer import EVAL_SEED_BASE, TrainConf, TrainingPipeline
from src.utils.logging import JSONLogger


OUTPUT = ROOT / "src" / "forecast_experiments"
SEEDS = (21, 42, 84)
TRAIN_EPISODES = 12
EPISODE_STEPS = 200
INFERENCE_EPISODES = 5


def prediction_record(result):
    prediction = result.prediction
    return {
        "agent_id": prediction.agent_id,
        "origin_price": prediction.origin_price,
        "predicted_prices": prediction.pred_val,
        "actual_prices": result.actual,
        "predicted_log_return_mean": prediction.return_mean,
        "predicted_log_return_std": prediction.return_std,
        "confidence_interval": list(prediction.conf_interv),
        "confidence_level": prediction.conf_lvl,
        "accuracy": result.accuracy,
        "confidence_calibration": result.conf_accuracy,
        "forecast_reward": result.forecast_reward,
        "is_kaiser": result.is_king,
        "timestamp": prediction.timestamp.isoformat(),
    }


def run_detailed_inference(pipeline, count, include_official=True):
    saved = {aid: agent.snapshot() for aid, agent in pipeline.env.agent_objects.items()}
    official = pipeline.evaluate(n_episodes=count) if include_official else None
    episodes = []
    for index in range(count):
        seed = EVAL_SEED_BASE + index
        env = pipeline.create_environment(seed=seed, evaluation=True)
        for aid, agent in env.agent_objects.items():
            agent.restore(saved[aid])
            agent.training_enabled = False
        observations, _ = env.reset(seed=seed)
        rewards = {aid: 0.0 for aid in env.agents}
        components = {aid: {} for aid in env.agents}
        step_count = 0
        result_cursor = 0
        predictions = []
        for _ in range(env.episode_length):
            actions = {aid: agent.decide_action(observations[aid], env.action_space)
                       for aid, agent in env.agent_objects.items()}
            observations, step_rewards, terminated, truncated, infos = env.step(actions)
            step_count += 1
            for aid, reward in step_rewards.items():
                rewards[aid] += float(reward)
                for key, value in infos[aid]["reward_components"].items():
                    components[aid][key] = components[aid].get(key, 0.0) + float(value)
            results = env.black_box_system.pred_results
            if len(results) > result_cursor:
                predictions.extend(prediction_record(result) for result in results[result_cursor:])
                result_cursor = len(results)
            if terminated or truncated:
                break

        agent_metrics = {}
        for aid, agent in env.agent_objects.items():
            individual = [p for p in predictions if p["agent_id"] == aid]
            portfolio_return = float(env.agent_states[aid].total_value / env.initial_cash - 1.0)
            agent_metrics[aid] = {
                "type": agent.agent_type.value,
                "portfolio_return": portfolio_return,
                "total_reward": rewards[aid],
                "reward_components": components[aid],
                "prediction_count": len(individual),
                "mean_accuracy": float(np.mean([p["accuracy"] for p in individual])) if individual else None,
                "mean_forecast_reward": float(np.mean([p["forecast_reward"] for p in individual])) if individual else None,
                "kaiser_wins": sum(p["is_kaiser"] for p in individual),
                "forecast_updates_before_inference": getattr(getattr(agent, "forecaster", None), "updates", 0),
            }
        episodes.append({
            "episode_index": index,
            "seed": seed,
            "steps": step_count,
            "termination_reason": next(iter(infos.values())).get("termination_reason") if infos else None,
            "buy_and_hold_return": float(env.current_prices[env.available_assets[0]] / env.price_data[env.available_assets[0]][0] - 1.0),
            "market_volatility": float(np.std(np.diff(np.log(env.price_history[env.available_assets[0]])))),
            "trade_count": int(sum(len(book.trades) for book in env.order_book.order_books.values())),
            "episode_summary": env.episode_log(),
            "agents": agent_metrics,
            "bbox_predictions": predictions,
        })
    return {"official_pipeline_evaluation": official, "episodes": episodes}


def summarize_inference(detailed):
    episodes = detailed["episodes"]
    agent_ids = sorted({aid for episode in episodes for aid in episode["agents"]})
    agents = {}
    for aid in agent_ids:
        records = [episode["agents"][aid] for episode in episodes]
        agent_type = records[0]["type"]
        returns = [record["portfolio_return"] for record in records]
        accuracies = [record["mean_accuracy"] for record in records if record["mean_accuracy"] is not None]
        scores = [record["mean_forecast_reward"] for record in records if record["mean_forecast_reward"] is not None]
        agents[aid] = {
            "type": agent_type,
            "mean_portfolio_return": float(np.mean(returns)),
            "std_portfolio_return_across_episodes": float(np.std(returns)),
            "per_episode_portfolio_return": returns,
            "mean_bbox_accuracy": float(np.mean(accuracies)) if accuracies else None,
            "mean_bbox_forecast_reward": float(np.mean(scores)) if scores else None,
            "total_bbox_predictions": int(sum(record["prediction_count"] for record in records)),
            "total_kaiser_wins": int(sum(record["kaiser_wins"] for record in records)),
        }
    by_type = {}
    for kind in sorted({record["type"] for record in agents.values()}):
        values = [record["mean_portfolio_return"] for record in agents.values() if record["type"] == kind]
        by_type[kind] = {"mean_portfolio_return": float(np.mean(values)), "std_across_agents": float(np.std(values))}
    return {"n_inference_episodes": len(episodes), "by_agent_type": by_type, "by_agent": agents}


def summarize_forecast_diversity(detailed):
    groups = []
    for episode in detailed["episodes"]:
        rows = episode["bbox_predictions"]
        population = sum(
            record["type"] in ("rlaif", "adversarial")
            for record in episode["agents"].values()
        )
        if population == 0:
            continue
        for start in range(0, len(rows), population):
            group = rows[start:start + population]
            if len(group) != population:
                continue
            unique_paths = len({tuple(round(value, 8) for value in row["predicted_prices"]) for row in group})
            groups.append({
                "unique_paths": unique_paths,
                "accuracy_std_across_traders": float(np.std([row["accuracy"] for row in group])),
                "mean_cross_trader_path_dispersion": float(np.mean([
                    np.std([row["predicted_prices"][index] for row in group])
                    for index in range(len(group[0]["predicted_prices"]))
                ])),
            })
    expected = (
        sum(record["type"] in ("rlaif", "adversarial") for record in detailed["episodes"][0]["agents"].values())
        if detailed["episodes"] else 0
    )
    return {
        "bbox_round_groups": len(groups),
        "all_traders_submitted_distinct_paths_in_every_round": bool(groups) and all(group["unique_paths"] == expected for group in groups),
        "min_unique_paths_per_round": min((group["unique_paths"] for group in groups), default=0),
        "mean_unique_paths_per_round": float(np.mean([group["unique_paths"] for group in groups])) if groups else 0.0,
        "max_unique_paths_per_round": max((group["unique_paths"] for group in groups), default=0),
        "mean_accuracy_std_across_traders": float(np.mean([group["accuracy_std_across_traders"] for group in groups])) if groups else 0.0,
        "mean_cross_trader_path_dispersion": float(np.mean([group["mean_cross_trader_path_dispersion"] for group in groups])) if groups else 0.0,
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
        "episode_steps": EPISODE_STEPS,
        "inference_episodes_per_seed": INFERENCE_EPISODES,
        "configuration": {
            "n_agents": 5,
            "adversarial_agents": 1,
            "black_box_frequency": 20,
            "black_box_duration": 5,
            "curriculum": False,
            "event_impact": "jump",
            "kaiser_effects": True,
            "forecast_reward_weight": 0.5,
        },
        "artifacts": {},
    }
    run_summaries = []
    for seed in SEEDS:
        folder = OUTPUT / f"seed_{seed}"
        folder.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = folder / "checkpoints"
        logger = JSONLogger(str(folder / "logger"), log_file="training.jsonl")
        config = TrainConf(
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
        )
        pipeline = TrainingPipeline(config, logger=logger)
        pipeline.train()
        logger.flush()
        (folder / "training_log_full.json").write_text(json.dumps(pipeline.train_log, indent=2, allow_nan=False))
        detailed = run_detailed_inference(pipeline, INFERENCE_EPISODES)
        inference_path = folder / "inference_detailed.json"
        inference_path.write_text(json.dumps(detailed, indent=2, allow_nan=False))
        summary = {
            "seed": seed,
            "training_episodes_completed": pipeline.curr_ep,
            "training_mean_episode_reward": float(np.mean(pipeline.ep_reward)),
            "training_std_episode_reward": float(np.std(pipeline.ep_reward)),
            "training_mean_episode_length": float(np.mean(pipeline.ep_len)),
            "training_bbox_rounds": int(sum(row["bbox"]["rounds"] for row in pipeline.train_log)),
            "training_mean_bbox_accuracy": float(np.mean([row["bbox"]["mean_accuracy"] for row in pipeline.train_log if row["bbox"]["rounds"]])),
            "training_mean_bbox_forecast_reward": float(np.mean([row["bbox"].get("mean_forecast_reward", 0.0) for row in pipeline.train_log if row["bbox"]["rounds"]])),
            "training_total_forecast_updates": {aid: model.forecaster.updates for aid, model in pipeline.env.agent_objects.items() if hasattr(model, "forecaster")},
            "training_forecast_mean_loss": {aid: model.forecaster.loss_sum / max(1, model.forecaster.updates) for aid, model in pipeline.env.agent_objects.items() if hasattr(model, "forecaster")},
            "training_reward_components_sum": {
                key: float(sum(row["reward_components"][aid][key] for row in pipeline.train_log for aid in row["reward_components"]))
                for key in ("pnl", "forecast", "preference", "drawdown", "damage", "kaiser_damage")
            },
            "training_termination_reasons": {
                reason: sum(row["termination_reason"] == reason for row in pipeline.train_log)
                for reason in sorted({row["termination_reason"] for row in pipeline.train_log})
            },
            "inference_summary": summarize_inference(detailed),
            "forecast_diversity": summarize_forecast_diversity(detailed),
            "official_pipeline_evaluation": detailed["official_pipeline_evaluation"],
            "checkpoint": str(checkpoint_dir / f"policies_ep{pipeline.curr_ep}.pt"),
        }
        summary_path = folder / "run_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False))
        run_summaries.append(summary)
        manifest["artifacts"][str(seed)] = {
            "run_directory": str(folder),
            "tracker_directory": str(pipeline.exp_tracker.exp_dir),
            "logger": str(folder / "logger" / "training.jsonl"),
            "training_log": str(folder / "training_log_full.json"),
            "inference_details": str(inference_path),
            "summary": str(summary_path),
            "checkpoint": summary["checkpoint"],
        }
        (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
        print(json.dumps({"completed_seed": seed, "training_bbox_rounds": summary["training_bbox_rounds"], "inference": summary["inference_summary"]["by_agent_type"]}, allow_nan=False), flush=True)

    aggregate = {
        "seeds": list(SEEDS),
        "training_episodes_per_seed": TRAIN_EPISODES,
        "episode_steps": EPISODE_STEPS,
        "inference_episodes_per_seed": INFERENCE_EPISODES,
        "runs": run_summaries,
        "aggregate": {},
    }
    for metric in ("training_mean_episode_reward", "training_mean_bbox_accuracy", "training_mean_bbox_forecast_reward"):
        values = [row[metric] for row in run_summaries]
        aggregate["aggregate"][metric] = {"mean": float(np.mean(values)), "std_across_seeds": float(np.std(values)), "per_seed": values}
    type_names = sorted({kind for row in run_summaries for kind in row["inference_summary"]["by_agent_type"]})
    aggregate["aggregate"]["inference_by_agent_type"] = {}
    for kind in type_names:
        values = [row["inference_summary"]["by_agent_type"][kind]["mean_portfolio_return"] for row in run_summaries if kind in row["inference_summary"]["by_agent_type"]]
        aggregate["aggregate"]["inference_by_agent_type"][kind] = {"mean_return": float(np.mean(values)), "std_across_seeds": float(np.std(values)), "per_seed": values}
    (OUTPUT / "all_runs_summary.json").write_text(json.dumps(aggregate, indent=2, allow_nan=False))
    lines = [
        "# Bbox forecast training and inference runs",
        "",
        f"Generated: {manifest['created_utc']}",
        f"Training: {TRAIN_EPISODES} episodes x {EPISODE_STEPS} steps per seed; inference: {INFERENCE_EPISODES} held-out episodes per seed.",
        f"Seeds: {', '.join(map(str, SEEDS))}. Bbox opens every 20 steps and lasts 5 steps.",
        "",
        "| Seed | Train reward | Train bbox accuracy | Train forecast reward | RLAIF inference return | Adversarial inference return | Bbox rounds |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in run_summaries:
        type_metrics = row["inference_summary"]["by_agent_type"]
        rlaif_return = type_metrics.get("rlaif", {}).get("mean_portfolio_return", float("nan"))
        adv_return = type_metrics.get("adversarial", {}).get("mean_portfolio_return", float("nan"))
        lines.append(f"| {row['seed']} | {row['training_mean_episode_reward']:.4f} | {row['training_mean_bbox_accuracy']:.6f} | {row['training_mean_bbox_forecast_reward']:.4f} | {rlaif_return:.4%} | {adv_return:.4%} | {row['training_bbox_rounds']} |")
    lines.extend([
        "",
        "## Aggregated results",
        "",
        "| Measure | Mean across seeds | Standard deviation |",
        "|---|---:|---:|",
    ])
    for metric, label in (("training_mean_episode_reward", "Training episode reward"), ("training_mean_bbox_accuracy", "Training bbox accuracy"), ("training_mean_bbox_forecast_reward", "Training bbox forecast reward")):
        value = aggregate["aggregate"][metric]
        lines.append(f"| {label} | {value['mean']:.6f} | {value['std_across_seeds']:.6f} |")
    for kind, value in aggregate["aggregate"]["inference_by_agent_type"].items():
        lines.append(f"| {kind} held-out portfolio return | {value['mean_return']:.4%} | {value['std_across_seeds']:.4%} |")
    lines.extend([
        "",
        "## Artifact guide",
        "",
        "Each seed folder includes the complete TrainingPipeline tracker directory, raw JSON logger, trainer checkpoints, complete per-episode training log, exact per-prediction inference records, and a run summary. The per-prediction file contains the sampled path, realized path, predicted log-return mean and standard deviation, confidence interval, accuracy, calibration, forecast reward, and Kaiser result.",
        "",
        "These are synthetic-market experiments with three seeds and five inference episodes per seed; they describe this simulation setup and do not establish real-market profitability.",
    ])
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n")
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["aggregate_results"] = str(OUTPUT / "all_runs_summary.json")
    manifest["report"] = str(OUTPUT / "report.md")
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    print(f"Artifacts written to {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
