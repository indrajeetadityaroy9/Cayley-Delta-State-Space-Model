"""Evaluation and benchmarking orchestration for CDSSM."""

import datetime
import json
from pathlib import Path

from cdssm.config.experiment import ExperimentConfig


def _stamp_outputs(output_dir: Path, config: ExperimentConfig, metrics: dict) -> None:
    """Write metrics.json and config.yaml into output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics["timestamp"] = datetime.datetime.now().isoformat()
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
        f.write(config.to_yaml())


def run_eval(exp: ExperimentConfig) -> None:
    """Evaluate a checkpoint's perplexity on the configured dataset."""
    from cdssm.data.datasets import TokenDataset, build_dataloader
    from cdssm.inference.predict import load_model_from_checkpoint
    from cdssm.training.trainer import compute_perplexity, evaluate_epoch

    model, tokenizer = load_model_from_checkpoint(exp.eval.checkpoint)

    val_dataset = TokenDataset(
        dataset_name=exp.data.dataset_name,
        dataset_config=exp.data.dataset_config,
        split="validation",
        context_length=model.config.context_length,
        num_tokens=exp.data.num_tokens,
        text_field=exp.data.text_field,
    )
    val_loader = build_dataloader(
        val_dataset, batch_size=exp.data.batch_size, shuffle=False,
    )

    val_loss = evaluate_epoch(model, val_loader)
    val_ppl = compute_perplexity(val_loss)
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation PPL:  {val_ppl:.2f}")

    _stamp_outputs(
        Path(exp.eval.output_dir),
        exp,
        {"val_loss": val_loss, "val_ppl": val_ppl},
    )


def run_bench(exp: ExperimentConfig) -> None:
    """Run lm-evaluation-harness benchmarks on a checkpoint."""
    import lm_eval

    from cdssm.evaluation.lm_eval_wrapper import CDSSMEvalWrapper

    wrapper = CDSSMEvalWrapper(exp.eval.checkpoint)

    results = lm_eval.simple_evaluate(
        model=wrapper,
        tasks=exp.eval.tasks,
        batch_size=wrapper.batch_size,
    )

    metrics = {}
    for task_name, task_results in results["results"].items():
        print(f"\n{task_name}:")
        task_metrics = {}
        for metric, value in task_results.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
                task_metrics[metric] = value
        metrics[task_name] = task_metrics

    _stamp_outputs(Path(exp.eval.output_dir), exp, metrics)
