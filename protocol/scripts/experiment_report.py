#!/usr/bin/env python3
"""Track run events, validate resume policy, and render portable Markdown reports."""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_ROOT = PROJECT_ROOT / "reports"
CHECKPOINT_GLOB = "ar-ckpt*.pth"
WEIGHT_HASHES = {
    "checkpoints/VARSR_C2I.pth": (
        "e6fc619171c810be79b5c3cae671c883b2ded21b912a093419a7ef3f79140984"
    ),
    "checkpoints/VQVAE.pth": (
        "b4b60aaa47e7cff7fdebb479521138efbbd968faa893fe0d4f8c703fd49ddf45"
    ),
}
TRAINING_FIELDS = [
    ("depth", "VAR depth"),
    ("batch_size", "每卡 micro-batch"),
    ("glb_batch_size", "每次前向全局 batch"),
    ("ac", "梯度累积"),
    ("ep", "epochs"),
    ("fp16", "fp16"),
    ("tblr", "传入基础学习率"),
    ("tlr", "代码内峰值学习率"),
    ("alng", "AdaLN gamma"),
    ("wpe", "末端学习率比例"),
    ("wp", "warmup epochs"),
    ("sche", "学习率计划"),
    ("twd", "weight decay"),
    ("twde", "末端 weight decay"),
    ("fuse", "fuse"),
    ("cfg_mode", "训练模式"),
    ("synthetic_negative_fraction", "合成负样本比例"),
    ("var_pretrain_path", "VAR warm start"),
    ("vae_model_path", "VQVAE"),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def relative(path: Path | str | None) -> str:
    if not path:
        return "-"
    value = Path(path)
    if not value.is_absolute():
        value = PROJECT_ROOT / value
    try:
        return value.resolve().relative_to(PROJECT_ROOT).as_posix()
    except (OSError, ValueError):
        return str(value)


def escape(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        text = f"{value:.8g}"
    else:
        text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def read_events(experiment: str) -> list[dict]:
    path = REPORTS_ROOT / experiment / "events.jsonl"
    events: list[dict] = []
    if not path.is_file():
        return events
    for line in path.read_text(errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def append_event(args: argparse.Namespace) -> None:
    report_dir = REPORTS_ROOT / args.experiment
    report_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "time_utc": utc_now(),
        "experiment": args.experiment,
        "mode": args.mode,
        "negative_fraction": args.negative_fraction,
        "phase": args.phase,
        "event": args.event,
        "status": args.status,
        "resume_policy": args.resume_policy,
        "log_path": relative(args.log_path),
        "checkpoint": relative(args.checkpoint),
        "metrics_path": relative(args.metrics_path),
        "exit_code": args.exit_code,
        "command": args.command,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
    }
    events_path = report_dir / "events.jsonl"
    with events_path.open("a", encoding="utf-8", buffering=1) as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    render_report(args.experiment)


def newest_checkpoint(experiment: str) -> Path | None:
    directory = PROJECT_ROOT / "results" / experiment
    candidates = list(directory.glob(CHECKPOINT_GLOB))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def load_checkpoint(path: Path, iters_per_epoch: int | None = None) -> dict:
    summary: dict[str, Any] = {
        "path": relative(path),
        "size_bytes": path.stat().st_size,
        "mtime_utc": datetime.fromtimestamp(
            path.stat().st_mtime, timezone.utc
        ).isoformat(timespec="seconds"),
    }
    try:
        import torch

        try:
            state = torch.load(path, map_location="cpu", mmap=True)
        except TypeError:
            state = torch.load(path, map_location="cpu")
        summary["saved_epoch_field"] = state.get("epoch")
        summary["completed_global_it"] = state.get("iter")
        args_state = state.get("args", {})
        if not isinstance(args_state, dict):
            try:
                args_state = vars(args_state)
            except TypeError:
                args_state = {}
        summary["args"] = {
            key: args_state.get(key)
            for key, _ in TRAINING_FIELDS
            if key in args_state
        }
        summary["best_validation"] = {
            key: args_state.get(key)
            for key in ("best_vL_mean", "best_vL_tail", "best_vacc_mean", "best_vacc_tail")
            if args_state.get(key) is not None
        }
        runtime_fields = (
            "cur_ep", "cur_it", "cur_lr", "cur_wd",
            "L_mean", "L_tail", "acc_mean", "acc_tail",
            "vL_mean", "vL_tail", "vacc_mean", "vacc_tail",
        )
        summary["runtime_state"] = {
            key: args_state.get(key)
            for key in runtime_fields
            if args_state.get(key) is not None
        }
        completed = state.get("iter")
        if isinstance(completed, int) and iters_per_epoch:
            next_global = completed + 1
            next_ep, next_it = divmod(next_global, iters_per_epoch)
            summary["next_global_it"] = next_global
            summary["next_epoch"] = next_ep
            summary["next_iteration"] = next_it
        del state
    except Exception as error:
        summary["load_error"] = f"{type(error).__name__}: {error}"
    return summary


def check_resume(args: argparse.Namespace) -> int:
    result_dir = PROJECT_ROOT / "results" / args.experiment
    checkpoint = newest_checkpoint(args.experiment)
    existing_files = list(result_dir.iterdir()) if result_dir.is_dir() else []

    print(
        f"[resume-check] policy={args.policy}, experiment={args.experiment}",
        flush=True,
    )
    if args.policy == "never":
        if existing_files:
            print(
                "[resume-check] REFUSED: --resume never requires an empty/new "
                f"experiment directory, but files exist in {relative(result_dir)}. "
                "Use --run-tag for a new experiment.",
                file=sys.stderr,
                flush=True,
            )
            return 2
        print("[resume-check] action=FRESH (no existing result files)", flush=True)
        return 0

    if checkpoint is None:
        if args.policy == "required":
            print(
                "[resume-check] REFUSED: --resume required but no "
                f"{CHECKPOINT_GLOB} exists in {relative(result_dir)}",
                file=sys.stderr,
                flush=True,
            )
            return 3
        if existing_files:
            print(
                "[resume-check] WARNING: result files exist but no resumable "
                "checkpoint was found; action=FRESH",
                flush=True,
            )
        else:
            print("[resume-check] action=FRESH (no checkpoint)", flush=True)
        return 0

    info = load_checkpoint(checkpoint, args.iters_per_epoch)
    if "load_error" in info:
        print(
            f"[resume-check] REFUSED: cannot read {relative(checkpoint)}: "
            f"{info['load_error']}",
            file=sys.stderr,
            flush=True,
        )
        return 4

    completed = info.get("completed_global_it")
    next_global = completed + 1 if isinstance(completed, int) else None
    target_iterations = args.epochs * args.iters_per_epoch
    print(f"[resume-check] checkpoint={relative(checkpoint)}", flush=True)
    print(
        f"[resume-check] completed_global_it={completed}, "
        f"next_global_it={next_global}, target_iterations={target_iterations}",
        flush=True,
    )
    if next_global is not None and next_global >= target_iterations:
        print(
            "[resume-check] action=COMPLETE; training will be skipped and "
            "evaluation can continue.",
            flush=True,
        )
        return 10

    print(
        f"[resume-check] action=RESUME at epoch={info.get('next_epoch')}, "
        f"iteration={info.get('next_iteration')}/{args.iters_per_epoch}",
        flush=True,
    )
    return 0


def parse_training_log(experiment: str) -> dict:
    path = PROJECT_ROOT / "results" / experiment / "log.txt"
    if not path.is_file():
        return {}
    records: list[dict] = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            value = ast.literal_eval(line)
        except (SyntaxError, ValueError):
            continue
        if isinstance(value, dict) and "ep" in value:
            records.append(value)
    if not records:
        return {"path": relative(path)}
    valid_losses = [
        item["vL_tail"]
        for item in records
        if isinstance(item.get("vL_tail"), (int, float))
        and item["vL_tail"] >= 0
    ]
    return {
        "path": relative(path),
        "records": len(records),
        "latest": records[-1],
        "best_validation_tail_loss": min(valid_losses) if valid_losses else None,
    }


def find_metric_files(experiment: str, events: list[dict]) -> list[Path]:
    found: dict[str, Path] = {}
    for event in events:
        value = event.get("metrics_path")
        if value and value != "-":
            path = Path(value)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            if path.is_file():
                found[str(path.resolve())] = path
    output_root = PROJECT_ROOT / "outputs" / "alignvar_eval"
    if output_root.is_dir():
        for path in output_root.glob(f"{experiment}_cfg*/**/metrics.json"):
            if path.is_file():
                found[str(path.resolve())] = path
    return sorted(found.values(), key=lambda path: path.stat().st_mtime_ns)


def environment_summary() -> dict:
    summary = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    try:
        import torch

        summary["torch"] = torch.__version__
        summary["cuda_runtime"] = torch.version.cuda
        count = torch.cuda.device_count()
        summary["visible_gpus"] = count
        summary["gpu_names"] = [
            torch.cuda.get_device_properties(index).name for index in range(count)
        ]
    except Exception as error:
        summary["torch_error"] = f"{type(error).__name__}: {error}"
    return summary


def table(rows: list[tuple[Any, Any]], headers: tuple[str, str]) -> list[str]:
    output = [f"| {headers[0]} | {headers[1]} |", "|---|---|"]
    output.extend(f"| {escape(left)} | {escape(right)} |" for left, right in rows)
    return output


def render_report(experiment: str) -> Path:
    report_dir = REPORTS_ROOT / experiment
    report_dir.mkdir(parents=True, exist_ok=True)
    events = read_events(experiment)
    latest_event = events[-1] if events else {}
    mode = latest_event.get("mode", "unknown")
    negative_fraction = latest_event.get("negative_fraction", "-")
    protocol = load_json(PROJECT_ROOT / "config/protocol.json")
    data_report = load_json(PROJECT_ROOT / "data/manifests/data_report.json")
    training = data_report.get("training", {})
    benchmarks = data_report.get("benchmarks", {})

    checkpoints = sorted(
        (PROJECT_ROOT / "results" / experiment).glob(CHECKPOINT_GLOB),
        key=lambda path: path.stat().st_mtime_ns,
    )
    checkpoint_info = [load_checkpoint(path) for path in checkpoints]
    newest = checkpoint_info[-1] if checkpoint_info else {}
    checkpoint_args = newest.get("args", {})
    try:
        forward_global_batch = int(checkpoint_args.get("glb_batch_size", 32))
    except (TypeError, ValueError):
        forward_global_batch = 32
    positive_train_images = int(training.get("total_positive", 94991))
    try:
        negative_fraction_value = float(
            checkpoint_args.get("synthetic_negative_fraction", negative_fraction)
        )
    except (TypeError, ValueError):
        negative_fraction_value = 0.0
    synthetic_negative_images = 0
    cfg_mode = checkpoint_args.get("cfg_mode", mode)
    if cfg_mode == "synthetic_cfg" and negative_fraction_value > 0:
        synthetic_negative_images = math.ceil(
            positive_train_images
            * negative_fraction_value
            / (1.0 - negative_fraction_value)
        )
    total_train_images = positive_train_images + synthetic_negative_images
    iters_per_epoch = (
        total_train_images + forward_global_batch - 1
    ) // forward_global_batch
    for info in checkpoint_info:
        completed = info.get("completed_global_it")
        if isinstance(completed, int):
            next_global = completed + 1
            next_ep, next_it = divmod(next_global, iters_per_epoch)
            info["next_global_it"] = next_global
            info["next_epoch"] = next_ep
            info["next_iteration"] = next_it
    official_log = parse_training_log(experiment)
    metric_files = find_metric_files(experiment, events)
    env = environment_summary()

    lines = [
        f"# 实验记录：{experiment}",
        "",
        f"> 自动更新时间（UTC）：{utc_now()}",
        "",
        "## 当前状态",
        "",
    ]
    lines.extend(
        table(
            [
                ("实验名", experiment),
                ("模式", mode),
                ("合成负样本比例", negative_fraction),
                ("最近阶段", latest_event.get("phase", "-")),
                ("最近事件", latest_event.get("event", "-")),
                ("状态", latest_event.get("status", "尚未启动")),
                ("最近退出码", latest_event.get("exit_code", "-")),
                ("运行机器", latest_event.get("hostname", env.get("hostname", "-"))),
            ],
            ("字段", "记录"),
        )
    )

    lines += ["", "## 数据与协议", ""]
    dataset_rows = [
        ("LSDIR", training.get("LSDIR", "-")),
        ("FFHQ（排序后前 10K）", training.get("FFHQ_first_sorted", "-")),
        ("训练正样本总数", training.get("total_positive", "-")),
        ("在线合成负样本索引", synthetic_negative_images),
        ("每 epoch 总样本数", total_train_images),
    ]
    for name, info in benchmarks.items():
        dataset_rows.append((f"{name} 测试对", info.get("pairs", "-")))
    lines.extend(table(dataset_rows, ("数据项", "数量/设置")))

    lines += ["", "## 训练配置", ""]
    if checkpoint_args:
        lines.extend(
            table(
                [
                    (display_name, checkpoint_args.get(key, "-"))
                    for key, display_name in TRAINING_FIELDS
                ],
                ("参数", "实际 checkpoint 记录"),
            )
        )
    else:
        defaults = protocol.get("official_readme_hyperparameters", {})
        lines.append(
            "尚无训练 checkpoint；以下为启动器锁定的官方 README 配置："
        )
        lines.append("")
        lines.extend(table(list(defaults.items()), ("参数", "设置")))

    lines += ["", "## 断点与训练结果", ""]
    if checkpoint_info:
        lines.append(
            "| checkpoint | 大小 (GiB) | 保存 epoch 字段 | 已完成全局 it | "
            "下次 epoch/it | 修改时间 UTC |"
        )
        lines.append("|---|---:|---:|---:|---|---|")
        for info in checkpoint_info:
            size_gib = info.get("size_bytes", 0) / 1024**3
            next_position = (
                f"{info.get('next_epoch', '-')}/{info.get('next_iteration', '-')}"
            )
            lines.append(
                f"| {escape(info.get('path'))} | {size_gib:.3f} | "
                f"{escape(info.get('saved_epoch_field'))} | "
                f"{escape(info.get('completed_global_it'))} | "
                f"{escape(next_position)} | {escape(info.get('mtime_utc'))} |"
            )
            if info.get("load_error"):
                lines.append(
                    f"| checkpoint 读取错误 | {escape(info['load_error'])} | | | | |"
                )
    else:
        lines.append("尚未生成训练 checkpoint。")

    best_validation = newest.get("best_validation", {})
    if best_validation:
        lines += ["", "checkpoint 恢复用的历史最佳验证统计：", ""]
        lines.extend(table(list(best_validation.items()), ("统计", "最佳值")))
    runtime_state = newest.get("runtime_state", {})
    if runtime_state:
        lines += ["", "最新 checkpoint 内的训练/验证状态：", ""]
        lines.extend(table(list(runtime_state.items()), ("状态字段", "值")))
    if official_log.get("latest"):
        lines += ["", "官方 results/.../log.txt 的最近训练记录：", ""]
        latest_rows = list(official_log["latest"].items())
        latest_rows.append(
            ("best_validation_tail_loss", official_log["best_validation_tail_loss"])
        )
        lines.extend(table(latest_rows, ("训练统计", "值")))
    elif official_log:
        lines.append(f"训练日志存在但暂无 epoch 汇总：{official_log.get('path')}")

    lines += ["", "## AlignVAR 测评结果", ""]
    if not metric_files:
        lines.append("尚未生成 metrics.json。生成阶段可断点跳过已完成且签名一致的预测图。")
    for metric_path in metric_files:
        metrics = load_json(metric_path)
        lines += [
            f"### {relative(metric_path)}",
            "",
            f"- 报告名称：{escape(metrics.get('experiment'))}",
            f"- mode / CFG：{escape(metrics.get('mode'))} / {escape(metrics.get('cfg'))}",
            f"- checkpoint：{escape(metrics.get('checkpoint'))}",
            "",
        ]
        results = metrics.get("results", {})
        metric_names: list[str] = []
        for values in results.values():
            for name in values:
                if name not in metric_names:
                    metric_names.append(name)
        if results:
            lines.append(
                "| 数据集 | " + " | ".join(escape(name) for name in metric_names) + " |"
            )
            lines.append("|---|" + "---:|" * len(metric_names))
            for benchmark, values in results.items():
                rendered = [
                    escape(values.get(name, "-")) for name in metric_names
                ]
                lines.append(
                    f"| {escape(benchmark)} | " + " | ".join(rendered) + " |"
                )

    lines += ["", "## 运行事件与日志", ""]
    if events:
        lines.append("| UTC 时间 | 阶段 | 事件 | 状态 | 退出码 | 日志 |")
        lines.append("|---|---|---|---|---:|---|")
        for event in events:
            lines.append(
                f"| {escape(event.get('time_utc'))} | "
                f"{escape(event.get('phase'))} | "
                f"{escape(event.get('event'))} | "
                f"{escape(event.get('status'))} | "
                f"{escape(event.get('exit_code'))} | "
                f"{escape(event.get('log_path'))} |"
            )
    else:
        lines.append("尚无事件。")

    log_dir = report_dir / "logs"
    logs = sorted(log_dir.glob("*.log")) if log_dir.is_dir() else []
    if logs:
        lines += ["", "日志文件：", ""]
        for path in logs:
            lines.append(
                f"- {relative(path)} ({path.stat().st_size / 1024**2:.2f} MiB)"
            )

    commands = [
        event for event in events if event.get("command") and event.get("command") != "-"
    ]
    if commands:
        lines += ["", "实际启动命令：", ""]
        for event in commands:
            lines.append(
                f"- [{escape(event.get('time_utc'))}] "
                f"{escape(event.get('command'))}"
            )

    lines += ["", "## 环境与可审计信息", ""]
    env_rows = list(env.items())
    env_rows.extend(
        [
            ("官方 VARSR commit", "71db492b7698b3df72413ea3552f3f627936b608"),
            ("模型结构/损失/tokenizer", "未修改"),
            ("断点恢复基础设施", "修复官方 start_it 未生效与全局 it 到下一批次的映射"),
        ]
    )
    for path, digest in WEIGHT_HASHES.items():
        env_rows.append((f"SHA-256 {path}", digest))
    lines.extend(table(env_rows, ("审计项", "记录")))

    lines += [
        "",
        "## 结果命名边界",
        "",
        "- no_cfg 必须报告为 VARSR without Image-CFG，并使用 cfg=0。",
        "- synthetic_cfg 必须标注 synthetic negative CFG 和实际负样本比例；它不是官方未公开的 IQA/IAA 负样本协议。",
        "- 初始化权重为官方 VARSR_C2I.pth 与官方 VQVAE.pth。",
        "",
    ]

    report_path = report_dir / "REPORT.md"
    temporary = report_path.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(report_path)
    render_index()
    print(f"[report] updated {relative(report_path)}", flush=True)
    return report_path


def render_index() -> None:
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for report in sorted(REPORTS_ROOT.glob("*/REPORT.md")):
        experiment = report.parent.name
        events = read_events(experiment)
        latest = events[-1] if events else {}
        rows.append(
            (
                experiment,
                latest.get("mode", "-"),
                latest.get("phase", "-"),
                latest.get("status", "-"),
                latest.get("time_utc", "-"),
                relative(report),
            )
        )
    lines = [
        "# VARSR 实验报告索引",
        "",
        f"> 自动更新时间（UTC）：{utc_now()}",
        "",
        "| 实验 | 模式 | 最近阶段 | 状态 | 最近时间 | 报告 |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(escape(value) for value in row) + " |")
    lines.append("")
    destination = REPORTS_ROOT / "INDEX.md"
    temporary = destination.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)

    resume = subparsers.add_parser("resume-check")
    resume.add_argument("--experiment", required=True)
    resume.add_argument(
        "--policy", choices=("auto", "required", "never"), required=True
    )
    resume.add_argument("--epochs", type=int, required=True)
    resume.add_argument("--iters-per-epoch", type=int, required=True)

    event = subparsers.add_parser("event")
    event.add_argument("--experiment", required=True)
    event.add_argument("--mode", required=True)
    event.add_argument("--negative-fraction", default="0")
    event.add_argument("--phase", required=True)
    event.add_argument("--event", required=True)
    event.add_argument("--status", required=True)
    event.add_argument("--resume-policy", default="-")
    event.add_argument("--log-path", default="-")
    event.add_argument("--checkpoint", default="-")
    event.add_argument("--metrics-path", default="-")
    event.add_argument("--exit-code", default="-")
    event.add_argument("--command", default="-")

    refresh = subparsers.add_parser("refresh")
    refresh.add_argument("--experiment", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.action == "resume-check":
        return check_resume(args)
    if args.action == "event":
        append_event(args)
        return 0
    render_report(args.experiment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
