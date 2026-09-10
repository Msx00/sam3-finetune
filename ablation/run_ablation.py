#!/usr/bin/env python3
"""Generate, validate, and execute the HMoE-SAM3 ablation matrix.

Examples
--------
python ablation/run_ablation.py list --suite primary
python ablation/run_ablation.py generate --suite primary --seeds 13 42 73
python ablation/run_ablation.py run --suite smoke --device-sets 0 --dry-run
python ablation/run_ablation.py run --suite primary --device-sets 0 1 2 3
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import yaml

from ablation_lib import (
    RunSpec,
    StudyError,
    experiment_table,
    generate_run_specs,
    load_study,
    load_yaml,
    resolve_path,
    select_experiment_ids,
)


HERE = Path(__file__).resolve().parent
DEFAULT_STUDY = HERE / "study.yaml"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def parse_seed_values(values: Sequence[int] | None) -> list[int] | None:
    if not values:
        return None
    seeds = [int(value) for value in values]
    if len(set(seeds)) != len(seeds):
        raise StudyError("--seeds contains duplicates")
    return seeds


def add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--suite", default="primary")
    parser.add_argument(
        "--include", default=None,
        help="Optional regular expression applied to experiment ids",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument(
        "--generated-root", type=Path, default=HERE / "generated",
        help="Directory for fully resolved YAML files",
    )
    parser.add_argument(
        "--output-root", type=Path, default=None,
        help="Training output root; default keeps the base config's output parent",
    )


def specs_from_args(args: argparse.Namespace, write: bool = True):
    study, study_path = load_study(args.study)
    specs = generate_run_specs(
        study=study,
        study_path=study_path,
        suite=args.suite,
        seeds=parse_seed_values(args.seeds),
        include=args.include,
        generated_root=args.generated_root.expanduser().resolve(),
        output_root=args.output_root.expanduser().resolve() if args.output_root else None,
        write=write,
    )
    return study, study_path, specs


def command_list(args: argparse.Namespace) -> int:
    study, _ = load_study(args.study)
    table = experiment_table(study)
    ids = select_experiment_ids(study, args.suite, args.include)
    print(f"suite={args.suite} experiments={len(ids)} seeds={args.seeds or study['seeds']}")
    for index, experiment_id in enumerate(ids, 1):
        entry = table[experiment_id]
        print(f"{index:02d}. {experiment_id}")
        print(f"    {entry.get('description', '')}")
        if entry.get("comparison"):
            print(f"    comparison: {entry['comparison']}")
    return 0


def command_validate(args: argparse.Namespace) -> int:
    study, study_path, specs = specs_from_args(args, write=False)
    base_path = resolve_path(str(study["base_config"]), study_path.parent)
    train_entry = resolve_path(str(study["train_entry"]), study_path.parent)
    if not train_entry.is_file():
        raise StudyError(f"Training entry does not exist: {train_entry}")
    print(f"OK study: {study_path}")
    print(f"OK base config: {base_path}")
    print(f"OK training entry: {train_entry}")
    print(f"OK protected data/checkpoint paths unchanged across {len(specs)} runs")
    print(f"Reference: {study['reference']['id']}")
    return 0


def command_generate(args: argparse.Namespace) -> int:
    _, _, specs = specs_from_args(args, write=True)
    print(f"Generated {len(specs)} resolved configs under {args.generated_root.resolve()}")
    print(f"Manifest: {(args.generated_root / args.suite / 'manifest.json').resolve()}")
    if specs:
        print(f"Suite output root: {specs[0].output_dir.parent.parent}")
    return 0


def parse_device_sets(values: Sequence[str]) -> list[list[int]]:
    groups: list[list[int]] = []
    seen: set[int] = set()
    for raw in values:
        try:
            group = [int(value) for value in raw.split(",") if value != ""]
        except ValueError as exc:
            raise StudyError(f"Invalid device set {raw!r}; use e.g. 0 or 0,1") from exc
        if not group:
            raise StudyError(f"Empty device set: {raw!r}")
        overlap = seen.intersection(group)
        if overlap:
            raise StudyError(
                f"GPU ids may not occur in two concurrent device sets: {sorted(overlap)}"
            )
        seen.update(group)
        groups.append(group)
    return groups


def state_path(args: argparse.Namespace) -> Path:
    return args.state_dir.expanduser().resolve() / f"{args.suite}.json"


def load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"runs": {}}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("runs", {}), dict):
        raise StudyError(f"Invalid state file: {path}")
    value.setdefault("runs", {})
    return value


def save_state(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def training_command(
    python: str,
    train_entry: Path,
    spec: RunSpec,
    devices: Sequence[int],
    port: int,
) -> list[str]:
    config = load_yaml(spec.config_path)
    stage = int(config.get("training", {}).get("stage", 5))
    return [
        python,
        "-u",
        str(train_entry),
        "--config",
        str(spec.config_path),
        "--device",
        *map(str, devices),
        "--master_port",
        str(port),
        "--stage",
        str(stage),
    ]


def command_run(args: argparse.Namespace) -> int:
    study, study_path, specs = specs_from_args(args, write=True)
    train_entry = resolve_path(str(study["train_entry"]), study_path.parent)
    repo_dir = train_entry.parent
    device_sets = parse_device_sets(args.device_sets)
    max_parallel = args.max_parallel or len(device_sets)
    if max_parallel < 1 or max_parallel > len(device_sets):
        raise StudyError("--max-parallel must be between 1 and number of device sets")
    if args.dry_run:
        for index, spec in enumerate(specs):
            devices = device_sets[index % len(device_sets)]
            command = training_command(
                args.python, train_entry, spec, devices, args.master_port + index
            )
            print(shlex.join(command))
        return 0

    status_file = state_path(args)
    state = load_state(status_file)
    pending = [
        spec for spec in specs
        if args.rerun_completed
        or state["runs"].get(spec.run_id, {}).get("status") != "completed"
    ]
    skipped = len(specs) - len(pending)
    if skipped:
        print(f"Skipping {skipped} completed runs; use --rerun-completed to override")
    if not pending:
        print("Nothing to run")
        return 0

    logs_dir = args.logs_dir.expanduser().resolve() / args.suite
    logs_dir.mkdir(parents=True, exist_ok=True)
    free_slots = list(range(max_parallel))
    active: dict[int, dict[str, Any]] = {}
    failed = 0

    def launch(spec: RunSpec, slot: int, ordinal: int) -> None:
        devices = device_sets[slot]
        port = args.master_port + slot
        command = training_command(args.python, train_entry, spec, devices, port)
        log_path = logs_dir / f"{spec.run_id}.log"
        log_handle = log_path.open("a", encoding="utf-8", buffering=1)
        header = f"\n[{utc_now()}] START {shlex.join(command)}\n"
        log_handle.write(header)
        print(f"[{ordinal}/{len(pending)}] START {spec.run_id} GPU={devices}")
        process = subprocess.Popen(
            command,
            cwd=repo_dir,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
        state["runs"][spec.run_id] = {
            "status": "running",
            "experiment_id": spec.experiment_id,
            "seed": spec.seed,
            "pid": process.pid,
            "devices": devices,
            "command": command,
            "config_path": str(spec.config_path),
            "output_dir": str(spec.output_dir),
            "log_path": str(log_path),
            "started_at": utc_now(),
        }
        save_state(status_file, state)
        active[slot] = {
            "process": process,
            "spec": spec,
            "log_handle": log_handle,
            "log_path": log_path,
        }

    next_index = 0
    try:
        while next_index < len(pending) or active:
            while free_slots and next_index < len(pending):
                slot = free_slots.pop(0)
                launch(pending[next_index], slot, next_index + 1)
                next_index += 1
            finished_slots = []
            for slot, item in list(active.items()):
                returncode = item["process"].poll()
                if returncode is None:
                    continue
                spec = item["spec"]
                item["log_handle"].write(
                    f"[{utc_now()}] END returncode={returncode}\n"
                )
                item["log_handle"].close()
                status = "completed" if returncode == 0 else "failed"
                state["runs"][spec.run_id].update({
                    "status": status,
                    "returncode": returncode,
                    "finished_at": utc_now(),
                })
                save_state(status_file, state)
                print(f"END {spec.run_id} status={status} log={item['log_path']}")
                if returncode != 0:
                    failed += 1
                finished_slots.append(slot)
            for slot in finished_slots:
                active.pop(slot)
                free_slots.append(slot)
            free_slots.sort()
            if failed and args.fail_fast:
                print("Failure detected; stopping active runs (--fail-fast)", file=sys.stderr)
                for item in active.values():
                    item["process"].terminate()
                for item in active.values():
                    item["process"].wait()
                    item["log_handle"].close()
                    spec = item["spec"]
                    state["runs"][spec.run_id].update({
                        "status": "cancelled_by_fail_fast",
                        "finished_at": utc_now(),
                    })
                save_state(status_file, state)
                return 1
            if active and not finished_slots:
                time.sleep(1.0)
    except KeyboardInterrupt:
        print("Interrupt received; terminating active child processes", file=sys.stderr)
        for item in active.values():
            item["process"].terminate()
        for item in active.values():
            item["process"].wait()
            item["log_handle"].close()
            spec = item["spec"]
            state["runs"][spec.run_id].update({
                "status": "interrupted", "finished_at": utc_now(),
            })
        save_state(status_file, state)
        return 130
    print(f"Ablation run finished: completed={len(pending)-failed}, failed={failed}")
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproducible HMoE-SAM3 ablation generator and runner"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="Show experiment hypotheses")
    list_parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    list_parser.add_argument("--suite", default="primary")
    list_parser.add_argument("--include", default=None)
    list_parser.add_argument("--seeds", type=int, nargs="+", default=None)
    list_parser.set_defaults(handler=command_list)

    validate_parser = subparsers.add_parser(
        "validate", help="Validate schema, files, and protected-path invariants"
    )
    add_selection_arguments(validate_parser)
    validate_parser.set_defaults(handler=command_validate)

    generate_parser = subparsers.add_parser(
        "generate", help="Generate one fully resolved YAML per experiment and seed"
    )
    add_selection_arguments(generate_parser)
    generate_parser.set_defaults(handler=command_generate)

    run_parser = subparsers.add_parser("run", help="Execute generated experiments")
    add_selection_arguments(run_parser)
    run_parser.add_argument(
        "--device-sets", nargs="+", default=["0"],
        help="Independent GPU groups, e.g. 0 1 2 3 or 0,1 2,3",
    )
    run_parser.add_argument("--max-parallel", type=int, default=None)
    run_parser.add_argument("--master-port", type=int, default=29600)
    run_parser.add_argument("--python", default=sys.executable)
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--rerun-completed", action="store_true")
    run_parser.add_argument("--fail-fast", action="store_true")
    run_parser.add_argument("--logs-dir", type=Path, default=HERE / "logs")
    run_parser.add_argument("--state-dir", type=Path, default=HERE / "state")
    run_parser.set_defaults(handler=command_run)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.handler(args))
    except (StudyError, OSError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
