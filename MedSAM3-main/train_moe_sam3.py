#!/usr/bin/env python3
"""Train the optional Hierarchical LoRA-MoE SAM3 variant.

The original ``train.py`` and ``train_sam3_lora_native.py`` entry points do
not import or enable this path.
"""

import argparse
import os

import yaml

from models.wandb_logger import add_wandb_arguments, resolve_wandb_settings
from models.runtime_config import resolve_moe_runtime_config
from train_sam3_lora_native import (
    SAM3TrainerNative,
    launch_distributed_training,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Hierarchical LoRA-MoE-SAM3")
    parser.add_argument(
        "--config",
        default="configs/moe_sam3.yaml",
        help="MoE YAML configuration",
    )
    parser.add_argument("--device", type=int, nargs="+", default=[0])
    parser.add_argument("--master_port", type=int, default=29500)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument(
        "--stage", type=int, choices=range(1, 6), default=None,
        help="Training stage: 1=baseline, 2=router, 3=MoE, 4=SvANet, 5=joint",
    )
    parser.add_argument("--_launched_by_torchrun", action="store_true")
    parser.add_argument("--resume", default=None, help="Resume same-stage checkpoint")
    add_wandb_arguments(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    num_devices = len(args.device)
    is_torchrun = args._launched_by_torchrun or "LOCAL_RANK" in os.environ
    if num_devices > 1 and not is_torchrun:
        launch_distributed_training(args)
        return
    if num_devices == 1 and not is_torchrun:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device[0])

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    wandb_settings = resolve_wandb_settings(args, config)
    moe_config = resolve_moe_runtime_config(config)
    if not moe_config or not moe_config.get("enabled", True):
        raise ValueError("train_moe_sam3.py requires moe.enabled=true")
    router_config = config.get("router") or {}
    training_stage = int(
        args.stage if args.stage is not None else config.get("training", {}).get("stage", 1)
    )
    trainer = SAM3TrainerNative(
        args.config,
        multi_gpu=(num_devices > 1 and is_torchrun),
        moe_config=moe_config,
        patient_dataset_config=config.get("dataset"),
        router_config=router_config,
        training_stage=training_stage,
        svanet_config=config.get("svanet"),
        resume_path=args.resume,
        wandb_settings=wandb_settings,
    )
    try:
        trainer.train()
    finally:
        trainer.finish_wandb()


if __name__ == "__main__":
    main()
