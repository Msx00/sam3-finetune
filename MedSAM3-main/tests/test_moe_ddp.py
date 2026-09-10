"""CPU DDP regression tests for post-forward MoE supervision."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from models.moe_injector import HierarchicalMoEController
from models.moe_lora import RoutedMoELinear
from models.moe_losses import HierarchicalMoELoss


class _AllFallbackModel(nn.Module):
    def __init__(self, *, use_image_locator: bool) -> None:
        super().__init__()
        self.controller = HierarchicalMoEController(
            embed_dim=4,
            rank=2,
            alpha=2,
            dropout=0.0,
            router_hidden_dim=8,
            routing_mode="top1",
            temperature=1.0,
            dac=False,
            use_image_locator=use_image_locator,
            confidence_routing=True,
            confidence_low_threshold=1.0,
            confidence_high_threshold=1.0,
        )
        self.projection = RoutedMoELinear(
            nn.Linear(4, 4), self.controller, projection_key="ddp_projection"
        )

    def forward(
        self, values: torch.Tensor, queries: torch.Tensor, image: torch.Tensor
    ) -> torch.Tensor:
        self.controller.current_routes = self.controller.router(queries, image)
        return self.projection(values)


def _run_all_fallback_backward(*, use_image_locator: bool, locator_weight: float) -> None:
    model = _AllFallbackModel(use_image_locator=use_image_locator)
    wrapped = DDP(model, find_unused_parameters=True)
    controller = model.controller
    controller.set_routing_targets(
        {
            "modality": torch.tensor([0, 1]),
            "area": torch.tensor([0, 2]),
            "boundary": torch.tensor([1, 2]),
        }
    )

    output = wrapped(
        torch.randn(2, 4, 4),
        torch.randn(5, 2, 4),
        torch.randn(2, 4, 4, 4),
    )
    routes = controller.current_routes
    assert routes is not None
    assert torch.count_nonzero(routes["area_joint"].detach()) == 0
    assert torch.count_nonzero(routes["boundary_joint"].detach()) == 0

    loss_module = HierarchicalMoELoss(
        {
            "sam3_loss": 1.0,
            "locator_loss": locator_weight,
            "modality_loss": 1.0,
            "area_loss": 1.0,
            "area_reg_loss": 1.0,
            "boundary_router_loss": 1.0,
            "load_balance_loss": 1.0,
        }
    )
    total, _ = loss_module(
        sam3_core_loss=output.square().mean(),
        final_logits=output,
        gt_masks=torch.zeros_like(output),
        aux_logits=None,
        routes=routes,
        routing_losses=controller.routing_supervision_losses(),
        area_ratio_gt=torch.tensor([0.1, 0.2]),
        locator_logits=routes["image_locator_logits"][:, 0],
    )
    # Before the graph anchors this raises "Expected to mark a variable ready
    # only once": DDP declares the fallback router unused at forward return,
    # then the losses above reach it after the fact.
    total.backward()
    assert any(
        parameter.grad is not None
        for parameter in controller.router.parameters()
        if parameter.requires_grad
    )


def test_ddp_all_shared_fallback_supports_post_forward_router_losses():
    if not dist.is_available() or not dist.is_gloo_available():
        return
    # Do not interfere with a process group owned by a wider distributed suite.
    if dist.is_initialized():
        return
    previous_interface = os.environ.get("GLOO_SOCKET_IFNAME")
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    try:
        with tempfile.TemporaryDirectory(prefix="hmoe_ddp_test_") as directory:
            store_path = Path(directory) / "store"
            try:
                dist.init_process_group(
                    backend="gloo",
                    init_method=store_path.as_uri(),
                    rank=0,
                    world_size=1,
                )
            except RuntimeError as error:
                # Some hermetic sandboxes forbid even loopback sockets. Normal
                # CI/training hosts still execute the regression; restricted
                # environments report an explicit skip instead of a false code
                # failure.
                message = str(error)
                if "Operation not permitted" in message or "Cannot resolve" in message:
                    pytest.skip(f"CPU Gloo unavailable in this sandbox: {message}")
                raise
            _run_all_fallback_backward(
                use_image_locator=True, locator_weight=0.5
            )
            # The no-locator ablation deliberately gives the independent locator
            # head zero loss weight; it must remain safe under the same DDP mode.
            _run_all_fallback_backward(
                use_image_locator=False, locator_weight=0.0
            )
            dist.destroy_process_group()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        if previous_interface is None:
            os.environ.pop("GLOO_SOCKET_IFNAME", None)
        else:
            os.environ["GLOO_SOCKET_IFNAME"] = previous_interface
