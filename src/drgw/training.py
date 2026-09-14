"""Three-stage training with explicit optimizer-update counts and resumable state."""

from __future__ import annotations

import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from .attacks import edge_flip, node_delete
from .data import prepare_banks
from .runtime import build_model, config_digest, environment, json_write, seed_everything


def training_objective(model, adj, mask, stage, cfg, step):
    """Compute one stage objective. Discrete edits use a straight-through gradient."""
    if stage == "naive":
        losses = model.encoder_losses(adj, mask, adj)
        h, _ = model.encode(adj, mask)
        losses["reconstruction"] = model.reconstruction_loss(adj, mask, h, h)
        losses["latent_l2"] = (h.square() * mask.unsqueeze(-1)).sum() / (mask.sum() * h.shape[-1])
        loss = losses["reconstruction"] + cfg["feature_weight"] * losses["feature"]
        loss = loss + cfg["variance_weight"] * losses["variance"] + 0.001 * losses["latent_l2"]
        return loss, losses
    if stage == "stage1":
        augmented, aug_mask = edge_flip(adj, mask, cfg["augmentation_rate"])
        losses = model.encoder_losses(adj, mask, augmented, aug_mask)
        loss = losses["invariance"] + cfg["orthogonality_weight"] * losses["orthogonality"]
        loss = loss + cfg["feature_weight"] * losses["feature"] + cfg["variance_weight"] * losses["variance"]
        return loss, losses
    w = torch.randn((len(adj), model.watermark_dim), device=adj.device)
    embedded = model.embed(adj, mask, w, alpha=cfg["alpha"], edit_budget=cfg["edit_budget"], straight_through=True)
    nll = model.flow_nll(embedded["z_node"], embedded["logdet"], mask)
    losses = {"nll": nll, "edits": embedded["edit_counts"].float().mean()}
    if stage == "stage2":
        reconstruction = model.reconstruction_loss(adj, mask, embedded["hs"], embedded["hw"])
        recovered = model.latent(embedded["adj"], mask)
        # Target detached to avoid moving both endpoints to minimize the cycle.
        cycle = F.mse_loss(recovered, embedded["target_z_graph"].detach())
        losses.update(reconstruction=reconstruction, cycle=cycle)
        return reconstruction + cfg["nll_weight"] * nll + cfg["cycle_weight"] * cycle, losses
    rate = cfg["robustness_rates"][step % len(cfg["robustness_rates"])]
    attack = (edge_flip, node_delete)[step % 2]
    attack_seed = int(torch.randint(2**31 - 1, (), device=adj.device))
    generator = torch.Generator(device=adj.device).manual_seed(attack_seed)
    attacked, attacked_mask = attack(embedded["adj"], mask, rate, generator=generator)
    recovered = model.latent(attacked, attacked_mask)
    signal = recovered
    if cfg.get("robust_control_variate", False):
        generator.manual_seed(attack_seed)
        null_adj, null_mask = attack(adj, mask, rate, generator=generator)
        # The null branch is independent of the zero-mean key, hence its
        # expected score AND expected parameter gradient are zero. Keeping
        # both gradients removes common graph-dependent Monte Carlo noise.
        signal = signal - model.latent(null_adj, null_mask)
    robustness = -(signal * w).sum(-1).mean()
    augmented, aug_mask = edge_flip(adj, mask, cfg["augmentation_rate"])
    auxiliary = model.encoder_losses(adj, mask, augmented, aug_mask)
    aux = auxiliary["invariance"] + cfg["orthogonality_weight"] * auxiliary["orthogonality"]
    aux = aux + cfg["feature_weight"] * auxiliary["feature"] + cfg["variance_weight"] * auxiliary["variance"]
    losses.update(robustness=robustness, auxiliary=aux)
    return robustness + cfg["nll_weight"] * nll + cfg["stage3_aux_weight"] * aux, losses


def train(config, seed=0, device="cpu", resume=False):
    seed_everything(seed)
    cfg = config["training"]
    kind = config["model"].get("kind", "drgw")
    output = Path(config["output_dir"]) / kind / f"seed-{seed}"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint.pt"
    if checkpoint.exists() and not resume:
        raise FileExistsError(f"{checkpoint} exists; use --resume or a new output directory")
    banks = prepare_banks(config["data"])
    arrays = []
    for name in config["data"]["datasets"]:
        with np.load(banks[name]["train"]) as bank:
            arrays.append(bank["adj"])
    # Equal-sized banks and uniform sampling give balanced dataset exposure.
    data = torch.from_numpy(np.concatenate(arrays)).to(device=device, dtype=torch.float32)
    model = build_model(config["model"]).to(device)
    stages = [("naive", cfg["naive_steps"])] if kind == "naive" else [
        (name, cfg[f"{name}_steps"]) for name in ("stage1", "stage2", "stage3")]
    saved = torch.load(checkpoint, map_location=device, weights_only=True) if resume and checkpoint.exists() else None
    if saved and (saved["config_digest"] != config_digest(config) or saved["seed"] != seed):
        raise ValueError("Resume requires exactly the saved configuration and seed")
    if saved and saved.get("device_type", "cuda" if saved["rng_device"].numel() != saved["rng_cpu"].numel() else "cpu") != torch.device(device).type:
        raise ValueError("Resume requires the same device type to restore the random-number state")
    if saved:
        model.load_state_dict(saved["model"])
        torch.set_rng_state(saved["rng_cpu"].cpu())
        if str(device).startswith("cuda"):
            torch.cuda.set_rng_state(saved["rng_device"].cpu(), device)
    provenance = dict(config=config, config_digest=config_digest(config), seed=seed,
                      environment=environment(), sampling_banks={k: {s: str(p) for s,p in v.items()} for k,v in banks.items()},
                      duration_units="optimizer updates", status="running")
    json_write(output / "run.json", provenance)
    start = time.monotonic()
    elapsed_before = saved.get("elapsed_seconds", 0) if saved else 0
    global_step = saved.get("global_step", 0) if saved else 0
    log_path = output / "training.jsonl"
    if saved and log_path.exists():
        # Discard updates not present in the last durable checkpoint before
        # replaying them, so each logged global_step occurs at most once.
        retained = []
        for line in log_path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row["global_step"] <= global_step:
                retained.append(line)
        log_path.write_text("\n".join(retained) + ("\n" if retained else ""))
    log = log_path.open("a" if saved else "w", buffering=1)
    try:
        for stage_index, (stage, steps) in enumerate(stages):
            if saved and stage_index < saved["stage_index"]:
                continue
            model.zero_grad(set_to_none=True)
            for parameter in model.parameters():
                parameter.requires_grad_(True)
            if stage == "stage2":
                for parameter in model.encoder.parameters():
                    parameter.requires_grad_(False)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                         lr=cfg["lr"], weight_decay=cfg["weight_decay"], betas=(0.9, 0.999))
            first = 0
            if saved and stage_index == saved["stage_index"]:
                optimizer.load_state_dict(saved["optimizer"])
                first = saved["stage_step"]
            model.train()
            for step in range(first, steps):
                lr = cfg["min_lr"] + 0.5 * (cfg["lr"] - cfg["min_lr"]) * (1 + math.cos(math.pi * step / max(steps - 1, 1)))
                for group in optimizer.param_groups:
                    group["lr"] = lr
                indices = torch.randint(len(data), (cfg["batch_size"],), device=device)
                adj = data[indices]
                mask = torch.ones(adj.shape[:2], dtype=torch.bool, device=device)
                model.zero_grad(set_to_none=True)
                loss, parts = training_objective(model, adj, mask, stage, cfg, step)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite {stage} loss at step {step + 1}")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                global_step += 1
                elapsed = elapsed_before + time.monotonic() - start
                if (step + 1) % cfg["log_every"] == 0 or step + 1 == steps:
                    row = dict(stage=stage, stage_step=step + 1, global_step=global_step,
                               loss=float(loss.detach()), gradient_norm=float(norm), lr=lr,
                               elapsed_seconds=elapsed, **{k:float(v.detach()) for k,v in parts.items()})
                    log.write(json.dumps(row) + "\n")
                    print(f"{kind} seed={seed} {stage} {step+1}/{steps} loss={row['loss']:.5f} elapsed={elapsed:.1f}s", flush=True)
                if (step + 1) % cfg["checkpoint_every"] == 0 or step + 1 == steps:
                    payload = dict(format_version=1, config=config, config_digest=config_digest(config), seed=seed, device_type=torch.device(device).type,
                                   model=model.state_dict(), optimizer=optimizer.state_dict(), stage_index=stage_index,
                                   stage_step=step+1, global_step=global_step, elapsed_seconds=elapsed,
                                   rng_cpu=torch.get_rng_state(),
                                   rng_device=torch.cuda.get_rng_state(device) if str(device).startswith("cuda") else torch.get_rng_state())
                    temporary = checkpoint.with_suffix(".tmp")
                    torch.save(payload, temporary)
                    temporary.replace(checkpoint)
            saved = None
        provenance.update(status="complete", optimizer_updates=global_step,
                          elapsed_seconds=elapsed_before + time.monotonic() - start)
        json_write(output / "run.json", provenance)
    finally:
        log.close()
    return checkpoint
