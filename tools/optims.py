import inspect
import os
import torch
import glob
from transformers import get_constant_schedule_with_warmup


def _torch_load_checkpoint(path, map_location="cpu"):
    """Load training checkpoint; explicit weights_only=False silences PyTorch 2.4+ FutureWarning on default."""
    kw = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kw["weights_only"] = False
    return torch.load(path, **kw)


def _remap_lora_checkpoint_key(key: str) -> str:
    """
    Remap legacy non-LoRA LLM keys to PEFT-LoRA wrapped key space.
    Example:
      lang_model.model.layers.0.self_attn.q_proj.weight
        -> lang_model.model.base_model.model.layers.0.self_attn.q_proj.base_layer.weight
    """
    new_key = key

    # LLaMA backbone path wrapped by PEFT: lang_model.model -> lang_model.model.base_model.model
    if new_key.startswith("lang_model.model.") and not new_key.startswith("lang_model.model.base_model.model."):
        suffix = new_key[len("lang_model.model."):]
        if (
            suffix.startswith("layers.")
            or suffix.startswith("embed_tokens.")
            or suffix.startswith("norm.")
        ):
            new_key = "lang_model.model.base_model.model." + suffix

    # For LoRA-targeted modules, original base weights are nested under `base_layer`.
    if new_key.endswith(".self_attn.q_proj.weight"):
        new_key = new_key[:-len(".weight")] + ".base_layer.weight"
    elif new_key.endswith(".self_attn.v_proj.weight"):
        new_key = new_key[:-len(".weight")] + ".base_layer.weight"

    return new_key


def check_checkpoint(args, model, optimizer, lr_scheduler, logger) -> int:
    resume_from_epoch = 0
    if args.resume_from_checkpoint is not None:
        ckpt = args.resume_from_checkpoint
        if not os.path.isfile(ckpt):
            if args.rank == 0:
                logger.warning(
                    "Checkpoint not found (%s); skipping load (pretrained weights only).",
                    os.path.abspath(ckpt),
                )
            return resume_from_epoch
        if args.rank == 0:
            logger.info(f"Loading checkpoint from {ckpt}")
        checkpoint = _torch_load_checkpoint(ckpt)
        model_state_dict = model.state_dict()
        state_disk_raw = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
        if getattr(args, "use_lora", False):
            state_disk = {}
            remapped_cnt = 0
            for k, v in state_disk_raw.items():
                nk = _remap_lora_checkpoint_key(k)
                if nk != k:
                    remapped_cnt += 1
                # Prefer remapped key; if collision happens, keep first one.
                if nk not in state_disk:
                    state_disk[nk] = v
            logger.info("LoRA checkpoint key remap applied: %d keys remapped", remapped_cnt)
        else:
            state_disk = state_disk_raw
        update_model_state = {}
        ignored_keys = []
        for key, val in state_disk.items():
            if key in model_state_dict and model_state_dict[key].shape == val.shape:
                update_model_state[key] = val
            else:
                ignored_keys.append((key, tuple(val.shape)))
        msg = model.load_state_dict(update_model_state, strict=False)
        missing_keys = list(getattr(msg, "missing_keys", []))
        unexpected_keys = list(getattr(msg, "unexpected_keys", []))
        expected_missing_lora = []
        if getattr(args, "use_lora", False):
            non_lora_missing = []
            for k in missing_keys:
                lk = k.lower()
                if ("lora_a" in lk) or ("lora_b" in lk):
                    expected_missing_lora.append(k)
                else:
                    non_lora_missing.append(k)
            missing_keys = non_lora_missing
        logger.info(
            "Checkpoint load summary: matched=%d, ignored=%d, missing=%d, expected_missing_lora=%d, unexpected=%d",
            len(update_model_state),
            len(ignored_keys),
            len(missing_keys),
            len(expected_missing_lora),
            len(unexpected_keys),
        )
        if ignored_keys:
            logger.info("Ignored keys (shape mismatch or absent): %s", ignored_keys)
        if missing_keys:
            logger.info("Missing keys: %s", missing_keys)
        if expected_missing_lora:
            logger.info("Expected missing LoRA keys: %d", len(expected_missing_lora))
        if unexpected_keys:
            logger.info("Unexpected keys: %s", unexpected_keys)

        if 'epoch' in checkpoint:
            resume_from_epoch = checkpoint['epoch'] + 1
            logger.info("Resume from Epoch {}".format(resume_from_epoch))
            optimizer.load_state_dict(checkpoint['optimizer'])


    return resume_from_epoch


def _log_trainable_parameters(model, logger):
    trainable = [
        (name, tuple(param.shape), param.numel())
        for name, param in model.named_parameters()
        if param.requires_grad
    ]
    total = sum(numel for _, _, numel in trainable)

    group_sums = {}
    lang_model_non_lora = []
    watched_keywords = ("projector", "vision", "embed", "lm_head")
    watched_trainable = []

    for name, shape, numel in trainable:
        lname = name.lower()
        if name.startswith("lang_model."):
            group = "lang_model.lora" if "lora_" in lname else "lang_model.non_lora"
            if "lora_" not in lname:
                lang_model_non_lora.append((name, shape, numel))
        else:
            group = name.split(".", 1)[0]

        group_sums[group] = group_sums.get(group, 0) + numel
        if any(keyword in lname for keyword in watched_keywords):
            watched_trainable.append((name, shape, numel))

    logger.info(
        "Trainable parameter detail: tensors=%d params=%.2fM",
        len(trainable),
        total / 1e6,
    )
    for group, numel in sorted(group_sums.items()):
        logger.info("Trainable group: %s %.2fM", group, numel / 1e6)

    if lang_model_non_lora:
        logger.warning(
            "Found %d trainable non-LoRA language-model tensors; expected zero for LoRA-only LLM training.",
            len(lang_model_non_lora),
        )
    if watched_trainable:
        logger.info("Watched trainable tensors matching projector/vision/embed/lm_head:")
        for name, shape, numel in watched_trainable:
            logger.info("WATCH_TRAINABLE %s shape=%s params=%.4fM", name, shape, numel / 1e6)

    for name, shape, numel in trainable:
        logger.info("TRAINABLE_PARAM %s shape=%s params=%.4fM", name, shape, numel / 1e6)


def dist_models(args, model, logger):
    logger.info("*************** init model *************** ")
    # args.rank: global rank.
    total_gpus = torch.cuda.device_count()
    device_id = args.rank % total_gpus

    model.to(device_id)
    
    optimizer = torch.optim.AdamW([p for n, p in model.named_parameters() if p.requires_grad], lr=args.lr)

    lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps)

    resume_from_epoch = check_checkpoint(
        args, model, optimizer, lr_scheduler, logger,
    )
    param_sums = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("model initialized with {:.2f} M trainable parameters".format(param_sums/1000**2))
    if getattr(args, "rank", 0) == 0:
        _log_trainable_parameters(model, logger)
    if args.distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[device_id], find_unused_parameters=True)

        # args.batch_size: BATCH_SIZE_PER_GPU
        logger.info('Training in distributed mode : total_batch_size: %d' % (total_gpus * args.batch_size))
    else:
        total_gpus = 1
        logger.info('Training with a single process')

    return model, optimizer, resume_from_epoch, lr_scheduler


def save_checkpoint(model, model_path, optimizer=None, epoch: int=0, save_states: bool=False):
    if hasattr(model, 'module'):
        model = model.module
    
    state_dict = {
        "model_state_dict": model.state_dict()
    }
    if save_states:
        state_dict.update({
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        })

    torch.save(state_dict, model_path)