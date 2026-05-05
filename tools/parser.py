import argparse
import random
import numpy as np
import torch
import os
import datetime
import yaml
from easydict import EasyDict
from .distributed import world_info_from_env, init_distributed_device
from .common_utils import create_logger, log_config_to_file
from pathlib import Path


def random_seed(seed=0, rank=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def _str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).lower().strip()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError("expected true or false, got %r" % (v,))


def _resolve_precision(requested_precision):
    """
    Auto-map precision by GPU model:
      - A100 -> amp_bf16
      - V100 -> fp16
    """
    if requested_precision != "auto":
        return requested_precision, None

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training, but no CUDA device is available.")

    dev_idx = torch.cuda.current_device()
    dev_name = torch.cuda.get_device_name(dev_idx)
    dev_name_upper = dev_name.upper()

    if "A100" in dev_name_upper:
        return "amp_bf16", f"auto precision: detected GPU {dev_name} -> amp_bf16"
    if "V100" in dev_name_upper:
        return "fp16", f"auto precision: detected GPU {dev_name} -> fp16"
    raise RuntimeError(
        f"Unsupported GPU model for --precision auto: {dev_name}. "
        "Only A100 and V100 are supported."
    )


def read_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data', help="dataset root path")
    parser.add_argument('--cfg_file', type=str, default=None, help='dataset configs', required=True)
    parser.add_argument('--pretrained_model_name_or_path', default=None, type=str, required=True, help="path to tokenizer")

    # local fusion
    parser.add_argument('--off_batch_task', action='store_true', default=False, help="whether all process is training same task")
    parser.add_argument(
        '--debug',
        action='store_true',
        help='调试模式（含 NaN 调试日志）；亦可设环境变量 DEBUG_NAN=1。见 tools/debug_nan.py',
    )
    parser.add_argument(
        '--debug_log_every',
        type=int,
        default=10,
        help='Debug 聚合日志打印间隔（每 N 个 rollout step 打印一次 stop_rate 和 step_path_len 分布）',
    )
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to .pt checkpoint to load after init. Omit to use pretrained LM weights only (Vicuna from_pretrained + LoRA if --use_lora).",
    )
    parser.add_argument("--from_scratch", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--val_batch_size", type=int, default=2)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--feat_dropout", type=float, default=0.4)
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--num_steps_per_epoch", type=int, default=-1)
    parser.add_argument("--gradient_accumulation_step", type=int, default=2)
    parser.add_argument(
        "--precision",
        choices=["auto", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
        default="auto",
        help="Floating point precision. auto: A100->amp_bf16, V100->fp16.",
    )
    parser.add_argument("--workers", type=int, default=0)

    # distributed training args
    parser.add_argument('--world_size', type=int, default=0, help='number of gpus')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )

    # Save checkpoints
    parser.add_argument('--output_dir', type=str, default=None, required=True, help="output logs and ckpts")
    parser.add_argument("--max_saved_checkpoints", type=int, default=0)
    parser.add_argument("--save_ckpt_per_epochs", type=int, default=10)
    parser.add_argument("--save_latest_states", action='store_true')
    parser.add_argument("--save_pred_results", action="store_true")
    parser.add_argument("--save_detail_results", action="store_true")
    parser.add_argument(
        "--log_trainable_params",
        action="store_true",
        help="Print every trainable parameter name during initialization. Off by default to keep logs compact.",
    )
    parser.add_argument(
        "--trainable_param_log_limit",
        type=int,
        default=30,
        help="Maximum number of watched trainable parameter names to print when detailed logging is off.",
    )

    # training
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"])
    parser.add_argument("--stage", type=str, required=True, choices=["pretrain", "multi"])
    parser.add_argument('--ignoreid', default=-100, type=int, help="criterion: ignore label")
    parser.add_argument('--enable_og', action='store_true', default=False, help="object grounding task")
    parser.add_argument("--enable_summarize", action="store_true", help="perform EQA or generate instructions")
    parser.add_argument("--enable_fgr2r", action="store_true", help="perform fgr2r for R2R")
    parser.add_argument("--gen_loss_coef", type=float, default=1.)
    parser.add_argument("--obj_loss_coef", type=float, default=1.)
    parser.add_argument("--teacher_forcing_coef", type=float, default=1.)
    parser.add_argument("--fuse_obj", action="store_true", help="whether fuse object features for REVERIE and SOON")
    parser.add_argument(
        "--use_lora",
        action="store_true",
        default=False,
        help="enable LoRA for the language model (default: off; requires --update_llm true)",
    )
    parser.add_argument(
        "--use_dora",
        action="store_true",
        default=False,
        help="enable DoRA on top of LoRA adapters (requires --use_lora and a PEFT version with use_dora support)",
    )
    parser.add_argument(
        "--update_llm",
        type=_str2bool,
        nargs="?",
        const=True,
        default=False,
        metavar="true|false",
        help=(
            "Train the LM or keep it frozen (default: false — LM frozen unless you enable training). "
            "false: freeze all LM weights; --use_lora is not allowed. "
            "true + --use_lora: backbone frozen, LoRA adapters trainable. "
            "true without --use_lora: full LM finetuning."
        ),
    )
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,v_proj",
        help="comma-separated module names for LoRA injection",
    )

    # datasets
    parser.add_argument("--multi_endpoints", type=int, default=1)
    parser.add_argument("--path_type", type=str, default="trusted_path", choices=["planner_path", "trusted_path"])
    parser.add_argument(
        "--train_datasets",
        type=str,
        default=None,
        nargs='+',
        help="Optional training dataset list for stage=multi (e.g., R2R SOON). If not set, uses cfg SOURCE for joint multi-task training.",
    )
    parser.add_argument(
        "--use_env_memory",
        action="store_true",
        help="enable environment topology memory updates",
    )

    # evaluation
    parser.add_argument('--test_datasets', type=str, default=None, nargs='+')
    parser.add_argument('--validation_split', type=str, default="val_unseen", help="validation split: val_seen, val_unseen, test")
    parser.add_argument("--do_sample", action="store_true", help="do_sample in evaluation")
    parser.add_argument("--temperature", type=float, default=1.)


    # others
    parser.add_argument(
        "--max_datapoints",
        default=None,
        type=int,
        help="The number of datapoints used for debug."
    )
    args = parser.parse_args()

    if os.environ.get("DEBUG_NAN", "").strip().lower() in ("1", "true", "yes", "on", "y"):
        args.debug = True

    if args.use_lora and not args.update_llm:
        parser.error(
            "--use_lora requires --update_llm true. When LLM updates are disabled, LoRA must not be enabled."
        )
    if args.use_dora and not args.use_lora:
        parser.error("--use_dora requires --use_lora because DoRA is configured through PEFT LoraConfig.")

    if args.resume_from_checkpoint is not None and not str(args.resume_from_checkpoint).strip():
        args.resume_from_checkpoint = None

    args.local_rank, args.rank, args.world_size = world_info_from_env()

    ###################### configurations #########################
    # single-gpu or multi-gpu
    device_id = init_distributed_device(args)
    resolved_precision, precision_note = _resolve_precision(args.precision)
    args.precision = resolved_precision
    global_cfg = EasyDict(yaml.safe_load(open(str(Path(args.cfg_file).resolve()))))

    args.data_dir = Path(args.data_dir).resolve()

    # off-line image features from Matterport3D
    args.image_feat_size = global_cfg.Feature.image_feat_size
    args.obj_feat_size = global_cfg.Feature.obj_feat_size

    ############# Configurations ###############
    args.angle_feat_size = global_cfg.Feature.angle_feat_size
    args.enc_full_graph = global_cfg.Model.enc_full_graph
    args.expert_policy = global_cfg.Model.expert_policy
    args.num_pano_layers = global_cfg.Model.num_pano_layers

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = Path(args.output_dir) / 'log.txt'

    logger = create_logger(log_file, rank=args.rank)
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)
    if precision_note is not None:
        logger.info(precision_note)
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(global_cfg, logger=logger)

    print(" + rank: {}, + device_id: {}".format(args.local_rank, device_id))
    print(f"Start running training on rank {args.rank}.")

    # Training resume only: do not auto-load latest_states in test mode (user expects pretrained-only eval unless explicit --resume_from_checkpoint).
    if (
        args.mode != "test"
        and args.resume_from_checkpoint is None
        and os.path.exists(os.path.join(args.output_dir, "latest_states.pt"))
    ):
        state_path = os.path.join(args.output_dir, "latest_states.pt")
        logger.info("Resume checkpoint from {}".format(state_path))
        args.resume_from_checkpoint = state_path

    if args.resume_from_checkpoint is not None and not os.path.isfile(args.resume_from_checkpoint):
        raise FileNotFoundError(
            "resume_from_checkpoint is missing or not a file: {}. "
            "Please provide a valid .pt checkpoint path.".format(
                os.path.abspath(args.resume_from_checkpoint)
            )
        )

    if args.rank == 0:
        logger.info("resume_from_checkpoint (resolved): {}".format(args.resume_from_checkpoint))

    return args, global_cfg, logger, device_id
