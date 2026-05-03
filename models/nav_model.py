import copy
import inspect
import torch
import collections
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from transformers.utils import logging
from .ops import pad_tensors_wgrad, gen_seq_masks
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from pathlib import Path
from .image_embedding import ImageEmbeddings
from .modified_lm import ModifiedOPTForCasualLM, ModifiedLlamaForCausalLM, TrieLogitsProcessor
from typing import Dict, List, Any

logging.set_verbosity_error()


def attach_generation_config(logger, lang_model):
    """Set generation_config from tokenizer only (do not use from_model_config: it mirrors PretrainedConfig and
    warns once tokenizer ids differ after resize_token_embeddings)."""
    try:
        from transformers import GenerationConfig
    except ImportError:
        return lang_model
    if not hasattr(lang_model, "tokenizer"):
        return lang_model
    tok = lang_model.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.unk_token_id
    gc = GenerationConfig(
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=pad_id,
    )
    if hasattr(gc, "_from_model_config"):
        try:
            gc._from_model_config = False
        except (AttributeError, TypeError):
            pass
    lang_model.generation_config = gc
    logger.info("Generation config attached to lang_model.")
    return lang_model


def build_gen_config(lang_model, **overrides):
    """Deep-copy model.generation_config and apply overrides for a single generate() call."""
    from transformers import GenerationConfig

    base = getattr(lang_model, "generation_config", None)
    if base is None:
        tok = lang_model.tokenizer
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.unk_token_id
        base = GenerationConfig(
            bos_token_id=tok.bos_token_id,
            eos_token_id=tok.eos_token_id,
            pad_token_id=pad_id,
        )
    else:
        base = copy.deepcopy(base)
    for k, v in overrides.items():
        if v is None or not hasattr(base, k):
            continue
        setattr(base, k, v)
    return base


# Kwargs forwarded into forward_3dqa that map to GenerationConfig fields
_GEN_KWARGS = frozenset(
    ("max_new_tokens", "do_sample", "temperature", "top_p", "top_k", "num_beams", "repetition_penalty")
)


def maybe_enable_lora(args, logger, lang_model):
    if not getattr(args, 'use_lora', False):
        return lang_model

    try:
        from peft import LoraConfig, get_peft_model
    except Exception as e:
        raise ImportError(
            'LoRA is enabled but `peft` is not installed. '
            'Please run `pip install peft` in your navillm environment.'
        ) from e

    target_modules = [m.strip() for m in args.lora_target_modules.split(',') if m.strip()]
    if not target_modules:
        raise ValueError('`--lora_target_modules` is empty. Please provide at least one target module.')

    use_dora = getattr(args, 'use_dora', False)
    lora_kwargs = dict(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias='none',
        target_modules=target_modules,
    )
    if use_dora:
        if 'use_dora' not in inspect.signature(LoraConfig).parameters:
            raise ImportError(
                'DoRA is enabled but this `peft` version does not support `LoraConfig(use_dora=...)`. '
                'Please upgrade peft, e.g. `pip install -U peft`.'
            )
        lora_kwargs['use_dora'] = True

    lora_cfg = LoraConfig(
        **lora_kwargs,
    )

    # Keep the custom Modified* wrapper intact and only adapt the transformer backbone.
    if hasattr(lang_model.model, 'decoder'):
        # OPT-style backbone
        lang_model.model.decoder = get_peft_model(lang_model.model.decoder, lora_cfg)
    else:
        # LLaMA-style backbone
        lang_model.model = get_peft_model(lang_model.model, lora_cfg)

    trainable_params = sum(p.numel() for p in lang_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in lang_model.parameters())
    logger.info(
        '%s enabled: r=%d alpha=%d dropout=%.3f target_modules=%s trainable=%.2fM / total=%.2fM',
        'DoRA' if use_dora else 'LoRA',
        args.lora_r,
        args.lora_alpha,
        args.lora_dropout,
        target_modules,
        trainable_params / 1e6,
        total_params / 1e6,
    )
    return lang_model


def configure_llm_training(args, logger, lang_model):
    """Set LM requires_grad from --update_llm and --use_lora.

    ``update_llm`` false: freeze all LM weights (no LM updates; typical for inference/eval-only semantics).
    ``update_llm`` true: LM is trainable — if ``use_lora``, backbone frozen and only LoRA adapters train;
    otherwise full LM finetuning.

    CLI enforces ``not (use_lora and not update_llm)`` in ``tools/parser.py``; the check below
    catches callers that construct ``args`` without the parser (e.g. tests).
    """
    update_llm = getattr(args, "update_llm", False)
    use_lora = getattr(args, "use_lora", False)
    use_dora = getattr(args, "use_dora", False)
    if use_lora and not update_llm:
        raise ValueError("use_lora requires update_llm true (same rule as argument parsing).")
    if use_dora and not use_lora:
        raise ValueError("use_dora requires use_lora true (same rule as argument parsing).")

    # 1) If LLM update is disabled, freeze all language-model parameters.
    if not update_llm:
        for param in lang_model.parameters():
            param.requires_grad = False
    # 2) If LLM update is enabled:
    #    - use_lora=True  -> LoRA-only finetuning
    #    - use_lora=False -> full LLM finetuning
    elif use_lora:
        for param in lang_model.parameters():
            param.requires_grad = False
        for name, param in lang_model.named_parameters():
            lname = name.lower()
            is_lora_param = "lora_" in lname or "lora_a" in lname or "lora_b" in lname
            is_dora_param = use_dora and ("lora_magnitude" in lname or "magnitude_vector" in lname)
            if is_lora_param or is_dora_param:
                param.requires_grad = True
    else:
        for param in lang_model.parameters():
            param.requires_grad = True

    trainable_params = sum(p.numel() for p in lang_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in lang_model.parameters())
    if not update_llm:
        llm_mode = "frozen"
    elif use_lora:
        llm_mode = "dora_only" if use_dora else "lora_only"
    else:
        llm_mode = "full_finetune"
    logger.info(
        "LLM training mode: %s (update_llm=%s, use_lora=%s, use_dora=%s): trainable=%.2fM / total=%.2fM",
        llm_mode,
        update_llm,
        use_lora,
        use_dora,
        trainable_params / 1e6,
        total_params / 1e6,
    )
    return lang_model


def init_vis_config(args, config):
    navillm_root = Path(__file__).resolve().parents[1]
    local_bert_cfg = navillm_root / 'data' / 'models' / 'bert-large-uncased'
    cfg_file = local_bert_cfg / 'config.json'
    if not cfg_file.exists():
        raise FileNotFoundError(
            f'Missing local BERT config: {cfg_file}. '
            'Please download bert-large-uncased config.json to this path.'
        )
    vis_config = PretrainedConfig.from_pretrained(str(local_bert_cfg), local_files_only=True)
    vis_config.num_pano_layers = config.num_pano_layers
    vis_config.precision = args.precision
    vis_config.pretrained_model_name_or_path = args.pretrained_model_name_or_path
    vis_config.max_action_steps = 100
    vis_config.image_feat_size = args.image_feat_size
    vis_config.angle_feat_size = args.angle_feat_size
    vis_config.obj_feat_size = args.obj_feat_size
    vis_config.obj_loc_size = 3
    vis_config.type_vocab_size = 3
    return vis_config


class NavModel(nn.Module):
    def __init__(self, args, logger, model_config):
        super().__init__()
        self.args = args
        config = init_vis_config(args, model_config)
        self.config = config

        # Large Language Model
        # Preferred loading flow for LoRA finetuning:
        # 1) load original LLM backbone from pretrained weights
        # 2) inject LoRA adapters
        # 3) optionally load task checkpoint (non-LLM parts and any matching keys)
        # Use config-based random init only when explicitly requested by --from_scratch.
        if args.from_scratch:
            logger.info("Initialize the model from config (--from_scratch).")
            model_config = AutoConfig.from_pretrained(config.pretrained_model_name_or_path)
            self.lang_model = ModifiedOPTForCasualLM(model_config, config) if 'opt' in config.pretrained_model_name_or_path \
                else ModifiedLlamaForCausalLM(model_config, config)
        else:
            logger.info("Initialize the model from pretrained backbone.")
            self.lang_model = ModifiedOPTForCasualLM.from_pretrained(config.pretrained_model_name_or_path, config) if "opt" in config.pretrained_model_name_or_path \
                else ModifiedLlamaForCausalLM.from_pretrained(config.pretrained_model_name_or_path, config)
        
        self.lang_model.init_tokenizer(config.pretrained_model_name_or_path)
        self.lang_model = maybe_enable_lora(args, logger, self.lang_model)
        self.lang_model = configure_llm_training(args, logger, self.lang_model)
        self.lang_model = attach_generation_config(logger, self.lang_model)

        self.hidden_size = self.lang_model.hidden_size
        self.model_type = self.lang_model.model_type

        # Panorama Encoding
        config.output_size = self.hidden_size
        self.img_embeddings = ImageEmbeddings(config, use_obj=args.enable_og, fuse_obj=args.fuse_obj)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, self.hidden_size)

        # global encoding
        self.gmap_pos_embeddings = nn.Sequential(
            nn.Linear(config.angle_feat_size + 3, self.hidden_size),
            nn.LayerNorm(self.hidden_size, eps=1e-12)
        )
        self.gmap_step_embeddings = nn.Embedding(config.max_action_steps, self.hidden_size)

        # local encoding
        self.vp_pos_embeddings = nn.Sequential(
            nn.Linear(config.angle_feat_size * 2 + 6, self.hidden_size),
            nn.LayerNorm(self.hidden_size, eps=1e-12)
        )

        self.obj_pos_embeddings = nn.Sequential(
            nn.Linear(config.angle_feat_size + 3, self.hidden_size),
            nn.LayerNorm(self.hidden_size, eps=1e-12)
        )

        if self.config.obj_feat_size > 0:
            self.og_head = nn.Sequential(
                nn.Linear(self.hidden_size, 100)
            ).to(self.lang_model.model_type) 

        # Classfification from candidates
        self.out_head = nn.Sequential(
            nn.Linear(self.hidden_size, 100)
        ).to(torch.float32)

        self.instruction = None
        self.history = None
        self.hist_vis = None

        self.drop_env = nn.Dropout(p=args.feat_dropout)

        logger.info("model type: {}".format(self.model_type))


    def forward(self, mode: str, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        batch = collections.defaultdict(lambda: None, batch)

        if mode == 'panorama':  # batch['view_img_fts'] [B, 36, D=768] --> dropout
            batch['view_img_fts'] = self.drop_env(batch['view_img_fts'])
            if 'obj_img_fts' in batch:
                batch['obj_img_fts'] = self.drop_env(batch['obj_img_fts'])
            return self.img_embeddings.forward_panorama_per_step(
                batch['view_img_fts'],
                batch['view_lens'],
                batch['loc_fts'],
                batch['nav_types'],
                batch['obj_img_fts'],
                batch['obj_lens'],
                batch['obj_loc_fts'],
            )

        elif mode == 'navigation':
            return self.forward_navigation(mode, batch, **kwargs)

        elif mode == "summarization" or mode == 'embodied_qa':
            return self.forward_summarization(mode, batch, **kwargs)

        elif mode == "3dqa":
            return self.forward_3dqa(mode, batch, **kwargs)
        
        elif mode == 'object_grounding':
            return self.forward_object_grounding(mode, batch, **kwargs)

        else:
            raise NotImplementedError('wrong mode: %s' % mode)
    

    def forward_navigation(
        self, 
        mode, 
        batch: Dict[str, Any], 
        training: bool=True, 
        **kwargs
    ) -> Dict[str, Any]:

        data_type = batch['data_type']
        vp_img_embeds = batch['vp_img_embeds']
        batch_size = vp_img_embeds.size(0)
        gmap_img_embeds, gmap_step_ids, gmap_pos_fts, \
            gmap_masks, gmap_pair_dists, gmap_visited_masks, gmap_vpids \
            = batch['gmap_img_embeds'], batch['gmap_step_ids'], batch['gmap_pos_fts'], \
            batch['gmap_masks'], batch['gmap_pair_dists'], batch['gmap_visited_masks'], batch['gmap_vpids'],

        # global branch [B, Nums, D=768]
        gmap_embeds = torch.zeros_like(gmap_img_embeds)
        for b_ix in range(len(data_type)):
                gmap_embeds[b_ix:b_ix + 1] = gmap_img_embeds[b_ix:b_ix + 1] + \
                                                self.gmap_step_embeddings(gmap_step_ids[b_ix:b_ix + 1]) + \
                                                self.gmap_pos_embeddings(gmap_pos_fts[b_ix:b_ix + 1])


        ##### local branch #####
        vp_img_embeds, vp_pos_fts, vp_nav_masks, vp_cand_vpids = \
            batch['vp_img_embeds'], batch['vp_pos_fts'], batch['vp_nav_masks'], batch['vp_cand_vpids']

        pano_masks = batch['pano_masks']

        vp_embeds = torch.zeros_like(vp_img_embeds)
        for b_ix in range(len(data_type)):
            vp_embeds[b_ix:b_ix + 1] = vp_img_embeds[b_ix:b_ix + 1] \
                                        + self.vp_pos_embeddings(vp_pos_fts[b_ix:b_ix + 1])

        ##### fuse embeds #####
        gmap_embeds.masked_fill_(gmap_visited_masks.unsqueeze(-1), 0.)
        gmap_embeds.masked_fill_(gmap_masks.logical_not().unsqueeze(-1), 0.)
        cand_token_type_ids = torch.zeros((gmap_embeds.shape[0], gmap_embeds.shape[1])).int().to(gmap_embeds.device)

        local_vp_embeds = vp_embeds
        local_vp_embeds.masked_fill_(pano_masks.logical_not().unsqueeze(-1), 0.)

        fuse_embeds = torch.clone(gmap_embeds)

        for i in range(batch_size):
            visited_nodes = set([vp for vp, mask in zip(gmap_vpids[i], gmap_visited_masks[i]) if mask])
            tmp = {}
            bw_logits = 0
            for j, cand_vpid in enumerate(vp_cand_vpids[i]):
                if j > 0:
                    if cand_vpid in visited_nodes:
                        bw_logits += local_vp_embeds[i, j]
                    else:
                        tmp[cand_vpid] = local_vp_embeds[i, j]
            for j, vp in enumerate(gmap_vpids[i]):
                if j > 0 and vp not in visited_nodes:
                    if vp in tmp:
                        fuse_embeds[i, j] += tmp[vp]
                    else:
                        # fuse_embeds[i, j] += bw_logits
                        cand_token_type_ids[i, j] = 1

        fuse_embeds += self.token_type_embeddings(cand_token_type_ids).to(fuse_embeds.device)
        fuse_embeds.masked_fill_(gmap_visited_masks.unsqueeze(-1), 0.)
        fuse_embeds.masked_fill_(gmap_masks.logical_not().unsqueeze(-1), 0.)

        cand_masks = torch.clone(gmap_masks & gmap_visited_masks.logical_not())
        cand_nums = cand_masks.sum(dim=-1)
        instruction = batch['instruction']
        history = batch['history']
        hist_vis = batch['hist_vis']
        hist_vis_input = []
        for vis in hist_vis:
            hist_vis_input.extend(vis)
        if hist_vis_input != []:
            hist_vis_input = torch.stack(hist_vis_input, dim=0)
        else:
            hist_vis_input = None

        hist_nums = [len(his) for his in history]

        text_input = self.lang_model.tokenize(batch["prompts"]).to(fuse_embeds.device)

        # cand_embeds = fuse_embeds[cand_masks]  # .to(self.model_type)
        cand_embeds = []
        inv_perms = []
        for bn in range(batch_size):
            # random permute
            cand_embed = fuse_embeds[bn][cand_masks[bn]][1:]
            rand_perm = torch.randperm(cand_embed.shape[0])
            inv_perm = torch.arange(cand_embed.shape[0])
            inv_perm[rand_perm] = torch.arange(cand_embed.shape[0])
            inv_perms.append(inv_perm)
            cand_embeds.append(cand_embed[rand_perm]) # remove stop features
        cand_embeds = torch.cat(cand_embeds, dim=0)

        output = self.lang_model(
            input_ids=text_input['input_ids'],
            attention_mask=text_input['attention_mask'],
            cand_vis=cand_embeds,
            hist_vis=hist_vis_input,
        )
        loss, hidden_states = output.loss, output.hidden_states

        fuse_logits = torch.zeros((fuse_embeds.shape[0], fuse_embeds.shape[1])).to(
            fuse_embeds.device).to(self.model_type)
        
        # Keep classification head in fp32 to avoid first-step fp16 overflow.
        predictions = self.out_head(
            hidden_states[text_input['input_ids']==self.lang_model.cls_token_id[0]].float()
        ).to(self.model_type)
        
        for i in range(batch_size):
            fuse_logits[i][cand_masks[i]] = torch.cat([predictions[i, 0:1],predictions[i, 1:cand_nums[i]][inv_perms[i]]],dim=0)
        # 原来：fuse_logits.masked_fill_(cand_masks.logical_not(), -float('inf'))
        # 现：有限负值避免 softmax / Categorical(probs 路径) 在整行无效时出现 NaN（尤其 fp16）。
        _neg = torch.finfo(fuse_logits.dtype).min
        fuse_logits.nan_to_num_(nan=_neg, posinf=_neg, neginf=_neg)
        fuse_logits.masked_fill_(cand_masks.logical_not(), _neg)

        return {
            'fuse_embeds': fuse_embeds.detach(),
            'fuse_logits': fuse_logits,
        }

    

    def forward_summarization(
        self, 
        mode, 
        batch: Dict[str, Any], 
        training: bool=True, 
        **kwargs
    ) -> Dict[str, Any]:

        vp_img_embeds = batch['vp_img_embeds']
        batch_size = vp_img_embeds.size(0)
        vp_img_embeds, vp_pos_fts, \
            vp_nav_masks, vp_cand_vpids = \
            batch['vp_img_embeds'], batch['vp_pos_fts'], \
                batch['vp_nav_masks'], batch['vp_cand_vpids']
        
        # remove `stop`
        vp_img_embeds = vp_img_embeds[:, 1:, :]
        vp_nav_masks = vp_nav_masks[:, 1:]

        vp_pos_fts = torch.zeros(vp_img_embeds.shape[:2]+(14,), dtype=torch.float).to(vp_img_embeds.device)
        token_type_ids = torch.zeros(vp_img_embeds.shape[:2], dtype=torch.int).to(vp_img_embeds.device)
        vp_img_embeds += self.vp_pos_embeddings(vp_pos_fts)
        vp_img_embeds += self.token_type_embeddings(token_type_ids)

        instruction = batch['instruction']
        labels = batch['answer']
        history = batch['history']
        hist_vis = batch['hist_vis']
        data_type = batch['data_type']
        hist_vis_input = []

        for vis in hist_vis:
            hist_vis_input.extend(vis)
        if hist_vis_input != []:
            hist_vis_input = torch.stack(hist_vis_input, dim=0)
        else:
            hist_vis_input = None

        hist_nums = [len(his) for his in history]
        cand_nums = vp_nav_masks.sum(1)
        
        all_text = []

        for bn in range(batch_size):
            prompt = batch["prompts"][bn]
            if data_type[0] == 'eqa' or data_type[0] == 'fgr2r':
                label = labels[bn] + f"{self.lang_model.tokenizer.eos_token}"
            else:
                label = batch["instruction"][bn] + f"{self.lang_model.tokenizer.eos_token}"
            if training:
                all_text.append([prompt, label])
            else:
                all_text.append(prompt)

        text_input = self.lang_model.tokenize(all_text).to(vp_img_embeds.device)
        if training:
            labels = text_input['input_ids'].clone()
            labels[text_input['token_type_ids'][:, -labels.shape[-1]:] == 0] = -100
            outputs = self.lang_model(
                input_ids=text_input['input_ids'],
                attention_mask=text_input['attention_mask'],
                labels=labels,
                cand_vis=vp_img_embeds[vp_nav_masks],
                hist_vis=hist_vis_input,
            )
            loss, logits, hidden_states = outputs.loss, outputs.logits, outputs.hidden_states
            outputs = {
                "loss": loss
            }
        else:
            trie = kwargs.get('trie', None)
            logits_processor = [TrieLogitsProcessor(trie)] if trie is not None else []

            gen_cfg = build_gen_config(self.lang_model, max_new_tokens=50, do_sample=False)
            generate_ids = self.lang_model.generate(
                input_ids=text_input['input_ids'],
                attention_mask=text_input['attention_mask'],
                cand_vis=vp_img_embeds[vp_nav_masks],
                hist_vis=hist_vis_input,
                generation_config=gen_cfg,
                logits_processor=logits_processor,
            ).tolist()

            generate_ids = [s[text_input["input_ids"].shape[1]:] for i, s in enumerate(generate_ids)]
            generated_sentences = self.lang_model.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            outputs = {
                "generated_sentences": generated_sentences
            }

        return outputs
        

    def forward_3dqa(
        self, 
        mode, 
        batch: Dict[str, Any], 
        training: bool=True, 
        **kwargs
    ) -> Dict[str, Any]:
        batch_size = len(batch['question'])
        data_type = batch['data_type']
        all_text = []
        for bn in range(batch_size):
            prompt = batch["prompts"][bn]
            if training:
                ans = batch["answers"][bn][0]+ f"{self.lang_model.tokenizer.eos_token}"
                all_text.append([prompt, ans])
            else:
                all_text.append(prompt)
        
        view_img_fts = pad_tensors_wgrad([batch["features"][bn] for bn in range(batch_size)])
        view_lens = torch.tensor([batch["features"][bn].shape[0] for bn in range(batch_size)]).to(view_img_fts.device)
        pano_outputs = self.img_embeddings.forward_panorama_per_step(
            view_img_fts=view_img_fts,
            view_lens=view_lens,
        )
        pano_embeds, pano_masks = pano_outputs["pano_embeds"], pano_outputs["pano_masks"]
        vp_pos_fts = torch.zeros(pano_embeds.shape[:2]+(14,), dtype=torch.float).to(pano_embeds.device)
        token_type_ids = torch.zeros(pano_embeds.shape[:2], dtype=torch.int).to(pano_embeds.device)
        pano_embeds += self.vp_pos_embeddings(vp_pos_fts)
        pano_embeds += self.token_type_embeddings(token_type_ids)

        text_input = self.lang_model.tokenize(all_text).to(pano_embeds.device)
        if training:
            labels = text_input['input_ids'].clone()
            labels[text_input['token_type_ids'][:, -labels.shape[-1]:] == 0] = -100
            outputs = self.lang_model(
                input_ids=text_input['input_ids'],
                attention_mask=text_input['attention_mask'],
                labels=labels,
                cand_vis=pano_embeds[pano_masks],
            )
        else:
            gen_overrides = {k: kwargs[k] for k in _GEN_KWARGS if k in kwargs}
            gen_cfg = build_gen_config(self.lang_model, **gen_overrides)
            generate_ids = self.lang_model.generate(
                input_ids=text_input['input_ids'],
                attention_mask=text_input['attention_mask'],
                cand_vis=pano_embeds[pano_masks],
                generation_config=gen_cfg,
            ).tolist()

            generate_ids = [s[text_input["input_ids"].shape[1]:] for i, s in enumerate(generate_ids)]
            generated_sentences = self.lang_model.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            outputs = {
                "generated_sentences": generated_sentences
            }

        return outputs


    def forward_object_grounding(
        self, 
        mode, 
        batch: Dict[str, Any], 
        training: bool=True, 
        **kwargs
    ) -> Dict[str, Any]:

        data_type = batch['data_type']
        obj_embeds, obj_masks, obj_loc_fts = batch['obj_embeds'], batch['obj_masks'], batch['obj_loc_fts']

        batch_size = obj_embeds.size(0)
        obj_embeds = obj_embeds + self.obj_pos_embeddings(obj_loc_fts)

        cand_nums = obj_masks.sum(dim=1) + 1    # add not exist

        instruction = batch['instruction']
        history = batch['history']
        hist_vis = batch['hist_vis']
        hist_vis_input = []
        for vis in hist_vis:
            hist_vis_input.extend(vis)
        if hist_vis_input != []:
            hist_vis_input = torch.stack(hist_vis_input, dim=0)
        else:
            hist_vis_input = None

        hist_nums = [len(his) for his in history]

        text_input = self.lang_model.tokenize(batch["prompts"]).to(obj_embeds.device)
        output = self.lang_model(
            input_ids=text_input['input_ids'],
            attention_mask=text_input['attention_mask'],
            cand_vis=obj_embeds[obj_masks],
            hist_vis=hist_vis_input,
        )
        loss, hidden_states = output.loss, output.hidden_states

        predictions = self.out_head(
            hidden_states[text_input['input_ids']==self.lang_model.cls_token_id[0]].float()
        ).to(self.model_type)
        for i in range(batch_size):
            predictions[i, cand_nums[i]:] = float('-inf')

        return {
            'obj_logits': predictions
        }
