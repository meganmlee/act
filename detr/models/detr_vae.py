# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes — modified for FAST action tokenization.

When use_fast_tokens=True:
  - CVAE encoder embeds discrete action tokens via nn.Embedding
  - CVAE decoder predicts token logits (vocab_size) instead of continuous actions
  - num_queries = max_token_len (number of token positions to predict)
"""
import torch
from torch import nn
from torch.autograd import Variable
from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer

import numpy as np

import IPython
e = IPython.embed


class AutoregressiveTokenHead(nn.Module):
    """
    Causal transformer decoder that generates BPE tokens left-to-right,
    cross-attending to the DETR decoder's context sequence as memory.

    Training: teacher-forced (shift targets right, prepend BOS).
    Inference: greedy decoding with early stopping on pad token.
    """
    def __init__(self, hidden_dim, vocab_size, max_token_len, bos_token_id, pad_token_id,
                 num_layers=4, nhead=8):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_token_len = max_token_len
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id

        # +1 to accommodate BOS token id (= vocab_size)
        self.token_embed = nn.Embedding(vocab_size + 1, hidden_dim)
        self.pos_embed = nn.Embedding(max_token_len + 1, hidden_dim)  # +1 for BOS position

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # pre-norm
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        # Predict over vocab_size tokens (includes pad as stop signal)
        self.out_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, memory, targets=None):
        """
        memory:  (B, context_len, hidden_dim) — DETR decoder context
        targets: (B, max_token_len) LongTensor — ground-truth tokens, None at inference

        Returns (training):  logits (B, max_token_len, vocab_size)
        Returns (inference): tokens (B, max_token_len), token_lens (B,)
        """
        B = memory.shape[0]
        device = memory.device

        if targets is not None:
            # Teacher-forced: prepend BOS, shift targets right
            bos = torch.full((B, 1), self.bos_token_id, dtype=torch.long, device=device)
            inp = torch.cat([bos, targets[:, :-1]], dim=1)  # (B, max_token_len)
            S = inp.shape[1]
            positions = torch.arange(S, device=device).unsqueeze(0)
            x = self.token_embed(inp) + self.pos_embed(positions)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(S, device=device)
            out = self.decoder(x, memory, tgt_mask=causal_mask)
            out = self.norm(out)
            return self.out_proj(out)  # (B, max_token_len, vocab_size)

        else:
            # Greedy decoding with early stopping on pad token
            generated = torch.full((B, 1), self.bos_token_id, dtype=torch.long, device=device)
            token_lens = torch.full((B,), self.max_token_len, dtype=torch.long, device=device)
            done = torch.zeros(B, dtype=torch.bool, device=device)

            for step in range(self.max_token_len):
                S = generated.shape[1]
                positions = torch.arange(S, device=device).unsqueeze(0)
                x = self.token_embed(generated) + self.pos_embed(positions)
                causal_mask = nn.Transformer.generate_square_subsequent_mask(S, device=device)
                out = self.decoder(x, memory, tgt_mask=causal_mask)
                out = self.norm(out)
                next_token = self.out_proj(out)[:, -1, :].argmax(dim=-1)  # (B,)

                newly_done = (~done) & (next_token == self.pad_token_id)
                token_lens[newly_done] = step
                done = done | (next_token == self.pad_token_id)
                generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

                if done.all():
                    break

            tokens = generated[:, 1:]  # Remove BOS
            # Pad or truncate to max_token_len
            if tokens.shape[1] < self.max_token_len:
                pad = torch.full((B, self.max_token_len - tokens.shape[1]),
                                 self.pad_token_id, dtype=torch.long, device=device)
                tokens = torch.cat([tokens, pad], dim=1)
            else:
                tokens = tokens[:, :self.max_token_len]

            return tokens, token_lens


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DETRVAE(nn.Module):
    def __init__(self, backbones, transformer, encoder, state_dim, action_dim,
                 num_queries, camera_names, use_fast_tokens=False,
                 vocab_size=None, max_token_len=None, pad_token_id=None):
        """
        Args:
            use_fast_tokens: if True, encoder/decoder work in discrete token space
            vocab_size: FAST vocabulary size + 1 (for pad token)
            max_token_len: max token sequence length for the AR head
            pad_token_id: token ID used for padding (= vocab_size - 1)
        """
        super().__init__()
        self.use_fast_tokens = use_fast_tokens
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model

        if use_fast_tokens:
            assert vocab_size is not None and max_token_len is not None and pad_token_id is not None
            self.vocab_size = vocab_size
            self.max_token_len = max_token_len
            self.pad_token_id = pad_token_id
            # Parallel prediction: one query per token position
            self.num_queries = max_token_len
            self.query_embed = nn.Embedding(max_token_len, hidden_dim)
            self.action_head = nn.Linear(hidden_dim, vocab_size)
            self.is_pad_head = nn.Linear(hidden_dim, 1)
        else:
            self.num_queries = num_queries
            self.action_head = nn.Linear(hidden_dim, action_dim)
            self.is_pad_head = nn.Linear(hidden_dim, 1)
            self.query_embed = nn.Embedding(num_queries, hidden_dim)

        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        else:
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # encoder extra parameters
        self.latent_dim = 32
        self.cls_embed = nn.Embedding(1, hidden_dim)

        if use_fast_tokens:
            self.encoder_action_embed = nn.Embedding(vocab_size, hidden_dim)
            encoder_seq_len = 1 + 1 + max_token_len
        else:
            self.encoder_action_proj = nn.Linear(action_dim, hidden_dim)
            encoder_seq_len = 1 + 1 + num_queries

        self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)
        self.register_buffer('pos_table', get_sinusoid_encoding_table(encoder_seq_len, hidden_dim))

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)

    def forward(self, qpos, image, env_state, actions=None, is_pad=None):
        """
        qpos: (batch, state_dim)
        image: (batch, num_cam, C, H, W)
        env_state: None
        actions: if use_fast_tokens: (batch, max_token_len) LongTensor
                 else: (batch, seq, action_dim) float
        is_pad: (batch, seq) bool
        """
        is_training = actions is not None
        bs, _ = qpos.shape

        if is_training:
            if self.use_fast_tokens:
                action_embed = self.encoder_action_embed(actions)
            else:
                action_embed = self.encoder_action_proj(actions)

            qpos_embed = self.encoder_joint_proj(qpos)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)
            cls_embed = self.cls_embed.weight
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1)
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1)
            encoder_input = encoder_input.permute(1, 0, 2)

            cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device)
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)

            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)

            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0]
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)

        if self.backbones is not None:
            all_cam_features = []
            all_cam_pos = []
            for cam_id, cam_name in enumerate(self.camera_names):
                features, pos = self.backbones[0](image[:, cam_id])
                features = features[0]
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)
            proprio_input = self.input_proj_robot_state(qpos)
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)
            hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight)[0]
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1)
            hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)[0]

        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar]


class CNNMLP(nn.Module):
    def __init__(self, backbones, state_dim, action_dim, camera_names):
        super().__init__()
        self.camera_names = camera_names
        self.action_head = nn.Linear(1000, action_dim)
        if backbones is not None:
            self.backbones = nn.ModuleList(backbones)
            backbone_down_projs = []
            for backbone in backbones:
                down_proj = nn.Sequential(
                    nn.Conv2d(backbone.num_channels, 128, kernel_size=5),
                    nn.Conv2d(128, 64, kernel_size=5),
                    nn.Conv2d(64, 32, kernel_size=5)
                )
                backbone_down_projs.append(down_proj)
            self.backbone_down_projs = nn.ModuleList(backbone_down_projs)
            mlp_in_dim = 768 * len(backbones) + state_dim
            self.mlp = mlp(input_dim=mlp_in_dim, hidden_dim=1024, output_dim=action_dim, hidden_depth=2)
        else:
            raise NotImplementedError

    def forward(self, qpos, image, env_state, actions=None):
        is_training = actions is not None
        bs, _ = qpos.shape
        all_cam_features = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            features = features[0]
            pos = pos[0]
            all_cam_features.append(self.backbone_down_projs[cam_id](features))
        flattened_features = []
        for cam_feature in all_cam_features:
            flattened_features.append(cam_feature.reshape([bs, -1]))
        flattened_features = torch.cat(flattened_features, axis=1)
        features = torch.cat([flattened_features, qpos], axis=1)
        a_hat = self.mlp(features)
        return a_hat


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


def build_encoder(args):
    d_model = args.hidden_dim
    dropout = args.dropout
    nhead = args.nheads
    dim_feedforward = args.dim_feedforward
    num_encoder_layers = args.enc_layers
    normalize_before = args.pre_norm
    activation = "relu"
    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
    return encoder


def build(args):
    state_dim = getattr(args, 'state_dim', 14)
    action_dim = getattr(args, 'action_dim', 14)
    use_fast_tokens = getattr(args, 'use_fast_tokens', False)
    vocab_size = getattr(args, 'fast_vocab_size', None)
    max_token_len = getattr(args, 'fast_max_token_len', None)
    pad_token_id = getattr(args, 'fast_pad_token_id', None)

    backbones = []
    backbone = build_backbone(args)
    backbones.append(backbone)
    transformer = build_transformer(args)
    encoder = build_encoder(args)

    model = DETRVAE(
        backbones, transformer, encoder,
        state_dim=state_dim, action_dim=action_dim,
        num_queries=args.num_queries, camera_names=args.camera_names,
        use_fast_tokens=use_fast_tokens,
        vocab_size=vocab_size, max_token_len=max_token_len,
        pad_token_id=pad_token_id,
    )
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters / 1e6,))
    return model


def build_cnnmlp(args):
    state_dim = getattr(args, 'state_dim', 14)
    action_dim = getattr(args, 'action_dim', 14)
    backbones = []
    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)
    model = CNNMLP(backbones, state_dim=state_dim, action_dim=action_dim, camera_names=args.camera_names)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters / 1e6,))
    return model