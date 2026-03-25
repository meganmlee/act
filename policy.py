import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import IPython
e = IPython.embed


class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model  # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        self.use_fast_tokens = args_override.get('use_fast_tokens', False)
        if self.use_fast_tokens:
            self.fast_pad_token_id = args_override['fast_pad_token_id']
        print(f'KL Weight {self.kl_weight}')
        print(f'Use FAST tokens: {self.use_fast_tokens}')

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)

        if actions is not None:  # training time
            if self.use_fast_tokens:
                # actions is (B, max_token_len) LongTensor of token IDs
                # is_pad is (B, max_token_len) bool — True where pad_token_id
                actions = actions[:, :self.model.num_queries]
                is_pad = is_pad[:, :self.model.num_queries]

                # Forward pass — model returns logits (B, max_token_len, vocab_size)
                logits, _, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)

                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
                loss_dict = dict()

                # Cross-entropy loss, ignoring pad positions
                B, S, V = logits.shape
                ce_loss = F.cross_entropy(
                    logits.reshape(B * S, V),
                    actions.reshape(B * S),
                )

                loss_dict['ce'] = ce_loss
                loss_dict['kl'] = total_kld[0]
                loss_dict['loss'] = loss_dict['ce'] + loss_dict['kl'] * self.kl_weight
                return loss_dict
            else:
                # Original continuous action mode
                actions = actions[:, :self.model.num_queries]
                is_pad = is_pad[:, :self.model.num_queries]

                a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
                loss_dict = dict()
                all_l1 = F.l1_loss(actions, a_hat, reduction='none')
                l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
                loss_dict['l1'] = l1
                loss_dict['kl'] = total_kld[0]
                loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
                return loss_dict
        else:  # inference time
            if self.use_fast_tokens:
                # Parallel prediction: model returns logits (B, max_token_len, vocab_size)
                logits, _, (_, _) = self.model(qpos, image, env_state)
                predicted_tokens = logits.argmax(dim=-1)  # (B, max_token_len)
                # Compute token lengths: first occurrence of pad token
                B = predicted_tokens.shape[0]
                token_lens = torch.full((B,), predicted_tokens.shape[1], dtype=torch.long)
                for b in range(B):
                    pad_positions = (predicted_tokens[b] == self.fast_pad_token_id).nonzero(as_tuple=False)
                    if len(pad_positions) > 0:
                        token_lens[b] = pad_positions[0].item()
                return predicted_tokens, token_lens
            else:
                a_hat, _, (_, _) = self.model(qpos, image, env_state)
                return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None:
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else:
            a_hat = self.model(qpos, image, env_state)
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld