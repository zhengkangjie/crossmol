# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss


@register_loss("smi2struct")
class Smi2StructLoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
        self.padding_idx = task.atom_dictionary.pad()
        self.smi_padding_idx = task.smi_dictionary.pad()
        self.seed = task.seed
        self.dist_mean = 6.312581655060595
        self.dist_std = 3.3899264663911888

    def forward(self, model, sample, reduce=True):
        input_key = "net_input"
        target_key = "target"
        masked_tokens = sample[target_key]["tokens_target"].ne(self.padding_idx)
        smi_masked_tokens = sample[target_key]["smi_tokens_target"].ne(self.smi_padding_idx)
        sample_size = masked_tokens.long().sum()
        smi_sample_size = smi_masked_tokens.long().sum()
        # print('smi_tokens size:', sample[input_key]['smi_tokens'].size())
        (
            logits_decoder,
            decoder_distance,
            decoder_coord,
            x_norm,
            delta_decoder_pair_rep_norm,
            encoder_logits,
        ) = model(**sample[input_key], encoder_masked_tokens=smi_masked_tokens, decoder_masked_tokens=masked_tokens)
        target = sample[target_key]["tokens_target"]
        if masked_tokens is not None:
            target = target[masked_tokens]
        masked_token_loss = F.nll_loss(
            F.log_softmax(logits_decoder, dim=-1, dtype=torch.float32),
            target,
            ignore_index=self.padding_idx,
            reduction="mean",
        )
        masked_pred = logits_decoder.argmax(dim=-1)
        masked_hit = (masked_pred == target).long().sum()
        masked_cnt = sample_size
        loss = masked_token_loss * self.args.masked_token_loss
        logging_output = {
            "sample_size": 1,
            "bsz": sample[target_key]["tokens_target"].size(0),
            "seq_len": sample[target_key]["tokens_target"].size(1)
            * sample[target_key]["tokens_target"].size(0),
            "masked_token_loss": masked_token_loss.data,
            "masked_token_hit": masked_hit.data,
            "masked_token_cnt": masked_cnt,
        }

        # print('smi_tokens_target size:', smi_tokens_target.size())
        # print('encoder_logits size:', encoder_logits.size())
        if encoder_logits is not None and self.args.masked_smi_loss > 0:
            smi_tokens_target = sample[target_key]["smi_tokens_target"]
            if smi_masked_tokens is not None:
                smi_tokens_target = smi_tokens_target[smi_masked_tokens]
            smi_masked_token_loss = F.nll_loss(
                F.log_softmax(encoder_logits, dim=-1, dtype=torch.float32),
                smi_tokens_target,
                ignore_index=self.smi_padding_idx,
                reduction="mean",
            )
            loss = smi_masked_token_loss * self.args.masked_smi_loss

            logging_output["smi_masked_token_loss"] = smi_masked_token_loss.data

            smi_masked_pred = encoder_logits.argmax(dim=-1)
            smi_masked_hit = (smi_masked_pred == smi_tokens_target).long().sum()
            smi_masked_cnt = smi_sample_size
            logging_output["smi_masked_token_hit"] = smi_masked_hit.data
            logging_output["smi_masked_token_cnt"] = smi_masked_cnt.data

        if decoder_coord is not None:
            # real = mask + delta
            coord_target = sample[target_key]["coord_target"]
            masked_coord_loss = F.smooth_l1_loss(
                decoder_coord[masked_tokens].view(-1, 3).float(),
                coord_target[masked_tokens].view(-1, 3),
                reduction="mean",
                beta=1.0,
            )
            loss = loss + masked_coord_loss * self.args.masked_coord_loss
            # restore the scale of loss for displaying
            logging_output["masked_coord_loss"] = masked_coord_loss.data

        if decoder_distance is not None:
            dist_masked_tokens = masked_tokens
            masked_dist_loss = self.cal_dist_loss(
                sample, decoder_distance, dist_masked_tokens, target_key, normalize=True
            )
            loss = loss + masked_dist_loss * self.args.masked_dist_loss
            logging_output["masked_dist_loss"] = masked_dist_loss.data

        if self.args.decoder_x_norm_loss > 0 and x_norm is not None:
            loss = loss + self.args.decoder_x_norm_loss * x_norm
            logging_output["x_norm_loss"] = x_norm.data

        if (
            self.args.decoder_delta_pair_repr_norm_loss > 0
            and delta_decoder_pair_rep_norm is not None
        ):
            loss = (
                loss + self.args.decoder_delta_pair_repr_norm_loss * delta_decoder_pair_rep_norm
            )
            logging_output[
                "delta_pair_repr_norm_loss"
            ] = delta_decoder_pair_rep_norm.data

        logging_output["loss"] = loss.data
        return loss, 1, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        bsz = sum(log.get("bsz", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        seq_len = sum(log.get("seq_len", 0) for log in logging_outputs)
        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3)
        metrics.log_scalar("seq_len", seq_len / bsz, 1, round=3)

        masked_loss = sum(log.get("masked_token_loss", 0) for log in logging_outputs)
        metrics.log_scalar(
            "masked_token_loss", masked_loss / sample_size, sample_size, round=3
        )

        smi_masked_token_loss = sum(log.get("smi_masked_token_loss", 0) for log in logging_outputs)
        if smi_masked_token_loss > 0:
            metrics.log_scalar(
                "smi_masked_token_loss", smi_masked_token_loss / sample_size, sample_size, round=3
            )
        
        masked_acc = sum(
            log.get("masked_token_hit", 0) for log in logging_outputs
        ) / sum(log.get("masked_token_cnt", 0) for log in logging_outputs)
        metrics.log_scalar("masked_acc", masked_acc, sample_size, round=3)

        smi_masked_token_cnt = sum(log.get("smi_masked_token_cnt", 0) for log in logging_outputs)
        if smi_masked_token_cnt > 0:
            smi_masked_acc = sum(
                log.get("smi_masked_token_hit", 0) for log in logging_outputs
            ) / smi_masked_token_cnt
            metrics.log_scalar("smi_masked_acc", smi_masked_acc, sample_size, round=3)

        masked_coord_loss = sum(
            log.get("masked_coord_loss", 0) for log in logging_outputs
        )
        if masked_coord_loss > 0:
            metrics.log_scalar(
                "masked_coord_loss",
                masked_coord_loss / sample_size,
                sample_size,
                round=3,
            )

        masked_dist_loss = sum(
            log.get("masked_dist_loss", 0) for log in logging_outputs
        )
        if masked_dist_loss > 0:
            metrics.log_scalar(
                "masked_dist_loss", masked_dist_loss / sample_size, sample_size, round=3
            )

        x_norm_loss = sum(log.get("x_norm_loss", 0) for log in logging_outputs)
        if x_norm_loss > 0:
            metrics.log_scalar(
                "x_norm_loss", x_norm_loss / sample_size, sample_size, round=3
            )

        delta_pair_repr_norm_loss = sum(
            log.get("delta_pair_repr_norm_loss", 0) for log in logging_outputs
        )
        if delta_pair_repr_norm_loss > 0:
            metrics.log_scalar(
                "delta_pair_repr_norm_loss",
                delta_pair_repr_norm_loss / sample_size,
                sample_size,
                round=3,
            )

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

    def cal_dist_loss(self, sample, dist, masked_tokens, target_key, normalize=False):
        dist_masked_tokens = masked_tokens
        masked_distance = dist[dist_masked_tokens, :]
        masked_distance_target = sample[target_key]["distance_target"][
            dist_masked_tokens
        ]
        non_pad_pos = masked_distance_target > 0
        if normalize:
            masked_distance_target = (
                masked_distance_target.float() - self.dist_mean
            ) / self.dist_std
        masked_dist_loss = F.smooth_l1_loss(
            masked_distance[non_pad_pos].view(-1).float(),
            masked_distance_target[non_pad_pos].view(-1),
            reduction="mean",
            beta=1.0,
        )
        return masked_dist_loss