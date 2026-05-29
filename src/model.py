import torch
import torch.nn as nn

from sbrec import SBRec, LayerNorm


class AttSBModel(nn.Module):
    def __init__(self, sb, args):
        super().__init__()
        self.emb_dim = args.hidden_size
        self.item_num = args.item_num + 1
        self.souce_embeddings = nn.Embedding(args.source_item_num + 1, self.emb_dim)
        self.target_embeddings = nn.Embedding(args.target_item_num + 1, self.emb_dim)
        self.shared_layer = nn.Linear(self.emb_dim, self.emb_dim)
        self.embed_dropout = nn.Dropout(args.emb_dropout)
        self.position_embeddings = nn.Embedding(args.max_len, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.sb = sb
        self.side_seq_embedding = nn.Embedding(args.source_item_num + 1, self.emb_dim)
        nn.init.zeros_(self.side_seq_embedding.weight)
        self.side_seq_embedding.weight.requires_grad = False
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_mse = nn.MSELoss()
        self.lambda_x1 = getattr(args, 'lambda_x1', 0.1)

    def reverse(self, item_rep, item_rep1, noise_x_t, mask_seq):
        return self.sb.reverse_p_sample(item_rep, item_rep1, noise_x_t, mask_seq)

    def loss_sb_ce(self, rep_sb, labels, forward_flag, x1=None, pred_x1=None):
        if forward_flag:
            scores = torch.matmul(rep_sb, self.shared_layer(self.souce_embeddings.weight).t())
        else:
            scores = torch.matmul(rep_sb, self.target_embeddings.weight.t())
        loss = self.loss_ce(scores, labels.squeeze(-1))
        if x1 is not None and pred_x1 is not None:
            loss = loss + self.lambda_x1 * self.loss_mse(pred_x1, x1.detach())
        return loss

    def sb_rep_pre(self, rep_sb, forward_flag):
        if forward_flag:
            scores = torch.matmul(rep_sb, self.shared_layer(self.souce_embeddings.weight).t())
        else:
            scores = torch.matmul(rep_sb, self.target_embeddings.weight.t())
        return scores

    def forward(self, sequence, tag, forward_flag, train_flag=True):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        mask_seq = (sequence > 0).float()

        if forward_flag:
            item_embeddings = self.souce_embeddings(sequence)
        else:
            item_embeddings = self.target_embeddings(sequence)

        item_embeddings = item_embeddings + position_embeddings
        item_embeddings = self.embed_dropout(item_embeddings)
        item_embeddings = self.LayerNorm(item_embeddings)

        if train_flag:
            if forward_flag:
                tag_emb = self.souce_embeddings(tag.squeeze(-1))
                side_seq = self.side_seq_embedding(sequence).mean(dim=1)
            else:
                tag_emb = self.target_embeddings(tag.squeeze(-1))
                side_seq = None
            pred_x0, x1, pred_x1 = self.sb(item_embeddings, tag_emb, mask_seq, side_seq=side_seq)
            rep_sb = pred_x0
        else:
            x1 = None
            pred_x1 = None
            noise_x_t = torch.randn_like(item_embeddings[:, -1, :])
            rep_sb = self.reverse(item_embeddings, item_embeddings, noise_x_t, mask_seq)
        return rep_sb, x1, pred_x1


def create_model_sb(args):
    return SBRec(args)
