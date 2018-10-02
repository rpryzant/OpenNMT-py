import torch.nn as nn
import torch
import onmt
from onmt.encoders.encoder import EncoderBase
from onmt.encoders.transformer import TransformerEncoder, TransformerEncoderLayer
# from onmt.utils.misc import aeq
from onmt.modules.position_ffn import PositionwiseFeedForward


class SimpleContextTransformerEncoder(EncoderBase):


    def __init__(self, num_layers, d_model, heads, d_ff, dropout, embeddings, 
            selected_ctx=0, fields=None):
        super(SimpleContextTransformerEncoder, self).__init__()
        self.selected_ctx = selected_ctx
        self.fields = fields


        self.num_layers = num_layers
        self.embeddings = embeddings
        self.layer_norm_shared = onmt.modules.LayerNorm(d_model)
        self.layer_norm_ctx = onmt.modules.LayerNorm(d_model)
        self.layer_norm_src_final = onmt.modules.LayerNorm(d_model)
        self.layer_norm_ctx_final = onmt.modules.LayerNorm(d_model)

        self.shared_layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
            for _ in range(num_layers - 1)])

        self.extra_ctx_layer = TransformerEncoderLayer(
            d_model, heads, d_ff, dropout)
        self.ctx_src_self_attn = onmt.modules.MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.ctx_src_layer_norm = onmt.modules.LayerNorm(d_model)

        self.src_self_attn = onmt.modules.MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.src_layer_norm = onmt.modules.LayerNorm(d_model)

        # TODO dim
        self.gate = nn.Linear(d_model * 2, 1)
        self.gate_sigmoid = nn.Sigmoid()

        self.final_feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.final_layer_norm = onmt.modules.LayerNorm(d_model)

    def partial_encode(self, input, input_lengths):
        emb = self.embeddings(input)
        out = emb.transpose(0, 1).contiguous()
        words = input[:, :, 0].transpose(0, 1)
        w_batch, w_len = words.size()
        padding_idx = self.embeddings.word_padding_idx

        mask = words.data.eq(padding_idx).unsqueeze(1).expand(w_batch, w_len, w_len)
        for i in range(len(self.shared_layers)):
            out = self.shared_layers[i](out, mask)
        return out, mask, emb

    def forward(self, src, src_lengths, ctx_0, ctx_0_lengths, ctx_1, ctx_1_lengths):
        # TODO refactor/clean up
        print(ctx_0); quit()
        # run src through n-1 layers of shared stack
        out_src, src_mask, src_emb = self.partial_encode(src, src_lengths)
        out_src = self.layer_norm_shared(out_src)

        # run ctx through n-1 layers of shared stack
        if self.selected_ctx == 0:
            out_ctx, ctx_mask, _ = self.partial_encode(ctx_0, ctx_0_lengths)
        elif self.selected_ctx == 1:
            out_ctx, ctx_mask, _ = self.partial_encode(ctx_1, ctx_1_lengths)
        out_ctx = self.layer_norm_shared(out_ctx)

        # finish off source: final self attn, norm + add
        final_src, _ = self.src_self_attn(out_src, out_src, out_src, mask=src_mask)
        final_src = self.layer_norm_src_final(final_src) + out_src

        # finish off ctx: extra layer, use src to attend over ctx
        out_ctx = self.extra_ctx_layer(out_ctx, ctx_mask)

        words = ctx_0 if self.selected_ctx == 0 else ctx_1
        words = words[:, :, 0].transpose(0, 1)
        w_batch, w_len = words.size()
        [_, src_len, _] = src_mask.size()
        padding_idx = self.embeddings.word_padding_idx
        mask = words.data.eq(padding_idx).unsqueeze(1).expand(w_batch, src_len, w_len)
        final_ctx, _ = self.ctx_src_self_attn(out_ctx, out_ctx, final_src, mask=mask) # TODO -- check masking
        final_ctx = self.layer_norm_ctx_final(final_ctx) + out_src

        # gate the ctx + src stuff
        g = self.gate_sigmoid(self.gate(torch.cat((final_ctx, final_src), 2)))
        gated_encoding = g * final_ctx + (1 - g) * final_src

        # final feedfowrward, layer norm + add
        output = self.final_feed_forward(gated_encoding)
        output = self.final_layer_norm(output) + gated_encoding


        # attn_input = torch.cat((out_ctx, out_src), 1)
        # attn_mask_words = torch.cat(
        #     (src[:, :, 0].transpose(0, 1), ctx_1[:, :, 0].transpose(0, 1)),
        #     1)
        # w_batch, w_len = attn_mask_words.size()
        # attn_mask = attn_mask_words.data.eq(self.embeddings.word_padding_idx).unsqueeze(1).expand(
        #     w_batch, w_len, w_len)
        # final_ctx, _ = self.ctx_src_self_attn(attn_input, attn_input, attn_input, mask=attn_mask)
        # final_ctx = self.ctx_src_layer_norm(final_ctx) + attn_input
        
        # final_ctx = final_ctx[:, ctx_len:, :]     # only take src-enriched vecs

        # gated sum

        return src_emb, output.transpose(0, 1).contiguous()
        
        
        
        
        
        
