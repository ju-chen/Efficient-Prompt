import clip
import torch
import einops
import numpy as np
import torch.nn as nn
from collections import OrderedDict


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln_1 = LayerNorm(d_model)

        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TemporalModelling(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, dropout: float, attn_mask: torch.Tensor = None, ):
        super(TemporalModelling, self).__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, dropout, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks((x))


class CLIPrompt(torch.nn.Module):
    def __init__(self, args, actionlist, actiondict, actiontoken, device):
        super(CLIPrompt, self).__init__()

        self.device = device
        self.clipmodel, _ = clip.load(args.backbone, device=self.device, jit=False, return_intermediate_text_feature=args.return_intermediate_text_feature) 

        for paramclip in self.clipmodel.parameters():
            paramclip.requires_grad = False

        self.dropout = 0.0 # if args.tfm_layers > 2 else 0.0
        self.hidden_size = 512
        self.numF = args.numFrames
        self.temporal = args.temporal
        self.has_feature_input = 'feature' in args.dataset

        self.prefix = args.prefix
        self.postfix = args.postfix
        self.actionlist = actionlist
        self.actiondict = actiondict
        self.actiontoken = actiontoken
        self.tfm_layers = args.tfm_layers
        self.tfm_heads = args.tfm_heads

        self.embedding = torch.nn.Embedding(77, self.hidden_size)
        self.temporalEmbedding = torch.nn.Embedding(self.numF, self.hidden_size)

        if self.temporal == 1:
            self.temporalModelling = TemporalModelling(width=self.hidden_size, layers=self.tfm_layers, heads=self.tfm_heads, dropout=self.dropout)

        self.initialize_parameters()


    def initialize_parameters(self):
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.normal_(self.temporalEmbedding.weight, std=0.01)


    def replace_text_embedding(self, actionlist):
        self.text_embedding = self.embedding(torch.arange(77).to(self.device))[None, :].repeat([len(actionlist), 1, 1])
        self.prompt_actiontoken = torch.zeros(len(actionlist), 77)  

        for i, a in enumerate(actionlist):
            embedding = torch.from_numpy(self.actiondict[a][0]).float().to(self.device)
            token = torch.from_numpy(self.actiontoken[a][0])
            self.text_embedding[i][0] = embedding[0]
            ind = np.argmax(token, -1)

            self.text_embedding[i][self.prefix + 1: self.prefix + ind] = embedding[1:ind]
            self.text_embedding[i][self.prefix + ind + self.postfix] = embedding[ind]

            self.prompt_actiontoken[i][0] = token[0]
            self.prompt_actiontoken[i][self.prefix + 1: self.prefix + ind] = token[1:ind]
            self.prompt_actiontoken[i][self.prefix + ind + self.postfix] = token[ind]

        self.text_embedding.to(self.device)
        self.prompt_actiontoken.to(self.device)


    def forward(self, vids, inp_actionlist):
        # replace_text_embedding at every iter
        # otherwise RuntimeError: backward through the graph a second time
        self.replace_text_embedding(inp_actionlist)

        # encode text
        tFeature = self.clipmodel.encode_text(self.text_embedding, self.prompt_actiontoken)

        # encode videos
        if self.has_feature_input:
            vFeature = einops.rearrange(vids.float(), 'b t c -> t b c', t=self.numF)
        else:
            iFeature = self.clipmodel.encode_image(einops.rearrange(vids, 'b t c h w -> (b t) c h w'))
            vFeature = einops.rearrange(iFeature, '(b t) c -> t b c', t=self.numF)

        if self.temporal == 1:  # temporal modelling
            tempEmbedding = einops.repeat(self.temporalEmbedding(torch.arange(self.numF).to(self.device)), 't c -> t b c', b=vFeature.size(1))
            vFeature = vFeature + tempEmbedding.to(self.device)
            vFeature = self.temporalModelling(vFeature)  
            vFeature = vFeature.mean(dim=0)
        else:
            vFeature = vFeature.type(torch.float32).mean(dim=0)

        return vFeature, tFeature
