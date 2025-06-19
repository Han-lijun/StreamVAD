from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math

def attention_pool(tensor, pool):
    tensor = tensor.permute(1, 0, 2)
    tensor = pool(tensor)
    tensor = tensor.permute(1, 0, 2)
    return tensor

def mask_memory(cached_mem, cached_video_names, cur_video_names):
    """
    Masking memory so that one video cannot attend to memory of a different video.
    """
    assert len(cached_mem) == len(cached_video_names)
    kept_ts = []

    for t, (cached_mem_t, cached_video_names_t) in enumerate(
            zip(cached_mem, cached_video_names)
    ):
        cached_mem_t = cached_mem_t.permute(1, 0, 2)
        keep = True
        try:
            assert len(cached_video_names_t) == cached_mem_t.shape[0]
        except:
            print(cached_mem_t.shape)

        if len(cached_video_names_t) >= len(cur_video_names):
            for i in range(len(cur_video_names)):
                if cached_video_names_t[i] != cur_video_names[i]:
                    cached_mem_t[i] = 0.0
                    keep = False
            if keep:
                kept_ts.append(t)
        else:
            for i in range(len(cached_video_names_t)):
                try:
                    if cached_video_names_t[i] != cur_video_names[i]:
                        cached_mem_t[i] = 0.0
                        keep = False
                except:
                    print('cached_video_names_t[i].device', cached_video_names_t[i].device)
                    print('cached_video_names_t[i]', cached_video_names_t[i])

                    print('cur_video_names[i].device', cur_video_names[i].device)
                    print('cur_video_names[i]', cur_video_names[i])
            if keep:
                kept_ts.append(t)

    if kept_ts == []:
        cached_mem = []
        cached_video_names = []

    elif len(cached_mem) > 0 and cached_mem[0].shape[0] == 1:

        return (
            [cached_mem[t] for t in kept_ts],

            [cached_video_names[t] for t in kept_ts],
        )
    return cached_mem, cached_video_names


def mask_time(cached_mem, cached_video_names, cur_video_names):
    """
    Masking memory so that one video cannot attend to memory of a different video.
    """

    kept_ts = []

    for t, (cached_mem_t, cached_video_names_t) in enumerate(
            zip(cached_mem, cached_video_names)
    ):

        keep = True

        if len(cached_video_names_t) >= len(cur_video_names):
            for i in range(len(cur_video_names)):
                if cached_video_names_t[i] != cur_video_names[i]:
                    keep = False
            if keep:
                kept_ts.append(t)
        else:
            for i in range(len(cached_video_names_t)):
                try:
                    if cached_video_names_t[i] != cur_video_names[i]:
                        keep = False
                except:
                    print('cached_video_names_t[i].device', cached_video_names_t[i].device)
                    print('cached_video_names_t[i]', cached_video_names_t[i])

                    print('cur_video_names[i].device', cur_video_names[i].device)
                    print('cur_video_names[i]', cur_video_names[i])
            if keep:
                kept_ts.append(t)

    if kept_ts == []:
        cached_mem = []
        cached_video_names = []

    elif len(cached_mem) > 0:

        return (
            [cached_mem[t] for t in kept_ts],

            [cached_video_names[t] for t in kept_ts],
        )
    return cached_mem, cached_video_names


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, p: int, attwindow, drop_rate, drop_attn_rate, drop_qkv_rate,
                 cache_keep_max_len):
        super().__init__()
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.p=p
        self.attnwindow=attwindow
        self.cached_k = []
        self.cached_time = []
        self.cached_q = []
        self.cached_video_names = []
        self.keep_max_len = cache_keep_max_len 
        self.proj = nn.Linear(512, 512)
        self.attenproj = nn.Linear(256, 512)
        self.compress_k = nn.Conv1d(
            attwindow,
            attwindow, 
            3,  
            stride=2, 
            padding=1,  
            groups=1,  
            bias=False,
        )
        self.compress_cur = nn.Conv1d(
            attwindow,
            attwindow, 
            3,  
            stride=2, 
            padding=1, 
            groups=1,  
            bias=False,
        )
        self.drop_rate = drop_rate  
        self.drop_attn_rate = drop_attn_rate  
        self.drop_qkv_rate = drop_qkv_rate  

        if drop_attn_rate > 0.0:
            self.attn_drop = nn.Dropout(drop_attn_rate)
        if drop_qkv_rate > 0.0:
            self.qkv_drop = nn.Dropout(drop_qkv_rate)

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        q=q.squeeze(1)
        k=k.squeeze(1)
        v=v.squeeze(1)
        d=1/math.sqrt(q.shape[1])

        attn = (q @ k.transpose(-2, -1)) * d
        attn = attn.softmax(dim=-1)
        N = q.shape[0]
        D= q.shape[1]
        tt=attn @ v + q
        x = tt.transpose(0, 1).reshape(-1, N, D)
        x=x.transpose(0, 1)
        return x
   
    def ebbinghaus_forgetting_curve(self, t, s):
        """
        Calculate the percentage of memory retention on the Ebbinghaus forgetting curve.

        Return to.
        Memory retention ratio R
        """
        R = math.exp(-t / s)
        return R

    def forward(self, x, video_names):
        x, needclean = x
        if needclean == 1:
            self.cached_k = []
            self.cached_video_names = []
            return (x, 0)
        x = self.ln_1(x)
        q = k  = x
        (
            self.cached_k,

            new_cached_video_names,
        ) = mask_memory(
            self.cached_k,
            self.cached_video_names,
            video_names,
        )

        (
            self.cached_time,
            new_cached_video_names,
        ) = mask_time(
            self.cached_time,
            self.cached_video_names,
            video_names,
        )

        self.cached_video_names = new_cached_video_names
        # hlj Updating the cache
        self.cached_k.append(k.detach())
        if not self.cached_time:
            self.cached_time.append(1)
        else:
            self.cached_time.append(self.cached_time[-1] + 1)

        self.cached_video_names.append(video_names)
        # hlj compress clip feature
        if len(self.cached_k) > 1:
            if self.cached_k[-2].shape[2]==512:
                self.cached_k[-2] = attention_pool(
                    self.cached_k[-2],
                    self.compress_k,
                )
                self.cached_k[-2] = self.cached_k[-2].detach()
            k = attention_pool(
                k,
                self.compress_cur,
            )
            k = self.cached_k[:-1] + [k]

        if len(self.cached_k) == self.keep_max_len:
            # hlj Control Maximum Length
            memory_retentions = []
            normalized_lst = (self.cached_time - np.min(self.cached_time)) / (np.max(self.cached_time) - np.min(self.cached_time))
            for idc, ic in enumerate(self.cached_k):
                average_value = ic.mean()
                s = torch.sigmoid(average_value)
                R = self.ebbinghaus_forgetting_curve(normalized_lst[idc], s)
                memory_retentions.append(R)

            # hlj get memory_retentions
            min_r_index = memory_retentions.index(min(memory_retentions))

            # hlj discard this clip from cache
            self.cached_k.pop(min_r_index)
            self.cached_video_names.pop(min_r_index)

            # FIFO version
            # self.cached_k= self.cached_k[1:]
            # self.cached_video_names = self.cached_video_names[1:]

        if isinstance(k, (tuple, list)):
            k = torch.cat(k, dim=0)
        if self.drop_qkv_rate > 0.0:
            q = self.qkv_drop(q)
            k = self.qkv_drop(k)
        if k.shape[2]==256:
            q = attention_pool(
                q,
                self.compress_k,
            )
        selfatten=self.attention(q, q, q)
        atten=self.attention(q, k, k)
        atten=atten+selfatten*self.p
        if atten.shape[2]==256:
            atten=self.attenproj(atten)
        x = x + atten
        x = x + self.mlp(self.ln_2(x))
        return (x, 1)

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, p: int, attwindow: int, drop_rate, drop_attn_rate, drop_qkv_rate,
                 cache_keep_max_len):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblock = ResidualAttentionBlock(width, p, attwindow, drop_rate, drop_attn_rate, drop_qkv_rate,
                                               cache_keep_max_len)

    def forward(self, x: torch.Tensor, video_names):
        return self.resblock(x, video_names)


class LSHMA(nn.Module):
    def __init__(self,
                 visual_width: int,
                 p: int,
                 visual_layers: int,
                 attn_window: int,             
                 device, drop_rate, drop_attn_rate, drop_qkv_rate, cache_keep_max_len):
        super().__init__()
        self.drop_rate = drop_rate  
        self.visual_width = visual_width
        self.attn_window = attn_window
        self.device = device
       
        self.blocks = nn.ModuleList()
        for i in range(visual_layers):
            attention_block = Transformer(
                width=visual_width,
                layers=visual_layers,
                p=p,
                attwindow=self.attn_window
                , drop_rate=drop_rate, drop_attn_rate=drop_attn_rate, drop_qkv_rate=drop_qkv_rate,
                cache_keep_max_len=cache_keep_max_len
            )
            self.blocks.append(attention_block)

     
        self.linear = nn.Linear(visual_width, visual_width)
        self.mlp2 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.classifier = nn.Linear(visual_width, 1)
        if drop_rate > 0.0:
            self.proj_drop = nn.Dropout(drop_rate)
        self.frame_position_embeddings = nn.Embedding(attn_window, visual_width)
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)
  
    def forward(self, visual, flag,  lengths, video_names,threshold=None):
        # hlj position embedding
        images = visual.to(torch.float)
        position_ids = torch.arange(lengths, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(images.shape[0], -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        frame_position_embeddings = frame_position_embeddings.permute(1, 0, 2)
        x = images.permute(1, 0, 2) + frame_position_embeddings
        if flag == 'Test':
            needclean = 0
            for blk_idx, blk in enumerate(self.blocks):
                # hlj cls is a flag for whether or not to clear the cache.
                x, cls = blk((x, needclean), video_names)
                # If the cache needs to be emptied, no computation is performed at subsequent network layers
                if cls == 0:
                    continue
                x1 = x.permute(1, 0, 2)
                x1 = self.linear(x1)
                if self.drop_rate > 0.0:
                    x1 = self.proj_drop(x1)

                logits = self.classifier(x1 + self.mlp2(x1))
                prob1 = logits.reshape(logits.shape[0] * logits.shape[1], logits.shape[2])
                prob1 = torch.sigmoid(prob1.squeeze(-1))
                if threshold is None:
                    print('threshold None')
                # If the result is greater than the threshold, clear the cache and output the result
                if prob1.max().item() > threshold:
                    needclean = 1
                else:
                    needclean = 0
            return logits
        elif flag == 'Train':
            # MTS version↓
            multiscale_list = []

            for blk_idx, blk in enumerate(self.blocks):
                multiscale_list.append(x.permute(1, 0, 2))
                x, _ = blk((x, None), video_names)
            multiscale_list.append(x.permute(1, 0, 2))
            multiscale_logits = []
            for item in multiscale_list:
                multiscale_logits.append(self.linear(item))
            multiscale_logits_tensor = torch.stack(multiscale_logits)
            max_values, _ = torch.max(multiscale_logits_tensor, dim=0)
            logits = self.classifier(max_values + self.mlp2(max_values))
            return logits

        return 0