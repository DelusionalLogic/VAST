import math
from functools import (
    partial,
)
from typing import (
    Optional,
    Tuple,
)

import numpy as np
import torch
import torchvision
from torch import (
    nn,
)
from transformers import (
    BertTokenizer,
)


class BEATsLayer(nn.Module):
    def __init__(self) -> None:

        super().__init__()
        self.k_proj = nn.Linear(768, 768)
        self.v_proj = nn.Linear(768, 768)
        self.q_proj = nn.Linear(768, 768)

        self.out_proj = nn.Linear(768, 768)

        self.grep_linear = nn.Linear(64, 8)
        self.grep_a = nn.Parameter(torch.ones(1, 12, 1, 1))

        self.attention_norm = nn.LayerNorm(768)

        self.fc1 = nn.Linear(768, 3072)
        self.fc2 = nn.Linear(3072, 768)

        self.output_norm = nn.LayerNorm(768)


    def forward(self, states_in: torch.Tensor, positional_bias):
        deep_norm_alpha = math.pow(24, 1 / 4)

        tgt_len, bsz, embed_dim = states_in.size()

        query_layer = self.q_proj(states_in)
        key_layer = self.k_proj(states_in)
        value_layer = self.v_proj(states_in)

        query_layer *= 0.125 * 1 / 32

        query_layer = query_layer.view(-1, bsz * 12, 64)
        query_layer = query_layer.transpose(0, 1)

        key_layer = key_layer.view(-1, bsz * 12, 64)
        key_layer = key_layer.transpose(0, 1)

        value_layer = value_layer.view(-1, bsz * 12, 64)
        value_layer = value_layer.transpose(0, 1)

        attention_scores : torch.Tensor = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = (attention_scores - attention_scores.max(dim=-1, keepdim=True)[0]) * 32
        query_layer = query_layer.view(bsz, 12, tgt_len, 64) * 32 / 0.125

        gate_a, gate_b = torch.sigmoid(self.grep_linear(query_layer).view(query_layer.size()[:-1] + (2, 4)).sum(dim=-1)).chunk(2, dim=-1)
        gate_a_1 = gate_a * (gate_b * self.grep_a - 1.0) + 2.0
        attention_mask = gate_a_1.view(bsz * 12, tgt_len, 1) * positional_bias

        attention_mask = attention_mask.view(attention_scores.size())

        attention_scores = attention_scores + attention_mask

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        self_attention_outputs = self.out_proj(context_layer)

        states_in = states_in * deep_norm_alpha + self_attention_outputs
        states_in = self.attention_norm(states_in)

        hidden_state = nn.functional.gelu(self.fc1(states_in))
        hidden_state = self.fc2(hidden_state)
        hidden_state += states_in * deep_norm_alpha
        hidden_state = self.output_norm(hidden_state)

        return hidden_state, positional_bias

class BEATs(nn.Module):
    def __init__(
            self,
    ) -> None:
        super().__init__()

        self.relative_attention_bias = nn.Embedding(320, 12)

        self.patch_embedding = nn.Conv2d(1, 512, kernel_size=16, stride=16, bias=False)
        self.layer_norm = nn.LayerNorm(512)
        self.post_extract_proj = nn.Linear(512, 768)

        self.pos_conv = nn.Conv1d(
            768,
            768,
            kernel_size=128,
            padding=64,
            groups=16,
        )
        self.pos_conv = nn.utils.parametrizations.weight_norm(self.pos_conv, name="weight", dim=2)

        self.layers = nn.ModuleList([
                BEATsLayer()
                for _ in range(12)
        ])

        self.post_norm = nn.LayerNorm(768)


    def forward(self, fbank):
        fbank = fbank.unsqueeze(1)
        features = self.patch_embedding(fbank)
        features = features.reshape(features.shape[0], features.shape[1], -1)
        features = features.transpose(1, 2)
        features = self.layer_norm(features)

        features = self.post_extract_proj(features)

        x = features
        x_conv = nn.functional.gelu(self.pos_conv(x.transpose(1, 2))[:,:,:-1])
        x_conv = x_conv.transpose(1, 2)

        x = x + x_conv
        x = self.post_norm(x)
        x = x.transpose(0, 1)

        tgt_len, bsz, _ = x.size()
        context_position = torch.arange(tgt_len, dtype=torch.long).unsqueeze(1)
        memory_position = context_position.transpose(0, 1)
        relative_position = memory_position - context_position

        relative_buckets = (relative_position > 0).to(torch.long) * 160
        relative_position = torch.abs(relative_position)

        is_small = relative_position < 80

        relative_postion_if_large = 80 + (torch.log(relative_position.float() / 80) / math.log(10) * (160 - 80)).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(relative_postion_if_large, 159)
        )
        relative_buckets += torch.where(is_small, relative_position, relative_postion_if_large)

        relative_position_bucket = relative_buckets.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(relative_position_bucket)
        positional_bias = values.permute([2, 0, 1])

        positional_bias = positional_bias.unsqueeze(0).repeat(bsz, 1, 1, 1).view(bsz * 12, tgt_len, tgt_len)

        for layer in self.layers:
            x, positional_bias = layer(x, positional_bias)

        return x.transpose(0, 1)

class BertLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_len_dim = 1

        self.query = nn.Linear(768, 768)
        self.key = nn.Linear(768, 768)
        self.value = nn.Linear(768, 768)

        self.attention_dense = nn.Linear(768, 768)
        self.attention_norm = nn.LayerNorm(768, eps=1e-12)

        self.intermediate = nn.Linear(768, 3072)

        self.output = nn.Linear(3072, 768)
        self.output_norm = nn.LayerNorm(768, eps=1e-12)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (12, 64)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        input_: Tuple[torch.Tensor, torch.FloatTensor]
    ) -> Tuple[torch.Tensor, Optional[torch.FloatTensor]]:
        (states_in, attention_mask) = input_

        key_layer = self.transpose_for_scores(self.key(states_in))
        value_layer = self.transpose_for_scores(self.value(states_in))
        query_layer = self.transpose_for_scores(self.query(states_in))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / 8

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (768,)
        self_attention_outputs = context_layer.view(new_context_layer_shape)

        self_attention_outputs = self.attention_dense(self_attention_outputs)
        self_attention_outputs = self.attention_norm(self_attention_outputs + states_in)

        hidden_states = nn.functional.gelu(self.intermediate(self_attention_outputs))

        hidden_states = self.output(hidden_states)
        hidden_states = self.output_norm(hidden_states + self_attention_outputs)

        return (hidden_states, attention_mask)

class BertModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.word_embeddings = nn.Embedding(30522, 768, padding_idx=0)
        self.position_embeddings = nn.Embedding(512, 768)
        self.token_type_embeddings = nn.Embedding(2, 768)

        self.embeddings_norm = nn.LayerNorm(768, eps=1e-12)
        self.register_buffer("position_ids", torch.arange(512).unsqueeze((0)))

        self.layer = nn.Sequential(*[BertLayer() for _ in range(12)])

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        embeddings = self.word_embeddings(input_ids)

        embeddings += self.position_embeddings(self.position_ids[:, :input_ids.size(1)])
        embeddings = self.embeddings_norm(embeddings)

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return self.layer((embeddings, extended_attention_mask))[0]

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)

        self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
        self.v_bias = nn.Parameter(torch.zeros(all_head_dim))

        self.inner_attn_ln = nn.Identity()
        # self.proj = nn.Linear(all_head_dim, all_head_dim)
        self.proj = nn.Linear(all_head_dim, dim)

    def forward(self, x, rel_pos_bias=None, attn_mask=None):
        B, N, C = x.shape

        qkv_bias = None
        qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # print('xshape',x.shape)
        # print('weight_shape',self.qkv.weight.shape)
        qkv = nn.functional.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)   # 3, B, num_heads, N, C
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias.type_as(attn)

        if attn_mask is not None:
            attn_mask = attn_mask.bool()
            attn = attn.masked_fill(~attn_mask[:, None, None, :], float("-inf"))

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.inner_attn_ln(x)
        x = self.proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qk_scale=None, init_values=None, norm_layer=nn.LayerNorm, postnorm=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qk_scale=qk_scale)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = torchvision.ops.misc.MLP(
            in_channels=dim,
            hidden_channels=[mlp_hidden_dim, dim],
            activation_layer=nn.GELU,
        )

        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

        self.postnorm = postnorm

    def forward(self, x, rel_pos_bias=None, attn_mask=None):
        if self.gamma_1 is None:
            if self.postnorm:
                x = x + self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))
                x = x + self.norm2(self.mlp(x))
            else:
                x = x + self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)
                x = x + self.mlp(self.norm2(x))
        else:
            if self.postnorm:
                x = x + self.gamma_1 * self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))
                x = x + self.gamma_2 * self.norm2(self.mlp(x))
            else:
                x = x + self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)
                x = x + self.gamma_2 * self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class EVAVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size, patch_size=16, in_chans=3, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qk_scale=None,
                 norm_layer=nn.LayerNorm, init_values=None,
                 use_abs_pos_emb=True,
                 init_scale=0.001, postnorm=False):
        super().__init__()
        self.image_size = img_size
        self.num_classes = 1024
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        assert use_abs_pos_emb
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qk_scale=qk_scale, norm_layer=norm_layer, init_values=init_values, postnorm=postnorm)
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.fc_norm = None
        self.head = nn.Linear(embed_dim, self.num_classes)
        # trunc_normal_(self.head.weight, std=.02)
        # self.head.weight.data.mul_(init_scale)
        # self.head.bias.data.mul_(init_scale)

        # trunc_normal_(self.pos_embed, std=.02)

        # trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.mask_token, std=.02)

    def forward_features(self, x):
        x = self.patch_embed(x)
        batch_size, _, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x, rel_pos_bias=None)

        x = self.norm(x)
        return x

    def forward(self, x):
        return self.forward_features(x)

class CustomCLIP(nn.Module):
    def __init__(self, image_size):
        super().__init__()

        self.visual = EVAVisionTransformer(
            img_size=image_size,
            patch_size=14,
            init_values=None,
            embed_dim=1408,
            depth=40,
            num_heads=16,
            mlp_ratio=4.3637,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            postnorm=False
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_image(self, image):
        features = self.visual(image)
        return nn.functional.normalize(features, dim=-1)

class VastTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.multimodal_encoder = BertModel()
        self.tokenizer = BertTokenizer('./pretrained_weights/bert/bert-base-uncased/vocab.txt')

        # VAST
        self.contra_head_t = nn.Linear(768, 512, bias=False)

    def _caption_tokens(self, raw_captions):
        caption_tokens = self.tokenizer(
            raw_captions,
            padding="max_length",
            truncation=True,
            max_length=70,
            return_tensors="pt"
        ).to(torch.device("cuda"))
        return caption_tokens


    def forward(self, raw_captions):
        caption_tokens = self._caption_tokens(raw_captions)
        input_ids = caption_tokens.input_ids
        attention_mask = caption_tokens.attention_mask

        caption_output = self.multimodal_encoder(
            input_ids = input_ids,
            attention_mask = attention_mask,
        )

        caption_output_pooled = caption_output[:,0]
        feat_t = self.contra_head_t(caption_output_pooled) 
        feat_t = nn.functional.normalize(feat_t, dim=-1)
        return feat_t

class VAST(nn.Module):
    """ VLP pretraining """
    def __init__(self):
        super().__init__()

        # CLIP
        self.vision_encoder = CustomCLIP(
            image_size=224,
        )
        self.vision_dim = 1408

        # BEATS
        self.audio_encoder = BEATs()
        self.audio_dim = 768

        # BERT
        self.multimodal_encoder = BertModel()
        self.tokenizer = BertTokenizer('./pretrained_weights/bert/bert-base-uncased/vocab.txt')
        self.multimodal_dim = 768

        # VAST
        contra_dim = 512
        self.contra_head_t = nn.Linear(self.multimodal_dim, contra_dim, bias=False) # This should be unused
        self.contra_head_s = nn.Linear(self.multimodal_dim, contra_dim, bias=False)
        self.contra_head_v = nn.Linear(self.vision_dim, contra_dim, bias=False)
        self.contra_head_a = nn.Linear(self.audio_dim, contra_dim, bias=False)
        self.contra_head_va = nn.Linear(self.vision_dim + self.audio_dim, contra_dim)
        self.contra_head_vs = nn.Linear(self.vision_dim + self.multimodal_dim, contra_dim)
        self.contra_head_vas = nn.Linear(self.vision_dim + self.audio_dim + self.multimodal_dim, contra_dim)
        self.contra_temp = nn.Parameter(torch.tensor(0.07))
        self.itm_head = nn.Sequential(
            nn.Linear(self.multimodal_dim, self.multimodal_dim),
            nn.GELU(),
            nn.LayerNorm(self.multimodal_dim, eps=1e-12),
            nn.Linear(self.multimodal_dim, 2),
        )
        self.vision_frame_embedding = nn.Parameter(0.02 * torch.randn(1, 8, self.multimodal_dim))
        self.audio_frame_embedding = nn.Parameter(0.02 * torch.randn(1, 1, self.multimodal_dim))
        self.hidden_trans_vision_multimodal = nn.Sequential(nn.Linear(self.vision_dim, self.multimodal_dim),nn.LayerNorm(self.multimodal_dim, eps=1e-12))
        self.hidden_trans_audio_multimodal = nn.Sequential(nn.Linear(self.audio_dim, self.multimodal_dim),nn.LayerNorm(self.multimodal_dim, eps=1e-12))
        self.hidden_trans_subtitle_multimodal = nn.Sequential(nn.Linear(self.multimodal_dim, self.multimodal_dim),nn.LayerNorm(self.multimodal_dim, eps=1e-12))
        self.vision_type_embeddings = nn.Parameter(0.02 * torch.randn(1, 1, self.multimodal_dim)) 
        self.audio_type_embeddings = nn.Parameter(0.02 * torch.randn(1, 1, self.multimodal_dim)) 
        self.subtitle_type_embeddings = nn.Parameter(0.02 * torch.randn(1, 1, self.multimodal_dim)) 
        self.max_subtitle_len = 70

    def pool_vision_for_contra(self, feature):
        # always use frame_avg for retrieval
        feature = feature[:,:,0]
        feature = torch.mean(feature, dim=1)
        return feature

    def pool_text_for_contra(self, feature):
        return feature[:,0]

    def pool_audio_for_contra(self, feature):
        feature = feature.mean(dim=2)
        feature = torch.mean(feature, dim=1)
        return feature  

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_multimodal_forward_input_vision(self, vision_output):
        b,n,x,c = vision_output.shape
        vision_output = self.hidden_trans_vision_multimodal(vision_output)  

        vision_frame_embedding = nn.functional.interpolate(self.vision_frame_embedding.float().permute(0, 2, 1), n, mode='nearest').permute(0, 2, 1).to(self.vision_frame_embedding)
        vision_output = vision_output + vision_frame_embedding.unsqueeze(-2)
        vision_output = vision_output.reshape(b, -1, self.multimodal_dim) 
        vision_output = vision_output + self.vision_type_embeddings

        return vision_output

    def get_multimodal_forward_input_audio(self, audio_output):
        b,n,x,c = audio_output.shape

        if n!= self.audio_frame_embedding.shape[1]: #### testing and interpolate
            audio_frame_embedding = nn.functional.interpolate(self.audio_frame_embedding.permute(0,2,1),n,mode='nearest').permute(0,2,1)
        else:
            audio_frame_embedding = self.audio_frame_embedding
        audio_output = self.hidden_trans_audio_multimodal(audio_output)
        audio_output =  audio_output + audio_frame_embedding.unsqueeze(-2)
        audio_output = audio_output.reshape(b,-1,self.multimodal_dim)
        audio_output = audio_output + self.audio_type_embeddings
        return audio_output


    def get_multimodal_forward_input_subtitle(self, subtitle_output):
        subtitle_output = self.hidden_trans_subtitle_multimodal(subtitle_output)
        subtitle_output = subtitle_output + self.subtitle_type_embeddings    
        return subtitle_output

    def _feat_vas(self, vision, audio, subtitle):
        vision_output_pooled = self.pool_vision_for_contra(vision)
        audio_output_pooled = self.pool_audio_for_contra(audio)
        subtitle_output_pooled = self.pool_text_for_contra(subtitle)

        feat_vas = torch.cat((vision_output_pooled, audio_output_pooled, subtitle_output_pooled), dim=1)
        feat_vas = self.contra_head_vas(feat_vas)
        feat_vas = nn.functional.normalize(feat_vas,dim=-1)

        return feat_vas

    def _feat_va(self, vision, audio):
        vision_output_pooled = self.pool_vision_for_contra(vision)
        audio_output_pooled = self.pool_audio_for_contra(audio)

        feat = torch.cat((vision_output_pooled, audio_output_pooled), dim=1)
        feat = self.contra_head_va(feat)
        feat = nn.functional.normalize(feat,dim=-1)

        return feat

    def _condition_feats_vas(self, vision, audio, subtitle):
        condition_feats_v = self.get_multimodal_forward_input_vision(vision)
        condition_feats_a = self.get_multimodal_forward_input_audio(audio)
        condition_feats_s = self.get_multimodal_forward_input_subtitle(subtitle)

        condition_feats_vas = torch.cat((condition_feats_v, condition_feats_a, condition_feats_s),dim=1)

        return condition_feats_vas

    def _vision_output(self, vision_pixels):
        vision_pixels = vision_pixels
        b,n,_,h,w = vision_pixels.shape
        vision_output = self.vision_encoder.visual(vision_pixels.reshape(b*n,3,h,w))
        vision_output = vision_output.reshape(b,-1,*vision_output.shape[-2:])

        return vision_output

    def _audio_output(self, audio_spectrograms):
        pre_size = audio_spectrograms.size()[:-2]
        audio_spectrograms = audio_spectrograms.reshape((-1, ) + audio_spectrograms.shape[2:])
        audio_output = self.audio_encoder(audio_spectrograms)
        audio_output = audio_output.reshape(pre_size + (-1, audio_output.shape[-1]))
        return audio_output

    def _subtitle_output(self, raw_subtitles):
        subtitle_tokens = self.tokenizer(
            raw_subtitles,
            padding="max_length",
            truncation=True,
            max_length=self.max_subtitle_len,
            return_tensors="pt"
        ).to(torch.device("cuda"))

        input_ids = subtitle_tokens.input_ids
        attention_mask = subtitle_tokens.attention_mask
        subtitle_output = self.multimodal_encoder(
            input_ids = input_ids,
            attention_mask = attention_mask,
        )
        return subtitle_output
