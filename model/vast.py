import collections.abc
import math
from functools import (
    partial,
)
from itertools import (
    repeat,
)
from typing import (
    Optional,
    Tuple,
    cast,
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import (
    Tensor,
    nn,
)
from torch.nn.init import (
    trunc_normal_,
)
from transformers import (
    BertTokenizer,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.activations import (
    ACT2FN,
    GELUActivation,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
)

from .audio_encoders.beats.beats import (
    BEATs,
    BEATsConfig,
)
from .general_module import (
    Contra_head,
    Match_head,
)


class BertConfig(PretrainedConfig):
    model_type = "bert"

    def __init__(
        self,
        pad_token_id=0,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)


class BertSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.attention_head_size = int(768 / 12)
        self.all_head_size = 12 * self.attention_head_size

        self.query = nn.Linear(768, self.all_head_size)
        self.key = nn.Linear(768, self.all_head_size)
        self.value = nn.Linear(768, self.all_head_size)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (12, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
        # Further calls to cross_attention layer can then reuse all cross-attention
        # key/value_states (first "if" case)
        # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
        # all previous decoder key/value_states. Further calls to uni-directional self-attention
        # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
        # if encoder bi-directional self-attention `past_key_value` is always `None`
        past_key_value : Tuple[Tensor, Tensor] = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(768, 768)
        self.LayerNorm = nn.LayerNorm(768, eps=1e-12)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.self = BertSelfAttention()
        self.output = BertSelfOutput()
        self.pruned_heads = set()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
        )
        return self.output(self_outputs, hidden_states)


class BertIntermediate(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(768, 3072)
        self.intermediate_act_fn = GELUActivation()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(3072, 768)
        self.LayerNorm = nn.LayerNorm(768, eps=1e-12)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states



class BertLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_len_dim = 1
        self.attention = BertAttention()
        self.intermediate = BertIntermediate()
        self.output = BertOutput()

    def forward(
        self,
        input_: Tuple[torch.Tensor, Optional[torch.FloatTensor]]
    ) -> Tuple[torch.Tensor, Optional[torch.FloatTensor]]:
        (hidden_states, attention_mask) = input_

        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
        )

        layer_output = self.feed_forward_chunk(self_attention_outputs)

        return (layer_output, attention_mask)

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(*[BertLayer() for _ in range(12)])

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
    ):
        hidden_states = self.layer((hidden_states, attention_mask))[0]

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
        )

class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(30522, 768, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(512, 768)
        self.token_type_embeddings = nn.Embedding(2, 768)

        self.LayerNorm = nn.LayerNorm(768, eps=1e-12)
        self.register_buffer("position_ids", torch.arange(512).expand((1, -1)))
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        assert input_ids is not None
        input_shape = input_ids.size()

        seq_length = input_shape[1]

        position_ids = self.position_ids[:, 0 : seq_length + 0]
        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        return embeddings

class BertModel(PreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "bert"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder()

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def get_extended_attention_mask(
        self,
        attention_mask: Tensor,
        input_shape: Tuple[int],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):

        input_shape = input_ids.size()
        batch_size, seq_length = input_shape

        buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
        buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
        token_type_ids = buffered_token_type_ids_expanded

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.

        assert attention_mask is not None
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask,
            cast(Tuple[int], input_shape),
            dtype=self.dtype
        )

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
        )
        sequence_output = encoder_outputs[0]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class BertForMaskedLM(PreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "bert"
    supports_gradient_checkpointing = True

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)

        # Initialize weights and apply final processing
        self.post_init()

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)

class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qk_scale=None):
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
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
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
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
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
        trunc_normal_(self.head.weight, std=.02)
        self.head.weight.data.mul_(init_scale)
        self.head.bias.data.mul_(init_scale)

        trunc_normal_(self.pos_embed, std=.02)

        trunc_normal_(self.cls_token, std=.02)
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
        return F.normalize(features, dim=-1)

class VastTextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        bertconfig = BertConfig()
        self.multimodal_encoder = BertForMaskedLM(bertconfig)

        self.multimodal_encoder.tokenizer = BertTokenizer('./pretrained_weights/bert/bert-base-uncased/vocab.txt')

        # VAST
        self.contra_head_t = Contra_head(768, self.config.contra_dim)

    def _caption_tokens(self, raw_captions):
        caption_tokens = self.multimodal_encoder.tokenizer(
            raw_captions,
            padding="max_length",
            truncation=True,
            max_length=self.config.max_caption_len,
            return_tensors="pt"
        ).to(torch.device("cuda"))
        return caption_tokens


    def forward(self, raw_captions):
        caption_tokens = self._caption_tokens(raw_captions)
        input_ids = caption_tokens.input_ids
        attention_mask = caption_tokens.attention_mask

        caption_output = self.multimodal_encoder.bert(
            input_ids = input_ids,
            attention_mask = attention_mask,
        ).last_hidden_state

        caption_output_pooled = caption_output[:,0]
        feat_t = self.contra_head_t(caption_output_pooled) 
        feat_t = F.normalize(feat_t, dim=-1)
        return feat_t

class VAST(nn.Module):
    """ VLP pretraining """
    def __init__(self, config):
        super().__init__()

        self.config = config

        # CLIP
        self.vision_encoder = CustomCLIP(
            image_size=self.config.vision_resolution,
        )
        self.vision_dim = 1408

        # BEATS
        cfg = BEATsConfig()

        self.audio_encoder = BEATs(cfg)
        self.audio_dim = 768

        # BERT
        bertconfig = BertConfig()
        self.multimodal_encoder = BertForMaskedLM(bertconfig)
        self.multimodal_encoder.tokenizer = BertTokenizer('./pretrained_weights/bert/bert-base-uncased/vocab.txt')
        self.multimodal_dim = 768

        # VAST
        contra_dim = self.config.contra_dim
        self.contra_head_t = Contra_head(self.multimodal_dim, contra_dim) # This should be unused
        self.contra_head_s = Contra_head(self.multimodal_dim, contra_dim)
        self.contra_head_v = Contra_head(self.vision_dim, contra_dim)
        self.contra_head_a = Contra_head(self.audio_dim, contra_dim)
        self.contra_head_va = nn.Linear(self.vision_dim + self.audio_dim, contra_dim)
        self.contra_head_vs = nn.Linear(self.vision_dim + self.multimodal_dim, contra_dim)
        self.contra_head_vas = nn.Linear(self.vision_dim + self.audio_dim + self.multimodal_dim, contra_dim)
        self.contra_temp = nn.Parameter(torch.tensor(0.07))
        self.itm_head = Match_head(self.multimodal_dim)
        self.vision_frame_embedding = nn.Parameter(0.02 * torch.randn(1, self.config.max_vision_sample_num, self.multimodal_dim))
        self.audio_frame_embedding = nn.Parameter(0.02 * torch.randn(1, self.config.max_audio_sample_num, self.multimodal_dim))
        self.hidden_trans_vision_multimodal = nn.Sequential(nn.Linear(self.vision_dim, self.multimodal_dim),nn.LayerNorm(self.multimodal_dim, eps=1e-12))
        self.hidden_trans_audio_multimodal = nn.Sequential(nn.Linear(self.audio_dim, self.multimodal_dim),nn.LayerNorm(self.multimodal_dim, eps=1e-12))
        self.hidden_trans_subtitle_multimodal = nn.Sequential(nn.Linear(self.multimodal_dim, self.multimodal_dim),nn.LayerNorm(self.multimodal_dim, eps=1e-12))
        self.vision_type_embeddings = nn.Parameter(0.02 * torch.randn(1, 1, self.multimodal_dim)) 
        self.audio_type_embeddings = nn.Parameter(0.02 * torch.randn(1, 1, self.multimodal_dim)) 
        self.subtitle_type_embeddings = nn.Parameter(0.02 * torch.randn(1, 1, self.multimodal_dim)) 
        self.max_subtitle_len = config.max_subtitle_len

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

        vision_frame_embedding = F.interpolate(self.vision_frame_embedding.float().permute(0, 2, 1), n, mode='nearest').permute(0, 2, 1).to(self.vision_frame_embedding)
        vision_output = vision_output + vision_frame_embedding.unsqueeze(-2)
        vision_output = vision_output.reshape(b, -1, self.multimodal_dim) 
        vision_output = vision_output + self.vision_type_embeddings

        return vision_output

    def get_multimodal_forward_input_audio(self, audio_output):
        b,n,x,c = audio_output.shape

        if n!= self.audio_frame_embedding.shape[1]: #### testing and interpolate
            audio_frame_embedding = F.interpolate(self.audio_frame_embedding.permute(0,2,1),n,mode='nearest').permute(0,2,1)
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
        feat_vas = F.normalize(feat_vas,dim=-1)

        return feat_vas

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
        audio_spectrograms = audio_spectrograms
        b,n,h,w, = audio_spectrograms.shape
        audio_spectrograms = audio_spectrograms.reshape(-1,*audio_spectrograms.shape[2:])
        audio_output = self.audio_encoder(audio_spectrograms)
        audio_output = audio_output.reshape(b,n,-1,audio_output.shape[-1])
        return audio_output

    def _subtitle_output(self, raw_subtitles):
        subtitle_tokens = self.multimodal_encoder.tokenizer(
            raw_subtitles,
            padding="max_length",
            truncation=True,
            max_length=self.max_subtitle_len,
            return_tensors="pt"
        ).to(torch.device("cuda"))

        input_ids = subtitle_tokens.input_ids
        attention_mask = subtitle_tokens.attention_mask
        subtitle_output = self.multimodal_encoder.bert(
            input_ids = input_ids,
            attention_mask = attention_mask,
        ).last_hidden_state
        return subtitle_output
