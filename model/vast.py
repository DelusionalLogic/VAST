import math
from functools import (
    partial,
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig as MainlineBertConfig
from transformers import BertForMaskedLM as MainlineBertMasked
from transformers import (
    BertTokenizer,
)

from model.vision_encoders.evaclip.eva_vit_model import (
    Block,
    PatchEmbed,
    trunc_normal_,
)

from .audio_encoders.beats.beats import (
    BEATs,
    BEATsConfig,
)
from .general_module import (
    Contra_head,
    Match_head,
)
# For some reason using the upstream bert model here tanks the
# inference quality. Maybe figure that out in the future
from .text_encoders.bert.bert import (
    BertConfig,
    BertForMaskedLM,
)


class EVAVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size, patch_size=16, in_chans=3, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, patch_dropout=0.,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, rope=False,
                 use_mean_pooling=True, init_scale=0.001, grad_checkpointing=False, xattn=False, postnorm=False,
                 pt_hw_seq_len=16, intp_freq=False, naiveswiglu=False, subln=False):
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
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values,
                xattn=xattn, postnorm=postnorm, subln=subln)
            for i in range(depth)])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
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
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x, rel_pos_bias=None)

        x = self.norm(x)
        return x

    def forward(self, x, return_all_features=True):
        return self.forward_features(x)

class CustomCLIP(nn.Module):
    def __init__(self, image_size):
        super().__init__()

        self.visual = EVAVisionTransformer(
            img_size=image_size,
            patch_size=14,
            use_mean_pooling=False,
            init_values=None,
            patch_dropout=.0,
            embed_dim=1408,
            depth=40,
            num_heads=16,
            mlp_ratio=4.3637,
            qkv_bias=True,
            drop_path_rate=0.4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            xattn=True,
            rope=False,
            postnorm=False, pt_hw_seq_len=16,   # 224/14
            intp_freq=False,
            naiveswiglu=False,
            subln=False
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
        bertconfig.add_cross_attention = True
        bertconfig.is_decoder = True
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
        bertconfig.add_cross_attention = True
        bertconfig.is_decoder = True
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

    def construct_multimodal_encoder(self):

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
        vision_output = self.vision_encoder.visual(vision_pixels.reshape(b*n,3,h,w), return_all_features=True)
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

    def compute_slice_scores(self, slice_multimodal_vision_input, slice_input_ids, slice_attention_mask):
        slice_output = self.multimodal_encoder.bert(
            input_ids = slice_input_ids,
            attention_mask = slice_attention_mask,
            encoder_hidden_states=slice_multimodal_vision_input,
        ).last_hidden_state
        slice_scores = F.softmax(self.itm_head(slice_output[:,0]),dim=1)[:,1]

        return slice_scores
