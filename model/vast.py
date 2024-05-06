import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from torch.nn import LayerNorm as LayerNorm

from .general_module import (
    Contra_head,
    Match_head,
)


class VAST(nn.Module):
    """ VLP pretraining """
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.load_clip_model()
        self.load_beats_model()
        self.construct_multimodal_encoder()

        contra_dim = self.config.contra_dim
        self.contra_head_t = Contra_head(self.multimodal_dim, contra_dim)
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
        self.hidden_trans_vision_multimodal = nn.Sequential(nn.Linear(self.vision_dim, self.multimodal_dim),LayerNorm(self.multimodal_dim, eps=1e-12))
        self.hidden_trans_audio_multimodal = nn.Sequential(nn.Linear(self.audio_dim, self.multimodal_dim),LayerNorm(self.multimodal_dim, eps=1e-12))
        self.hidden_trans_subtitle_multimodal = nn.Sequential(nn.Linear(self.multimodal_dim, self.multimodal_dim),LayerNorm(self.multimodal_dim, eps=1e-12))
        self.vision_type_embeddings = nn.Parameter(0.02 * torch.randn(1, 1, self.multimodal_dim)) 
        self.audio_type_embeddings = nn.Parameter(0.02 * torch.randn(1, 1, self.multimodal_dim)) 
        self.subtitle_type_embeddings = nn.Parameter(0.02 * torch.randn(1, 1, self.multimodal_dim)) 
        self.beam_size  = config.beam_size
        self.itm_ratio = config.itm_ratio   
        self.max_omni_caption_len = config.max_omni_caption_len
        self.max_caption_len = config.max_caption_len
        self.max_subtitle_len = config.max_subtitle_len

    def load_beats_model(self):
        from .audio_encoders.beats.beats import (
            BEATs,
            BEATsConfig,
        )
        cfg = BEATsConfig()

        self.audio_encoder = BEATs(cfg, checkpointing = self.config.checkpointing)
        # self.audio_encoder.load_state_dict(checkpoint['model'])
        self.audio_dim = 768

    def load_clip_model(self):
        from .vision_encoders.evaclip import (
            create_model,
        )
        model_name = "EVA01-CLIP-g-14"
        pretrained = "./pretrained_weights/clip/EVA01_CLIP_g_14_psz14_s11B.pt"
        self.vision_dim = 1408
        self.vision_encoder = create_model(model_name, pretrained, image_size = self.config.vision_resolution)

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
        # For some reason using the upstream bert model here tanks the
        # inference quality. Maybe figure that out in the future
        from .text_encoders.bert.bert import (
            BertConfig,
            BertForMaskedLM,
        )

        bertconfig = BertConfig()
        bertconfig.add_cross_attention = True
        bertconfig.is_decoder = True
        self.multimodal_encoder = BertForMaskedLM(bertconfig)
        self.multimodal_dim = 768

        if self.config.checkpointing:
            self.multimodal_encoder._set_gradient_checkpointing(self.multimodal_encoder.bert.encoder, True)

        from transformers import (
            BertTokenizer,
        )

        self.multimodal_encoder.tokenizer = BertTokenizer.from_pretrained('./pretrained_weights/bert/bert-base-uncased')

    def _caption_tokens(self, batch):
        caption_tokens = self.multimodal_encoder.tokenizer(
            batch.raw_captions,
            padding="max_length",
            truncation=True,
            max_length=self.max_caption_len,
            return_tensors="pt"
        ).to(torch.device('cuda'))
        return caption_tokens


    def _feat_t(self, caption_tokens):
        input_ids = caption_tokens.input_ids
        attention_mask = caption_tokens.attention_mask
        caption_output = self.multimodal_encoder.bert(
            input_ids = input_ids,
            attention_mask = attention_mask,
        ).last_hidden_state

        caption_output_pooled = self.pool_text_for_contra(caption_output)
        feat_t = self.contra_head_t(caption_output_pooled) 
        feat_t = F.normalize(feat_t,dim=-1)
        return feat_t

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

    def _vision_output(self, batch):
        vision_pixels = batch.vision_pixels
        b,n,_,h,w = vision_pixels.shape
        vision_output = self.vision_encoder.visual(vision_pixels.reshape(b*n,3,h,w), return_all_features=True)
        vision_output = vision_output.reshape(b,-1,*vision_output.shape[-2:])

        return vision_output

    def _audio_output(self, batch):
        audio_spectrograms = batch.audio_spectrograms
        b,n,h,w, = audio_spectrograms.shape
        audio_spectrograms = audio_spectrograms.reshape(-1,*audio_spectrograms.shape[2:])
        audio_output = self.audio_encoder(audio_spectrograms)
        audio_output = audio_output.reshape(b,n,-1,audio_output.shape[-1])
        return audio_output

    def _subtitle_output(self, batch):
        subtitle_tokens = self.multimodal_encoder.tokenizer(
            batch.raw_subtitles,
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

    def forward(self, batch):
        batch = edict(batch)

        caption_tokens = self._caption_tokens(batch)

        evaluation_dict = {}
        evaluation_dict['feat_t'] = self._feat_t(caption_tokens)
        evaluation_dict['input_ids'] = caption_tokens.input_ids
        evaluation_dict['attention_mask'] = caption_tokens.attention_mask

        vision = self._vision_output(batch)
        audio = self._audio_output(batch)
        subtitle = self._subtitle_output(batch)

        #### compute_itc
        evaluation_dict[f'feat_cond_tvas'] = self._feat_vas(vision, audio, subtitle)
        evaluation_dict[f'condition_feats_tvas'] = self._condition_feats_vas(vision, audio, subtitle)

        return evaluation_dict

    def compute_slice_scores(self, slice_multimodal_vision_input, slice_input_ids, slice_attention_mask):
        slice_output = self.multimodal_encoder.bert(
            input_ids = slice_input_ids,
            attention_mask = slice_attention_mask,
            encoder_hidden_states=slice_multimodal_vision_input,
        ).last_hidden_state
        slice_scores = F.softmax(self.itm_head(slice_output[:,0]),dim=1)[:,1]

        return slice_scores
