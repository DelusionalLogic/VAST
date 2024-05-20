import random

import coverage
import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
)

from data.IndexAnno import (
    Captions,
    Videos,
)
from evaluation.evaluation_mm import (
    evaluate_ret,
)
from model.vast import (
    VAST,
    VastTextEncoder,
)


def main():

    # cov = coverage.Coverage()
    # cov.start()

    random.seed(50)
    np.random.seed(50)
    torch.manual_seed(50)
    torch.cuda.manual_seed_all(50)
    torch.backends.cudnn.benchmark = True

    gpu = torch.device("cuda")

    ### build model
    model = VAST()
    checkpoint = torch.load("output/vast/pretrain_vast/downstream/retrieval-msrvtt/ckpt/model_step_10109.pt")
    # checkpoint["position_ids"] = checkpoint["multimodal_encoder.bert.embeddings.position_ids"]
    # del checkpoint["multimodal_encoder.bert.embeddings.position_ids"]
    for k in [k for k in checkpoint if "vision_encoder.text" in k]:
            del checkpoint[k]
    for k in [k for k in checkpoint if "multimodal_encoder.cls" in k]:
            del checkpoint[k]
    for k in [k for k in checkpoint if "multimodal_encoder.bert.encoder.layer" in k and "crossattention" in k]:
            del checkpoint[k]

    replacement = []
    for k in checkpoint:
        new_k = k
        new_k = new_k.replace("mlp.fc1", "mlp.0")
        new_k = new_k.replace("mlp.act", "mlp.1")
        new_k = new_k.replace("mlp.fc2", "mlp.3")

        new_k = new_k.replace(".intermediate.dense", ".intermediate")
        new_k = new_k.replace(".attention.self.", ".")
        new_k = new_k.replace(".attention.output.dense", ".attention_dense")
        new_k = new_k.replace(".attention.output.LayerNorm", ".attention_norm")
        new_k = new_k.replace(".output.dense", ".output")
        new_k = new_k.replace(".output.LayerNorm", ".output_norm")

        new_k = new_k.replace(".embeddings.LayerNorm", ".embeddings_norm")
        new_k = new_k.replace(".embeddings.position_ids", ".position_ids")
        new_k = new_k.replace(".embeddings.word_embeddings", ".word_embeddings")
        new_k = new_k.replace(".embeddings.position_embeddings", ".position_embeddings")
        new_k = new_k.replace(".embeddings.token_type_embeddings", ".token_type_embeddings")
        new_k = new_k.replace(".bert.encoder.layer", ".bert.layer")
        new_k = new_k.replace("multimodal_encoder.bert.", "multimodal_encoder.")

        new_k = new_k.replace("audio_encoder.encoder.layer_norm", "audio_encoder.post_norm")
        new_k = new_k.replace("audio_encoder.encoder.", "audio_encoder.")

        new_k = new_k.replace(".self_attn.", ".")

        new_k = new_k.replace("audio_encoder.layers.0.relative_attention_bias", "audio_encoder.relative_attention_bias")
        new_k = new_k.replace("itm_head.linear1", "itm_head.0")
        new_k = new_k.replace("itm_head.layernorm", "itm_head.2")
        new_k = new_k.replace("itm_head.linear2", "itm_head.3")

        new_k = new_k.replace("contra_head_t.linear.", "contra_head_t.")
        new_k = new_k.replace("contra_head_s.linear.", "contra_head_s.")
        new_k = new_k.replace("contra_head_v.linear.", "contra_head_v.")
        new_k = new_k.replace("contra_head_a.linear.", "contra_head_a.")

        new_k = new_k.replace("audio_encoder.pos_conv.0.", "audio_encoder.pos_conv.")

        new_k = new_k.replace(".final_layer_norm.", ".output_norm.")
        new_k = new_k.replace(".self_attn_layer_norm.", ".attention_norm.")

        if k != new_k:
            replacement.append((k, new_k))

    for (k, new_k) in replacement:
        checkpoint[new_k] = checkpoint[k]
        del checkpoint[k]

    for k in [k for k in checkpoint if "audio_encoder.layers." in k and ".relative_attention_bias." in k]:
            del checkpoint[k]

    model.load_state_dict(checkpoint)
    model.to(gpu)
    model.eval()

    text_model = VastTextEncoder()
    text_model.load_state_dict(checkpoint, strict=False)
    text_model.to(gpu)
    text_model.eval()

    ### Open the dataset
    val_loaders = DataLoader(
        Videos(),
        batch_size = 12,
    )

    caption_loader = DataLoader(
        Captions(),
        batch_size = 512,
    )

    ### start evaluation
    val_log = evaluate_ret(model, text_model, val_loaders, caption_loader)
    print(f"==== evaluation--msrvtt_ret========\n")
    print(val_log)

#     cov.stop()
#     cov.save()
#     cov.html_report()


if __name__ == "__main__":
    main()
