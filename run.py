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
from utils.args import (
    get_args,
    logging_cfgs,
)
from utils.build_dataloader import (
    create_val_dataloaders,
)
from utils.initialize import (
    initialize,
)


def main():

    ### init 
    args = get_args()
    initialize(args)

    ### logging cfgs
    logging_cfgs(args)

    gpu = torch.device("cuda")

    assert args.run_cfg.checkpoint
    ### build model
    model = VAST(args.model_cfg)
    checkpoint = torch.load(args.run_cfg.checkpoint)
    # checkpoint["position_ids"] = checkpoint["multimodal_encoder.bert.embeddings.position_ids"]
    # del checkpoint["multimodal_encoder.bert.embeddings.position_ids"]
    for k in [k for k in checkpoint if "vision_encoder.text" in k]:
            del checkpoint[k]

    replacement = []
    for k in checkpoint:
        new_k = k
        new_k = new_k.replace("mlp.fc1", "mlp.0")
        new_k = new_k.replace("mlp.act", "mlp.1")
        new_k = new_k.replace("mlp.fc2", "mlp.3")

        if k != new_k:
            replacement.append((k, new_k))

    for (k, new_k) in replacement:
        checkpoint[new_k] = checkpoint[k]
        del checkpoint[k]

    model.load_state_dict(checkpoint)
    model.to(gpu)
    model.eval()

    text_model = VastTextEncoder(args.model_cfg)
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


if __name__ == "__main__":
    main()
