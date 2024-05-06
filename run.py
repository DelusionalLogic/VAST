import torch
from torch.utils.data import (
    DataLoader,
)

from data.IndexAnno import (
    AnnoIndexedDataset,
    collate,
)
from evaluation.evaluation_mm import (
    evaluate_ret,
)
from model.vast import (
    VAST,
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
    # for k, v in checkpoint.items():
    #     if k == ""
    model.load_state_dict(checkpoint)
    model.to(gpu)
    model.eval()

    ### Open the dataset
    val_loaders = DataLoader(
        AnnoIndexedDataset(),
        batch_size = 12,
        collate_fn = collate,
    )

    ### start evaluation
    val_log = evaluate_ret(model, "ret%tvas", val_loaders, 0)
    print(f"==== evaluation--msrvtt_ret========\n")
    print(val_log)


if __name__ == "__main__":
    main()
