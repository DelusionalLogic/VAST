import torch


@torch.no_grad()
def evaluate_ret(model, text_model, video_loader, caption_loader):
    val_log = {}
 
    feat_t = []
    feat_cond = []

    caption_ids = []
    video_ids = []

    for _, batch in enumerate(caption_loader):
        caption_ids += batch["ids"]

        text_features = text_model(batch["raw_captions"])
        feat_t.append(text_features)

    feat_t = torch.cat(feat_t, dim=0).cpu()

    for _, batch in enumerate(video_loader):
        video_ids += batch["ids"]

        vision = model._vision_output(batch["vision_pixels"])
        audio = model._audio_output(batch["audio_spectrograms"])
        # subtitle = torch.zeros(len(batch["ids"]), 1, 768).cuda()
        subtitle = model._subtitle_output(batch["raw_subtitles"])
        feat_cond.append(model._feat_vas(vision, audio, subtitle))

    assert caption_ids == video_ids

    feat_t = feat_t.cuda()

    score_matrix_t_cond = {}
    feat_cond =  torch.cat(feat_cond, dim = 0)
    score_matrix_t_cond = torch.matmul(feat_t, feat_cond.permute(1,0))
    log = compute_metric_ret(score_matrix_t_cond, caption_ids, video_ids)
    val_log[f'ret_itc_tvas'] = log

    return val_log

def compute_metric_ret(score_matrix, caption_ids, video_ids):
    assert score_matrix.shape == (len(caption_ids), len(video_ids))

    indice_matrix = score_matrix.sort(dim=-1, descending=True)[1].tolist()
    rank = []
    for i in range(len(caption_ids)):
        gt_indice = video_ids.index(caption_ids[i])
        rank.append(indice_matrix[i].index(gt_indice))

    rank = torch.tensor(rank).to(score_matrix)

    vr_r1 = (rank < 1).sum().item() / len(caption_ids)
    vr_r5 = (rank < 5).sum().item() / len(caption_ids)
    vr_r10 = (rank < 10).sum().item() / len(caption_ids)

    eval_log = {
        'forward_r1': round(vr_r1*100,1),
        'forward_recall': f'{round(vr_r1*100,1)}/{round(vr_r5*100,1)}/{round(vr_r10*100,1)}',
        'forward_ravg': round((vr_r1 + vr_r5 + vr_r10)/3 *100,1)
    }

    return eval_log
