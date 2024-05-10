import itertools
import json
import os
from time import (
    time
)

import numpy as np
import torch
import torch.distributed as dist
from easydict import EasyDict as edict
from torch.nn import functional as F
from tqdm import (
    tqdm
)

from evaluation_tools.caption_tools.pycocoevalcap.eval import (
    COCOEvalCap
)
from evaluation_tools.caption_tools.pycocotools.coco import (
    COCO
)
from evaluation_tools.vqa_tools.vqa import (
    VQA
)
from evaluation_tools.vqa_tools.vqa_eval import (
    VQAEval
)
from utils.distributed import (
    all_gather_list,
    ddp_allgather
)
from utils.logger import (
    LOGGER
)
from utils.tool import (
    NoOp
)

cpu_device = torch.device("cpu")


def evaluate_mm(model, loader, run_cfg, global_step):
    eval_log = {}
    model.eval()
    LOGGER.info(f"evaluate task")
    val_log = evaluate_single(model, loader, "ret%tvas", run_cfg, global_step, "msrvtt_ret")
    eval_log["ret%tvas"] = val_log
    model.train()
    return eval_log


@torch.no_grad()
def evaluate_single(model, val_loader, task, run_cfg, global_step,dset_name):
    LOGGER.info("start running {} validation...".format(task))
    return evaluate_ret(model, task, val_loader, global_step)

@torch.no_grad()
def evaluate_ret(model, tasks, val_loader, global_step):
    val_log = {}
    ids = []
    ids_txt = []
    input_ids = []
    attention_mask = []
 
    feat_t = []

    feat_cond = []
    condition_feats = []

    for _, batch in enumerate(val_loader):
        batch = edict(batch)
        evaluation_dict= model(batch)
        feat_t.append(evaluation_dict['feat_t'])

        input_ids.append(evaluation_dict['input_ids'])
        attention_mask.append(evaluation_dict['attention_mask'])

        ids += batch.ids
        ids_txt += list(itertools.chain.from_iterable(batch.ids_txt))

        feat_cond.append(evaluation_dict['feat_cond_tvas'])
        # condition_feats.append(evaluation_dict['condition_feats_tvas'])

    feat_t = torch.cat(feat_t, dim=0)
    input_ids = torch.cat(input_ids, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)

    score_matrix_t_cond = {}
    ### compute itc_score
    feat_cond =  torch.cat(feat_cond, dim = 0)
    score_matrix_t_cond = torch.matmul(feat_t, feat_cond.permute(1,0))
    log = compute_metric_ret(score_matrix_t_cond, ids, ids_txt)
    log = {k.replace('forward','video'): v for k,v in log.items()}
    val_log[f'ret_itc_tvas'] = log


    #### compute itm_score
    # for task in subtasks:
    #     store_dict[f'condition_feats_{task}'] = torch.cat(store_dict[f'condition_feats_{task}'],dim=0)
    #     itm_rerank_num = model.config.itm_rerank_num
    #     score_matrix = refine_score_matrix(store_dict[f'condition_feats_{task}'], input_ids, attention_mask, store_dict[f'score_matrix_t_cond_{task}'], model, itm_rerank_num, direction='forward')
    #     log = compute_metric_ret(score_matrix, ids, ids_txt, direction='forward')
    #     log = {k.replace('forward','video'): v for k,v in log.items()}

    #     if model.config.ret_bidirection_evaluation:
    #         score_matrix = refine_score_matrix(store_dict[f'condition_feats_{task}'], input_ids, attention_mask, store_dict[f'score_matrix_t_cond_{task}'], model, itm_rerank_num, direction='backward')
    #         log2 = compute_metric_ret(score_matrix, ids, ids_txt, direction='backward')
    #         log2 = {k.replace('backward','txt'): v for k,v in log2.items()}
    #         log.update(log2)

    #     val_log[f'ret_itm_{task}'] = log

    return val_log

def refine_score_matrix(condition_feats, input_ids, attention_mask, score_matrix_t_cond, model, itm_rerank_num, direction='forward'):

    top_k = itm_rerank_num
    if direction=='forward':
        idxs = score_matrix_t_cond.topk(top_k,dim=1)[1]
    else:
        idxs = score_matrix_t_cond.topk(top_k,dim=0)[1]
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    nums = score_matrix_t_cond.shape[0]//world_size +1
    
    score_matrix_t_cond_new = torch.zeros_like(score_matrix_t_cond)
    idxs_new = torch.zeros_like(score_matrix_t_cond_new).long()
    if direction=='forward':
        for i in range(len(idxs)):
            for j in idxs[i]:
                idxs_new[i][j] = 1
    else:
        for i in range(idxs.shape[1]):
            for j in idxs[:,i]:
                idxs_new[j][i] = 1
    cur_length = condition_feats.shape[0]
    length_ls = all_gather_list(cur_length)
    start = 0
    start_ls = []
    end_ls = []
    for l in range(len(length_ls)):
        start_ls.append(start)
        end_ls.append(start+length_ls[l])
        start = start+length_ls[l]
    
    cur_score_matrix_t_cond = score_matrix_t_cond[:,start_ls[rank]:end_ls[rank]]
    cur_score_matrix_t_cond_new = score_matrix_t_cond_new[:,start_ls[rank]:end_ls[rank]]
    cur_idxs_new = idxs_new[:,start_ls[rank]:end_ls[rank]]

    if dist.get_rank() == 0:
        pbar = tqdm(total=cur_length)
    else:
        pbar = NoOp()
    for i in range(cur_length):
        if sum(cur_idxs_new[:,i] == 1) == 0:
            continue
        cur_scores = []
        cur_input_ids = input_ids[(cur_idxs_new[:,i] == 1)]
        cur_attention_mask = attention_mask[(cur_idxs_new[:,i] == 1)]
        

        cur_condition_feats = condition_feats[i].unsqueeze(0).expand(cur_input_ids.shape[0],-1,-1)
        total_len = len(cur_condition_feats)
        small_batch=25
        times = total_len//small_batch if total_len%small_batch==0 else total_len//small_batch+1

        for k in range(times):
            slice_input_ids = cur_input_ids[k*small_batch:(k+1)*small_batch]
            slice_attention_mask = cur_attention_mask[k*small_batch:(k+1)*small_batch]
            slice_condition_feats = cur_condition_feats[k*small_batch:(k+1)*small_batch]
            slice_scores = model.compute_slice_scores(slice_condition_feats, slice_input_ids, slice_attention_mask) 
            cur_scores.append(slice_scores)
        cur_scores = torch.cat(cur_scores,dim=0)

        cur_score_matrix_t_cond_new[:,i][(cur_idxs_new[:,i] == 1)] = cur_scores
        pbar.update(1)
    pbar.close()
    
    score_matrix_t_cond = ddp_allgather(cur_score_matrix_t_cond_new.T.contiguous()).T

    return score_matrix_t_cond






def compute_metric_ret(score_matrix, ids, ids_txt):
    assert score_matrix.shape == (len(ids_txt), len(ids))

    indice_matrix = score_matrix.sort(dim=-1, descending=True)[1].tolist()
    rank = []
    for i in range(len(ids_txt)):
        gt_indice = ids.index(ids_txt[i])
        rank.append(indice_matrix[i].index(gt_indice))

    rank = torch.tensor(rank).to(score_matrix)
    
    vr_r1 = (rank < 1).sum().item() / len(ids_txt)
    vr_r5 = (rank < 5).sum().item() / len(ids_txt)
    vr_r10 = (rank < 10).sum().item() / len(ids_txt)

    eval_log = {
        'forward_r1': round(vr_r1*100,1),
        'forward_recall': f'{round(vr_r1*100,1)}/{round(vr_r5*100,1)}/{round(vr_r10*100,1)}',
        'forward_ravg': round((vr_r1 + vr_r5 + vr_r10)/3 *100,1)
    }

    return eval_log

def compute_metric_cap(results, annfile_path, process=True):
    coco = COCO(annfile_path)
    cocoRes = coco.loadRes(results)
    cocoEval = COCOEvalCap(coco, cocoRes, process)
    cocoEval.evaluate()
    metric = cocoEval.eval
    metric = {k: round(v*100,2)  for k,v in metric.items()}
    return metric
