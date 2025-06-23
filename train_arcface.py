import pickle
import json
import torch
import torch.nn as nn
from tqdm import tqdm
import utils.config as config
from torch.nn import functional as F
import numpy as np
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from matplotlib.patches import Ellipse

writer_tsne = SummaryWriter("runs/tsne")
from modules.proco import ProCoLoss
from modules.logitadjust import LogitAdjust


def compute_supcon_loss(feats, qtype):
    tau = 1.0
    if isinstance(qtype, tuple):
        i = 0
        dic = {}
        for item in qtype:
            if item not in dic:
                dic[item] = i
                i = i + 1
        tau = 1.0
        qtype = torch.tensor([dic[item] for item in qtype]).cuda()
    feats_filt = F.normalize(feats, dim=1)
    targets_r = qtype.reshape(-1, 1)
    targets_c = qtype.reshape(1, -1)
    mask = targets_r == targets_c
    mask = mask.int().cuda()
    feats_sim = torch.exp(torch.matmul(feats_filt, feats_filt.T) / tau)
    negatives = feats_sim * (1.0 - mask)
    negative_sum = torch.sum(negatives)
    positives = torch.log(feats_sim / negative_sum) * mask
    positive_sum = torch.sum(positives)
    positive_sum = positive_sum / torch.sum(mask)

    sup_con_loss = -1 * torch.mean(positive_sum)
    return sup_con_loss


def compute_acc(logits, labels):
    pred = torch.argmax(logits, dim=1)
    pred = pred.detach().cpu().numpy()
    score = pred == np.array(labels)
    tot_correct = score.sum()
    return tot_correct


def compute_score_with_logits(logits, labels):
    _, log_index = logits.max(dim=1, keepdim=True)
    scores = labels.gather(dim=1, index=log_index)
    return scores


def compute_loss(output, labels):

    # Function for calculating loss

    ce_loss = nn.CrossEntropyLoss(reduction="mean")(output, labels.squeeze(-1).long())

    return ce_loss


def saved_for_eval(dataloader, results, question_ids, answer_preds):
    """Save as a format accepted by the evaluation server."""
    _, answer_ids = answer_preds.max(dim=1)
    answers = [dataloader.dataset.label2ans[i] for i in answer_ids]
    for q, a in zip(question_ids, answers):
        entry = {
            "question_id": q.item(),
            "answer": a,
        }
        results.append(entry)
    return results

def train(model, m_model, optim, train_loader, train_dset, loss_fn, tracker, writer, tb_count, epoch, args):
    loader = tqdm(train_loader, ncols=0)
    loss_trk = tracker.track("loss", tracker.MovingMeanMonitor(momentum=0.99))
    acc_trk = tracker.track("acc", tracker.MovingMeanMonitor(momentum=0.99))
    cls_num_list = train_dset.cls_num_list
    criterion_scl = ProCoLoss(contrast_dim=1024, temperature=0.1, num_classes=len(cls_num_list)).cuda(args.gpu)
    logitAdjust = LogitAdjust(cls_num_list).cuda(args.gpu)
    if hasattr(criterion_scl, "_hook_before_epoch"):
        criterion_scl._hook_before_epoch(epoch, args.epochs)

    for v, q, a, mg, bias, q_id, f1, qtype, ans_type in loader:
        v = v.cuda()
        q = q.cuda()
        a = a.cuda()
        ans = []
        ans_tokens = []
        ans_index = torch.argmax(a, dim=1, keepdim=True).data.cpu()
        for index in ans_index:
            ans.append(train_loader.dataset.label2ans[index])
        for w in ans:
            if w not in train_dset.dictionary.word2idx:
                ans_tokens.append(18455)
            else:
                ans_tokens.append(train_dset.dictionary.word2idx[w])
        # print('ans_tokens',ans_tokens)
        ans_tokens = torch.from_numpy(np.array(ans_tokens))
        ans_tokens = Variable(ans_tokens).cuda()

        mg = mg.cuda()
        bias = bias.cuda()
        hidden_, ce_logits, a_emb, loss_CEI, q_repr = model(v, q, ans_tokens, a, epoch)
        hidden, pred, loss_DDL = m_model(hidden_, ce_logits, mg, epoch, a, a_emb, q_repr)
        f1 = f1.cuda()
        dict_args = {"margin": mg, "bias": bias, "hidden": hidden, "epoch": epoch, "per": f1}
        gt = torch.argmax(a, 1)

        
        # If bias-injection or learnable margins is enabled.
        if config.learnable_margins or config.bias_inject:
            # Use cross entropy loss to train the bias-injecting module
            ce_loss = -F.log_softmax(ce_logits, dim=-1) * a
            ce_loss = ce_loss * f1
            loss_BII = ce_loss.sum(dim=-1).mean()
            loss_ESD = loss_BII + loss_CEI
            loss_AAM = loss_fn(hidden, a, **dict_args)
        else:
            loss_AAM = loss_fn(hidden, a, **dict_args)

        contrast_logits = criterion_scl(hidden_, gt, args=args)
        # loss_ICL = F.cross_entropy(contrast_logits, gt)
        loss_ICL = logitAdjust(contrast_logits, gt)
        # Add the supcon loss, as mentioned in Section 3 of main paper.
        if config.supcon:
            loss = compute_supcon_loss(hidden_, gt) + loss_AAM + args.lambda1 * loss_ESD + args.lambda2 * loss_ICL + args.lambda3 * loss_DDL
        writer.add_scalars("data/losses", {}, tb_count)
        tb_count += 1

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optim.step()
        optim.zero_grad()

        # Ensemble the logit heads, as mentioned in Section 3 of the main paper, if bias-injection is enabled
        if config.bias_inject or config.learnable_margins:
            ce_logits = F.normalize(ce_logits)
            pred_l = F.normalize(pred)
            pred = (ce_logits + pred_l) / 2
        batch_score = compute_score_with_logits(pred, a.data)

        fmt = "{:.4f}".format
        loss_trk.append(loss.item())
        acc_trk.append(batch_score.mean())
        loader.set_postfix(loss=fmt(loss_trk.mean.value), acc=fmt(acc_trk.mean.value))

    return tb_count


# Evaluation code
def evaluate(model, m_model, dataloader, eval_dset, epoch=0, write=False):
    score = 0
    upper_bound = 0
    results = []  # saving for evaluation
    qt_score = {}
    qt_tot = {}
    for v, q, a, mg, _, q_id, _, qtype, ans_type in tqdm(dataloader, ncols=0, leave=True):
        v = v.cuda()
        q = q.cuda()
        mg = mg.cuda()
        a = a.cuda()
        ans = []
        ans_tokens = []
        ans_index = torch.argmax(a, dim=1, keepdim=True).data.cpu()
        for index in ans_index:
            ans.append(dataloader.dataset.label2ans[index])
        for w in ans:
            if w not in eval_dset.dictionary.word2idx:
                ans_tokens.append(18455)
            else:
                ans_tokens.append(eval_dset.dictionary.word2idx[w])
        # print('ans_tokens',ans_tokens)
        ans_tokens = torch.from_numpy(np.array(ans_tokens))
        ans_tokens = Variable(ans_tokens).cuda()
        hidden, ce_logits, a_emb, loss_self, q_repr = model(v, q, ans_tokens, a, epoch)
        hidden_, pred, _ = m_model(hidden, ce_logits, mg, epoch, a, a_emb, q_repr)

        # Ensemble the logit heads
        if config.learnable_margins or config.bias_inject:
            ce_logits = F.softmax(F.normalize(ce_logits) / config.temp, 1)
            pred_l = F.softmax(F.normalize(pred), 1)
            pred = config.alpha * pred_l + (1 - config.alpha) * ce_logits
        if write:
            results = saved_for_eval(dataloader, results, q_id, pred)
        batch_score = compute_score_with_logits(pred, a.cuda()).cpu().numpy().sum(1)

        score += batch_score.sum()
        upper_bound += (a.max(1)[0]).sum()
        qids = q_id.detach().cpu().int().numpy()
        for j in range(len(qids)):
            qt_score[qtype[j]] = qt_score.get(qtype[j], 0) + batch_score[j]
            qt_tot[qtype[j]] = qt_tot.get(qtype[j], 0) + 1
            qt_score[ans_type[j]] = qt_score.get(ans_type[j], 0) + batch_score[j]
            qt_tot[ans_type[j]] = qt_tot.get(ans_type[j], 0) + 1
            

    print(score, len(dataloader.dataset))
    print('upper_bound', upper_bound / len(dataloader.dataset))
    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)

    # if write:
    #     print("saving prediction results to disk...")
    #     result_file = "vqa_{}_{}_{}_{}_results.json".format(config.task, config.test_split, config.version, epoch)
    #     with open(result_file, "w") as fd:
    #         json.dump(results, fd)
    print(score)
    print("--------score------", score)
    for qt, qtype_score in qt_score.items():
        print(qt + ':   ', qtype_score / qt_tot[qt] )
    return score
