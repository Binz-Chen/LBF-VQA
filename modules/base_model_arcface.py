import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
import math
import utils.config as config
from modules.fc import FCNet
from modules.classifier import SimpleClassifier
from modules.attention import Attention, NewAttention
from modules.language_model import WordEmbedding, QuestionEmbedding
from collections import Counter
from modules.utils_k import GradReverse, TopK_custom
from torch.nn import CosineSimilarity, Parameter
from torch.autograd import Variable
import numpy as np
# self-loss
import random
import torch.nn.init as init


class squeeze(nn.Module):
    def __init__(self):
        super(squeeze, self).__init__()

    def forward(self, input):
        return input.squeeze()


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


class LogitNormLoss(nn.Module):

    def __init__(self, device, t=1.0):
        super(LogitNormLoss, self).__init__()
        self.device = device
        self.t = t

    def forward(self, x, target):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-7
        logit_norm = torch.div(x, norms) / self.t
        return F.cross_entropy(logit_norm, target)


def compute_self_loss(logits_neg, a):
    prediction_ans_k, top_ans_ind = torch.topk(
        F.softmax(a, dim=-1), k=1, dim=-1, sorted=False)
    # print('-----prediction_ans_k--------',prediction_ans_k.shape)
    neg_top_k = torch.gather(
        F.softmax(logits_neg, dim=-1), 1, top_ans_ind).sum(1)
    # print('----neg_top_k--------', neg_top_k.shape)
    qice_loss = neg_top_k.mean()
    return qice_loss


def geo_similarity(zi, zj):
    # Calculate the standard inner product between zi and zj
    inner_product = torch.sum(zi * zj, dim=1)

    # Normalize the vectors and calculate the cosine similarity
    cos_sim = inner_product / (torch.norm(zi) * torch.norm(zj))

    # Calculate the angle between zi and zj in radians
    angle = torch.acos(cos_sim)

    # Convert the angle to a similarity score using the provided formula
    sim_score = 1.0 - angle / torch.tensor(3.141592653589793)

    return sim_score


class Contrastive_eur_loss(nn.Module):
    def __init__(self, tao=1.0):
        super(Contrastive_eur_loss, self).__init__()
        self.sim = CosineSimilarity(dim=-1)
        self.tao = 1.0

    def forward(self, fea, pos_fea, neg_fea):
        fea = F.normalize(fea, dim=1)
        pos_fea = F.normalize(pos_fea, dim=1)
        neg_fea = F.normalize(neg_fea, dim=1)

        pos_sim = self.sim(fea, pos_fea)
        neg_sim = self.sim(fea, neg_fea)
        pos_cos = torch.acos((fea * pos_fea).sum(dim=-1) / self.tao)
        neg_cos = torch.acos((fea * neg_fea).sum(dim=-1) / self.tao)
        logits = torch.exp(-pos_cos) / (torch.exp(-pos_cos) + torch.exp(-neg_cos))
        acosloss = (-1.0 * torch.log(logits))
        acosloss = acosloss.mean()
        logits = torch.exp(pos_sim / self.tao) / \
                 (torch.exp(pos_sim / self.tao) + torch.exp(neg_sim / self.tao))
        loss = (-1.0 * torch.log(logits))
        eurloss = loss.mean()
        return eurloss


class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, fusion, num_hid, num_class):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.weight = SimpleClassifier(num_hid, num_hid * 2, num_class, 0.5)
        self.weight_a = SimpleClassifier(300, 300 * 2, num_class, 0.5)
        self.weight_q = SimpleClassifier(num_hid, num_hid * 2, num_class, 0.5)
        self.topk = TopK_custom(k=8)
        self.text_scorer_net = nn.Sequential(nn.Linear(300, 1),
                                             nn.Sigmoid())

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.generate = nn.Sequential(
            *block(num_hid // 8, num_hid // 4),
            *block(num_hid // 4, num_hid // 2),
            *block(num_hid // 2, num_hid),
            nn.Linear(num_hid, num_hid * 2),
            nn.ReLU(inplace=True)
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    kaiming_init(m)
            except:
                kaiming_init(block)

    def compute_predict(self, q_repr, q_emb, v):
        att_gv = self.v_att(v, q_emb)
        gv_embs = (att_gv * v)  # [batch, v_dim]
        gv_emb = gv_embs.sum(1)
        gv_repr = self.v_net(gv_emb)
        joint_repr = q_repr * gv_repr
        logits = self.weight(joint_repr)
        out = logits

        return out, joint_repr, att_gv

    def forward(self, v, q, ans_tokens, a, epoch):
        """
        Forward=
        v: [batch, num_objs, obj_dim]
        q: [batch_size, seq_length]
        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb, _ = self.q_emb(w_emb)  # [batch, q_dim]

        att = self.v_att(v, q_emb)

        v_emb = (att * v).sum(1)  # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr  # Final multimodal features, denoted as x in the main paper. This is the UpDn model.
        # q_logit = self.weight_q(q_repr)
        # This is the bias injecting component, as shown in subsection 3.4 of the main paper
        ce_logits = self.weight(joint_repr)
        # q_logits = self.qweight(joint_repr)
        a_emb = self.w_emb(ans_tokens).view(ans_tokens.size(0), -1)  # [512, 300]
        # a_logits = self.weight_a(a_emb)  # [512, 3129]

        #
        if epoch > 8:

            batch_size = q.size(0)

            b, c, f = v.shape
            v_z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (b, c, 128))))
            # v_prompt = self.generate(v_z.view(-1, 128)).view(b, c, f)

            s1 = self.text_scorer_net(w_emb)
            scores1 = GradReverse.grad_reverse(s1.squeeze(-1), 0.01)
            w_emb_masked = (1 - self.topk(scores1).unsqueeze(-1)) * w_emb
            q_emb_masked, q_hidden_masked = self.q_emb(w_emb_masked)  # [batch, q_dim]
            # q_repr_mask = self.q_net(q_emb_masked)

            index_v = random.sample(range(0, batch_size), batch_size)
            gv_neg = v[index_v]
            out_neg_v, joint_neg_v, att_gv_neg_v = \
                self.compute_predict(q_repr, q_emb, gv_neg)

            index_q = random.sample(range(0, batch_size), batch_size)
            q_emb_neg = q_emb[index_q]
            q_repr_neg = q_repr[index_q]
            out_neg_q, joint_neg_q, att_q_neg_q = \
                self.compute_predict(q_emb_neg, q_repr_neg, v)

            self_loss_q = compute_self_loss(out_neg_q, a)
            self_loss_v = compute_self_loss(out_neg_v, a)
            self_loss = 0.7 * self_loss_q + self_loss_v
        else:
            self_loss = 0.0
        # self_loss = 0.0

        return joint_repr, ce_logits, a_emb, self_loss, q_repr


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """

    def __init__(self, in_features, out_features, s=config.scale, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_a = nn.Parameter(torch.FloatTensor(out_features, 300))
        self.weight_b = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight_a)
        self.Contrastive_eur_loss = Contrastive_eur_loss()
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.std = 0.1
        self.temp = config.temp

    def forward(self, input, learned_mg, m, epoch, label, a_emb, q_repr):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        
        if self.training is False:
            return cosine, cosine, 0
    
        cosine_a = F.linear(F.normalize(a_emb), F.normalize(self.weight_a))
        # cosine_b = F.linear(F.normalize(q_repr), F.normalize(self.weight_b))

        # Set beta (Subsecion 3.3 in main paper
        beta_factor = epoch // 15
        beta = 1.0 - (beta_factor * 0.1)

        # Calculate the learnable instance-level margins, Subsection 3.3 in main paper
        learned_mg = torch.where(m > 1e-12, learned_mg.double(), -1000.0).float()
        margin = F.softmax(learned_mg / self.temp, dim=1)

        # Perform randomization as mentioned in Section 3 of main paper
        if config.randomization:
            m = torch.normal(mean=m, std=self.std)

        # Combine the margins, as in Subsection 3.3 of main paper.
        if config.learnable_margins:
            m[label != 0] = beta * m[label != 0] + (1 - beta) * margin[label != 0]
        m = 1 - m

        # Compute the AdaArc angular margins and the corresponding logits
        self.cos_m = torch.cos(m)
        self.sin_m = torch.sin(m)
        self.th = torch.cos(math.pi - m)
        self.mm = torch.sin(math.pi - m) * m
        # --------------------------- cos(theta) & phi(theta) ---------------------------

        # cosine = input
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        output = phi * self.s

        log_prob_lab = F.log_softmax(output, dim=-1).double()
        log_prob_cosine_a = F.log_softmax(cosine_a, dim=-1).double()
        loss_kl = (
                          torch.mean(
                              torch.diag(torch.mm(torch.exp(log_prob_lab), log_prob_lab.t())).view(-1, 1) -
                              torch.mm(torch.exp(log_prob_lab), log_prob_cosine_a.t())
                          ) +
                          torch.mean(
                              torch.diag(torch.mm(torch.exp(log_prob_cosine_a), log_prob_cosine_a.t())).view(-1, 1) -
                              torch.mm(torch.exp(log_prob_cosine_a), log_prob_lab.t())
                          )
                  ) / 2

        return output, cosine, loss_kl


def l2_norm(input, dim=-1):
    norm = torch.norm(input, dim=dim, keepdim=True)
    output = torch.div(input, norm)
    return output


def build_baseline(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    fusion = FCNet([num_hid, num_hid * 2], dropout=0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net,
                     fusion, num_hid, dataset.num_ans_candidates)


def build_baseline_newatt(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    fusion = FCNet([num_hid, num_hid * 2], dropout=0.5)
    basemodel = BaseModel(w_emb, q_emb, v_att, q_net, v_net,
                          fusion, num_hid, dataset.num_ans_candidates)
    margin_model = ArcMarginProduct(num_hid, dataset.num_ans_candidates)
    return basemodel, margin_model
