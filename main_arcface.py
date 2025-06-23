import os
import sys
import json
import argparse
from pprint import pprint

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import datetime
import utils.utils as utils
import utils.config as config
from train_arcface import train, evaluate
import modules.base_model_arcface as base_model
from utils.dataset import Dictionary, VQAFeatureDataset
from utils.losses import Plain


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=45,
                        help='number of running epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate for adamax')
    parser.add_argument('--loss-fn', type=str, default='Plain',
                        help='chosen loss function')
    parser.add_argument('--num-hid', type=int, default=1024,
                        help='number of dimension in last layer')
    parser.add_argument('--model', type=str, default='baseline_newatt',
                        help='model structure')
    parser.add_argument('--name', type=str, default='exp0.pth',
                        help='saved model name')
    parser.add_argument('--name-new', type=str, default=None,
                        help='combine with fine-tune')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='training batch size')
    parser.add_argument('--fine-tune', action='store_true',
                        help='fine tuning with our loss')
    parser.add_argument('--resume', action='store_true',
                        help='whether resume from checkpoint')
    parser.add_argument('--not-save', action='store_true',
                        help='do not overwrite the old model')
    parser.add_argument('--test', dest='test_only', action='store_true',
                        help='test one time')
    parser.add_argument('--eval-only', action='store_true',
                        help='evaluate on the val set one time')
    parser.add_argument("--gpu", type=str, default='0',
                        help='gpu card ID')
    parser.add_argument('--lambda1', type=float, default=1.0,
                        help='lambda for the loss_ESD')
    parser.add_argument('--lambda2', type=float, default=2.0,
                        help='lambda for the loss_ICL')
    parser.add_argument('--lambda3', type=float, default=0.05,
                        help='lambda for the loss_DDL')
    parser.add_argument(
        "--dataset",
        default="slake",
        choices=["slake", "slake-cp", "vg", "vizwiz", "vqacp-v2", "vqace", "vqacp-v1", "vqa-v2", "gqaood"],
        help="choose dataset",
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    now = datetime.datetime.now()
    start_time = now.strftime('%Y-%m-%d %H:%M:%S')
    print("开始时间:", start_time)
    args = parse_args()
    dataset = args.dataset
    config.dataset = dataset
    config.update_paths(args.dataset)
    if dataset in ['slake', 'slake-cp', 'vqa-rad', 'vqa-rad-cp']:
        args.batch_size = 64
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cudnn.benchmark = True

    seed = 1111
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    if 'log' not in args.name:
        args.name = 'logs/' + args.name
    if args.test_only or args.fine_tune or args.eval_only:
        args.resume = True
    if args.resume and not args.name:
        raise ValueError("Resuming requires folder name!")
    if args.resume:
        logs = torch.load(args.name)
        print("loading logs from {}".format(args.name))

    # ------------------------DATASET CREATION--------------------
    dictionary = Dictionary.load_from_file(config.dict_path)
    if args.test_only:
        eval_dset = VQAFeatureDataset('test', dictionary, dataset)
    else:
        train_dset = VQAFeatureDataset('train', dictionary, dataset)
        eval_dset = VQAFeatureDataset('test', dictionary, dataset)
    if config.train_set == 'train+val' and not args.test_only:
        train_dset = train_dset + eval_dset
        eval_dset = VQAFeatureDataset('test', dictionary)
    if args.eval_only:
        eval_dset = VQAFeatureDataset('val', dictionary)

    tb_count = 0
    writer = SummaryWriter() # for visualization

    if not config.train_set == 'train+val' and 'LM' in args.loss_fn:
        utils.append_bias(train_dset, eval_dset, len(eval_dset.label2ans))

    # ------------------------MODEL CREATION------------------------
    constructor = 'build_{}'.format(args.model)
    model, metric_fc = getattr(base_model, constructor)(eval_dset, args.num_hid)
    model = model.cuda()
    metric_fc = metric_fc.cuda()
    model.w_emb.init_embedding(config.glove_embed_path)

    # model = nn.DataParallel(model).cuda()
    optim = torch.optim.Adamax([{'params': model.parameters()}, {'params': metric_fc.parameters()}], lr=args.lr)

    if args.loss_fn == 'Plain':
        loss_fn = Plain()
    else:
        raise RuntimeError('not implement for {}'.format(args.loss_fn))

    # ------------------------STATE CREATION------------------------
    eval_score, best_val_score, start_epoch, best_epoch = 0.0, 0.0, 0, 0
    tracker = utils.Tracker()
    if args.resume:
        model.load_state_dict(logs['model_state'])
        metric_fc.load_state_dict(logs['margin_model_state'])
        optim.load_state_dict(logs['optim_state'])
        if 'loss_state' in logs:
            loss_fn.load_state_dict(logs['loss_state'])
        start_epoch = logs['epoch']
        best_epoch = logs['epoch']
        best_val_score = logs['best_val_score']
        if args.fine_tune:
            print('best accuracy is {:.2f} in baseline'.format(100 * best_val_score))
            args.epochs = start_epoch + 10 # 10 more epochs
            for params in optim.param_groups:
                params['lr'] = config.ft_lr

            # if you want save your model with a new name
            if args.name_new:
                if 'log' not in args.name_new:
                    args.name = 'logs/' + args.name_new
                else:
                    args.name = args.name_new

    eval_loader = DataLoader(eval_dset,
                    args.batch_size, shuffle=False, num_workers=4)
    if args.test_only or args.eval_only:
        model.eval()
        metric_fc.eval()
        evaluate(model, metric_fc, eval_loader,eval_dset, write=True)
    else:
        train_loader = DataLoader(
            train_dset, args.batch_size, shuffle=True, num_workers=4)
        for epoch in range(start_epoch, args.epochs):
            print("training epoch {:03d}".format(epoch))
            tb_count = train(model, metric_fc, optim, train_loader,train_dset, loss_fn, tracker, writer, tb_count, epoch, args)

            if not (config.train_set == 'train+val' and epoch in range(args.epochs - 3)):
                # save for the last three epochs
                write = True if config.train_set == 'train+val' else False
                print("validating after epoch {:03d}".format(epoch))
                model.train(False)
                metric_fc.train(False)
                eval_score = evaluate(model, metric_fc, eval_loader,eval_dset, epoch, write=write)
                model.train(True)
                metric_fc.train(True)
                print("eval score: {:.2f} \n".format(100 * eval_score))

            if eval_score > best_val_score:
                best_val_score = eval_score
                best_epoch = epoch
                results = {
                    'epoch': epoch + 1,
                    'best_val_score': best_val_score,
                    'model_state': model.state_dict(),
                    'optim_state': optim.state_dict(),
                    'loss_state': loss_fn.state_dict(),
                    'margin_model_state': metric_fc.state_dict()
                }
                if not args.not_save:
                    torch.save(results, args.name)
        print("best accuracy {:.2f} on epoch {:03d}".format(
            100 * best_val_score, best_epoch))
    now = datetime.datetime.now()
    end_time = now.strftime('%Y-%m-%d %H:%M:%S')
    print("结束时间:", end_time)