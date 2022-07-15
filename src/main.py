import sys
import torch
import numpy as np
import config as cg

import val
import train

import torch.optim as optim
from prompt import text_prompt
from utils import FastDataLoader
from argparse import ArgumentParser
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from model import CLIPrompt


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # setup log and model path, initialize tensorboard,
    [logPath, modelPath] = cg.setup_path(args)
    args.model_path = modelPath
    logger = SummaryWriter(logdir=logPath)
    args.return_intermediate_text_feature = 0

    # initialize dataloader
    [trn_dataset, val_datasete, trainactions, valactions] = cg.setup_dataloader(args)
    trnloader = FastDataLoader(trn_dataset, batch_size=args.batchsize, num_workers=args.workers,
                               shuffle=True, pin_memory=False, drop_last=True)

    valloader = FastDataLoader(val_datasete, batch_size=args.batchsize, num_workers=args.workers,
                           shuffle=False, pin_memory=False, drop_last=False)

    # initialize models
    print('==> reading meta data for {}'.format(args.dataset))
    actionlist, actiondict, actiontoken = text_prompt(dataset=args.dataset, clipbackbone=args.backbone, device=device)

    print('==> initialising action recognition model')
    model = CLIPrompt(args, actionlist, actiondict, actiontoken, device)

    model.float()
    model.to(device)

    # initialize training
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01) 
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,  
                                                                        T_0=int(args.decay_steps), 
                                                                        eta_min=args.lr*0.01, 
                                                                        last_epoch=-1)
    args.start_iter = 0

    if args.test:
        print('loading checkpoint {}'.format(args.test))
        if args.test == 'random/random':
            iteration = 0
            print('loading random weights')
        else:
            checkpoint = torch.load(args.test, map_location=torch.device('cpu'))
            iteration = checkpoint['iteration']
            state_dict = checkpoint['state_dict']
            model.load_state_dict(state_dict)
            print('loading successful')
        val.val_CLIPrompt(args, valloader, [actionlist, actiondict, actiontoken, trainactions, valactions], model, None, device, iteration)
        print('test finish, exiting')
        sys.exit()

    if args.resume:
        print('loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        iteration = checkpoint['iteration']
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
        args.start_iter = iteration
        print('loading successful')

    print('======> start training {}, {}, use {}.'.format(args.dataset, args.verbose, device))

    train.train_CLIPrompt(args, [trnloader, valloader], [actionlist, actiondict, actiontoken, trainactions, valactions], model, optimizer, lr_scheduler, logger, device)



if __name__ == "__main__":
    parser = ArgumentParser()

    # optimization
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--prefix', type=int, default=0)
    parser.add_argument('--postfix', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--openset', type=int, default=0)
    parser.add_argument('--temporal', type=int, default=1)
    parser.add_argument('--featnorm', type=int, default=1)
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--numFrames', type=int, default=16)
    parser.add_argument('--valEnsemble', type=int, default=5)
    parser.add_argument('--numContrast', type=int, default=400)

    parser.add_argument('--tfm_heads', type=int, default=8)
    parser.add_argument('--tfm_layers', type=int, default=2)

    parser.add_argument('--test', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--verbose', type=str, default=None)
    parser.add_argument('-j', '--workers', default=8, type=int)

    parser.add_argument('--decay_steps', type=int, default=2e5) 
    parser.add_argument('--decay_rate', type=float, default=0.5)
    parser.add_argument('--warmup_steps', type=int, default=1e3)
    parser.add_argument('--num_iterations', type=int, default=1e5)  
    parser.add_argument('--save_iterations', type=int, default=1e3)
    parser.add_argument('--dir_postfix', type=str, default='')

    parser.add_argument('--backbone', type=str, default='ViT-B/16', choices=['ViT-B/16'])
    parser.add_argument('--dataset', type=str, default='HMDB51-feature-30fps-center', 
                         choices=['HMDB51-feature-30fps-center','Debug',])

    args = parser.parse_args()
    main(args)
