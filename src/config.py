import os
import json
import torch
import itertools
import glob as gb
import numpy as np
from data import *
import utils as ut
from datetime import datetime


def setup_path(args):
    prefix = args.prefix
    postfix = args.postfix
    openset = args.openset
    temporal = args.temporal
    tfmlayers=args.tfm_layers
    batchsize = args.batchsize
    numFrames = args.numFrames
    iters = args.num_iterations
    verbose = args.verbose if args.verbose else 'none'
    dataset = args.dataset.split('-')[0]

    # make all the essential folders, e.g. models, logs, results, etc.
    global dt_string, logPath, modelPath, resultsPath
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M")

    if args.test or args.resume:
        if args.test:
            basename = args.test.split('/')[-2]
        elif args.resume:
            basename = args.resume.split('/')[-2]
        logPath = os.path.join('../logs/', basename)
        modelPath = os.path.join('../models/', basename)
        try:
            with open('{}/running_command.txt'.format(modelPath), 'a') as f:
                json.dump({'command_time_stamp':dt_string, **args.__dict__}, f, indent=2)
        except:
            print({'command_time_stamp':dt_string, **args.__dict__})

    else:
        os.makedirs(f'../logs{"_"+args.dir_postfix if args.dir_postfix != "" else ""}/', exist_ok=True)
        os.makedirs(f'../models{"_"+args.dir_postfix if args.dir_postfix != "" else ""}/', exist_ok=True)

        logPath = os.path.join(f'../logs{"_"+args.dir_postfix if args.dir_postfix != "" else ""}/', 
                                           f'{dt_string}-dataset_{dataset}-openset_{openset}-iter_{iters:.0e}-'
                                           f'bs_{batchsize}-numFrames_{numFrames}-temporal_{temporal}-tfmL_{tfmlayers}-'
                                           f'prompt_{prefix}+X+{postfix}-{verbose}')

        modelPath = os.path.join(f'../models{"_"+args.dir_postfix if args.dir_postfix != "" else ""}/', 
                                               f'{dt_string}-dataset_{dataset}-openset_{openset}-iter_{iters:.0e}-'
                                               f'bs_{batchsize}-numFrames_{numFrames}-temporal_{temporal}-tfmL_{tfmlayers}-'
                                               f'prompt_{prefix}+X+{postfix}-{verbose}')

        os.makedirs(logPath, exist_ok=True)
        os.makedirs(modelPath, exist_ok=True)

        # save all the experiment settings.
        with open('{}/running_command.txt'.format(modelPath), 'w') as f:
            json.dump({'command_time_stamp':dt_string, **args.__dict__}, f, indent=2)

    return [logPath, modelPath]



def setup_dataloader(args):
    # load from extracted visual feature
    if args.dataset == 'HMDB51-feature-30fps-center':
        feature_root = '../feat/HMDB'
    # More datasets to be continued
    else:
        raise ValueError ('Unknown dataset.')

    if args.dataset.startswith('HMDB'):
        trainactions, valactions = [], []
        trn_dataset = readFeatureHMDB51(root=feature_root, frames=args.numFrames, fpsR=[1, 1/2, 1/3, 1/3, 1/3, 1/4], ensemble=1, mode='train')
        val_dataset = readFeatureHMDB51(root=feature_root, frames=args.numFrames, fpsR=[1, 1/2, 1/3, 1/3, 1/3, 1/4], ensemble=args.valEnsemble, mode='val')
    # More datasets to be continued

    return [trn_dataset, val_dataset, trainactions, valactions]


    