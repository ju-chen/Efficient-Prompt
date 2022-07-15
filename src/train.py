import os
import val
import clip
import time
import torch
import einops
import random
import numpy as np
from torch import nn
from utils import save_checkpoint


def train_CLIPrompt(args, dataloader, text, model, optimizer, lr_scheduler, logger, device):
    xentropy = nn.CrossEntropyLoss()

    timestart = time.time()
    iteration = args.start_iter
    numContrast = args.numContrast
    featnorm = args.featnorm
    trnloader, valloader = dataloader
    iteration_epoch = len(trnloader)
    actionlist, actiondict, actiontoken, trainactions, valactions = text
    dataset = args.dataset.split('-')[0]

    model.train()
    model.clipmodel.eval()

    while iteration < args.num_iterations:
        for idx, sample in enumerate(trnloader):
            vids, name = sample

            # sample positive and negative
            uniqname = np.unique(name)
            numNeg = numContrast - len(uniqname)
            complement = list(set(actionlist) - set(uniqname))
            inp_actionlist = uniqname.tolist() + random.sample(complement, min(numNeg, len(complement)))
            targets = torch.tensor([inp_actionlist.index(n) for n in name]).to(device)

            vFeature, tFeature = model(vids.to(device), inp_actionlist)
            if featnorm:
                vFeature = vFeature / vFeature.norm(dim=-1, keepdim=True)
                tFeature = tFeature / tFeature.norm(dim=-1, keepdim=True)
                logits = vFeature @ tFeature.t() / 0.07  
            else:
                logits = vFeature @ tFeature.t() 
                
            loss = xentropy(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step(iteration)    

            if iteration % 5 == 0:
                # calculate batchwise top1, top5 acc
                similarity = logits.softmax(dim=-1)
                values, indices = similarity.topk(5)
                top1 = indices[:,0] == targets
                top5 = (indices == einops.repeat(targets, 'b -> b k', k=5)).sum(-1)
                top1ACC = top1.sum() / len(top1)
                top5ACC = top5.sum() / len(top5)
 
                print('dataset: {},'.format(dataset),
                      'epoch[{}][{}/{}]'.format(iteration//iteration_epoch, idx, iteration_epoch),
                      'iter {},'.format(iteration),
                      'time {:.01f}s,'.format(time.time() - timestart),
                      'loss {:.03f},'.format(loss.detach().cpu().numpy()),
                      'top1 {:.03f},'.format(top1ACC.detach().cpu().numpy()),
                      'top5 {:.03f}.'.format(top5ACC.detach().cpu().numpy()))

                logger.add_scalar('train/loss', loss.detach().cpu().numpy(), iteration)
                logger.add_scalar('train/top1', top1ACC.detach().cpu().numpy(), iteration)
                logger.add_scalar('train/top5', top5ACC.detach().cpu().numpy(), iteration)
                logger.add_scalar('train/lr', lr_scheduler.get_last_lr(), iteration)

            iteration += 1
            timestart = time.time()

            if iteration % iteration_epoch == 0 and iteration > 1:
                val.val_CLIPrompt(args, valloader, text, model, logger, device, iteration)
                # switch back to training mode
                model.train()
                model.clipmodel.eval()

            # save model
            if iteration % args.save_iterations == 0:
                print('saving checkpoint')
                state_dict = model.state_dict()
                save_dict = {
                    'state_dict': state_dict,
                    'optimizer': optimizer.state_dict(),
                    'iteration': iteration}
                save_checkpoint(save_dict, is_best=False, gap=1, 
                    filename=os.path.join(args.model_path, 'checkpoint_iter%d.pth.tar' % iteration), 
                    keep_all=True)

