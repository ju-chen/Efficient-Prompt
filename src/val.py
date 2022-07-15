import torch
import einops
import numpy as np
from tqdm import tqdm
import clip


def val_CLIPrompt(args, dataloader, text, model, logger, device, iteration):

    loss, top1, top5 = [], [], []
    valEnsemble = args.valEnsemble
    featnorm = args.featnorm
    actionlist, actiondict, actiontoken, trainactions, valactions = text

    model.eval()
    with torch.no_grad():
        similarity, targets = torch.zeros(0).to(device), torch.zeros(0).to(device)
        for idx, sample in tqdm(enumerate(dataloader), total=len(dataloader)):
            vids, name = sample
            if idx == 0:
                vFeature, tFeature = model(vids.to(device), actionlist)
            else:
                vFeature, _ = model(vids.to(device), actionlist[:1])
            
            target_batch = torch.tensor([actionlist.tolist().index(n) for n in name]).to(device)
            if featnorm:
                vFeature = vFeature / vFeature.norm(dim=-1, keepdim=True)
                tFeature = tFeature / tFeature.norm(dim=-1, keepdim=True)
                logits = vFeature @ tFeature.t() / 0.07  
            else:
                logits = vFeature @ tFeature.t()

            # calculate batchwise top1, top5 acc
            sim_batch = logits.softmax(dim=-1)
            similarity = torch.cat([similarity, sim_batch], dim=0)
            targets = torch.cat([targets, target_batch], dim=0)

        sim_ensemble = torch.zeros(0).to(device)
        test_num = int(len(similarity)/valEnsemble)
        for enb in range(valEnsemble):
            sim_ensemble = torch.cat([sim_ensemble, similarity[enb*test_num:enb*test_num+test_num].unsqueeze(0)], dim=0) 
        target_final = targets[:test_num]

        sim_final = torch.mean(sim_ensemble, 0)
        values, indices = sim_final.topk(5)
        top1 = (indices[:, 0] == target_final).tolist()
        top5 = ((indices == einops.repeat(target_final, 'b -> b k', k=5)).sum(-1)).tolist()

        top1ACC = np.array(top1).sum() / len(top1)
        top5ACC = np.array(top5).sum() / len(top5)

        print('iteration {},'.format(iteration),
              'valtop1 {:.03f},'.format(top1ACC),
              'valtop5 {:.03f}.'.format(top5ACC))

        if logger:
            logger.add_scalar('val/top1', top1ACC, iteration)
            logger.add_scalar('val/top5', top5ACC, iteration)
