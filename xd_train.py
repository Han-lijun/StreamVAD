import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import random
import os
import wandb
from model import LSHMA
from xd_test import test
from dataset import XDDataset_Whole_test_train as XDDataset
from utils.tools import get_batch_label
import xd_option as xd_option

def local_similarity(A, B, num_blocks=4):
    batch_size, seq_length, feature_dim = A.shape
    block_size = feature_dim // num_blocks
    local_similarities = []

    for i in range(num_blocks):
        start_idx = i * block_size
        end_idx = start_idx + block_size
        A_block = A[:, :, start_idx:end_idx]
        B_block = B[:, :, start_idx:end_idx]
        cos_sim = F.cosine_similarity(A_block, B_block, dim=2)
        cos_sim = (cos_sim + 1) / 2
        avg_sim = cos_sim.mean().item()
        local_similarities.append(avg_sim)
    return min(local_similarities)

def calculate_jitter(simlist):
    first_order_diff = np.diff(simlist)
    jitter = np.std(first_order_diff)
    return jitter

def restore_logits(logits,idlist,normal_lengths):
    # restore logits list
    restored_list = [logits[0]] * normal_lengths
    for ix in range(len(idlist)):
        score = logits[ix]
        start_index = idlist[ix]
        if ix == len(idlist) - 1:
            restored_list[start_index:] = [score] * (normal_lengths - start_index)
        else:
            end_index = idlist[ix + 1]
            restored_list[start_index:end_index] = [score] * (end_index - start_index)
    return restored_list


def CLAS(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])
    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
        tmp = torch.mean(tmp).view(1)
        instance_logits = torch.cat((instance_logits, tmp))
    clsloss = F.binary_cross_entropy(instance_logits, labels)
    return clsloss



def train(model,train_loader, testloader, args, device,project_name):
    model.to(device)

    metrics2 = {}
    gt = np.load(args.gt_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    
    ap_best = 0
    auc_best = 0
    epoch = 0
    step=0
    if args.use_checkpoint == True:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        ap_best = checkpoint['ap']
        print("checkpoint info:")
        print("epoch:", epoch + 1, " ap:", ap_best)
    step_num =len(train_loader)
    for e in range(args.max_epoch):
        model.train()
        loss_total1 = 0
       
        for i, item in enumerate(train_loader):
            visual_feat, text_labels, feat_lengths,video_names = item
            visual_feat = visual_feat.to(device)
            feat_lengths = feat_lengths.to(device)
            text_labels = get_batch_label(text_labels,'XD').to(device)
            visual_feat_spilts=np.split(visual_feat,  visual_feat.shape[1] /args.attn_window, axis=1)
            simlist=[]
            logits=[]
            for n,visual_feat_spilt in enumerate(visual_feat_spilts):
                cos_sim_percentage = local_similarity(visual_feat_spilt[:,:int(args.attn_window/2), :].to(device), visual_feat_spilt[:,int(args.attn_window/2):, :].to(device), 64)
                simlist.append(cos_sim_percentage)
                if n > 3:
                    njitter = calculate_jitter(simlist[-3:])
                    if njitter:
                        if abs(cos_sim_percentage-simlist[-2]) > njitter:
                            logits1 = model(visual_feat_spilt, 'Train',  args.attn_window,
                                                   video_names,None)
                            logits.append(logits1)                                              
                else:
                    logits1 = model(visual_feat_spilt, 'Train', args.attn_window,
                                           video_names,None)
                    logits.append(logits1)
            logits = torch.cat(logits, dim=1)
            # loss1
            loss1 = CLAS(logits, text_labels, feat_lengths, device)
            loss_total1 += loss1.item()
            loss = loss1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            step = 1+step

            if step % 4800 == 0 and step != 0:
                print('epoch: ', e + 1, '| step: ', step, '| loss1: ', loss_total1 / (i + 1))
                AUC, AP = test(model, testloader,   gt,device, args)
                metrics2['AUC'] = AUC
                metrics2['AP'] = AP

                wandb.log(metrics2, step=i + step_num * e)

                if AUC > auc_best:
                    auc_best = AUC
                    checkpoint = {
                        'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'ap': ap_best}
                    torch.save(checkpoint, '/root/LSHMA/model/'+project_name+'.pth')
        scheduler.step()
        print('||epoch: ', e + 1,  '| loss1: ', loss_total1 / (i + 1))
        AUC, AP = test(model, testloader,  gt,  device, args)
        metrics2['AUC'] = AUC
        metrics2['AP'] = AP
        wandb.log(metrics2, step= step_num * (e+1))

        if AUC > auc_best:
            auc_best = AUC
            checkpoint = {
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ap': ap_best}
            torch.save(checkpoint, '/root/LSHMA/model/'+project_name+'.pth')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    args = xd_option.parser.parse_args()
    setup_seed(args.seed)
    project_name='xd_ls_cache_'+str(args.cache_keep_max_len)+'_layer_'+str(args.visual_layers)+'_window_'+str(args.attn_window)
    wandb.init(
        project="LSHMA",
        name=project_name,

        settings=wandb.Settings(code_dir=os.path.dirname(os.path.abspath(__file__))),
        save_code=True,
    )

    train_dataset = XDDataset( args.train_list, False, args.attn_window)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    test_dataset = XDDataset( args.test_list, True, args.attn_window)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = LSHMA( args.visual_width, args.p,args.visual_layers, args.attn_window, device,args.drop_rate,args.drop_attn_rate,args.drop_qkv_rate,args.cache_keep_max_len)

    train(model,train_loader, test_loader, args,  device,project_name)