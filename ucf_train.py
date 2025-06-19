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
from ucf_test import test
from dataset import UCFDataset_Whole_test_train
from utils.tools import  get_batch_label
import ucf_option as ucf_option

def calculate_average(numbers):
    if not numbers:  
        return 0
    filtered_array = [x for x in numbers if not np.isnan(x)]

    return sum(filtered_array) / len(filtered_array)


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


def train(model, normal_loader, anomaly_loader, testloader, args, device,project_name):
    model.to(device)
    metrics2 = {}
    gt = np.load(args.gt_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    ap_best = 0
    auc_best = 0
    epoch = 0

    if args.use_checkpoint == True:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        ap_best = checkpoint['ap']
        print("checkpoint info:")
        print("epoch:", epoch + 1, " ap:", ap_best)
    step_num = min(len(normal_loader), len(anomaly_loader))
    for e in range(args.max_epoch):
        model.train()
        loss_total1 = 0
        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)
        for i in range(min(len(normal_loader), len(anomaly_loader))):


            normal_features, normal_label, normal_lengths,normal_video_names = next(normal_iter)
            anomaly_features, anomaly_label, anomaly_lengths,anomaly_video_names = next(anomaly_iter)
            normal_features = normal_features.to(device)
            anomaly_features = anomaly_features.to(device)
            text_normal_label = get_batch_label(normal_label,'UCF').to(device)
            text_anomaly_label = get_batch_label(anomaly_label,'UCF').to(device)


            normal_simlist = []
            normal_idlist=[]
            normal_logits=[]
            # hlj  Split input video into small clips
            normal_visual_feat_spilts = np.split(normal_features, normal_features.shape[1] / args.attn_window, axis=1)

            for n,normaL_visual_feat_spilt in enumerate(normal_visual_feat_spilts):
                cos_sim_percentage = local_similarity(normaL_visual_feat_spilt[:,:int(args.attn_window/2), :].to(device), normaL_visual_feat_spilt[:,int(args.attn_window/2):, :].to(device), 64)
                normal_simlist.append(cos_sim_percentage)
                if n > 3:
                    njitter = calculate_jitter(normal_simlist[-3:])
                    if njitter:
                        # hlj If the similarity meets the requirements then input the model for training
                        if abs(cos_sim_percentage-normal_simlist[-2]) > njitter:
                            normal_logits1 = model(normaL_visual_feat_spilt, 'Train',  args.attn_window,
                                                   normal_video_names,None)
                            normal_logits.append(normal_logits1)
                            normal_idlist.append(n)                        
                else:
                    normal_logits1 = model(normaL_visual_feat_spilt, 'Train',  args.attn_window,
                                           normal_video_names,None)
                    normal_logits.append(normal_logits1)
                    normal_idlist.append(n)

            #hlj retore list
            normal_logits=restore_logits(normal_logits,normal_idlist,normal_lengths)
            normal_logits = torch.cat(normal_logits, dim=1)


            normal_loss1 = CLAS(normal_logits, text_normal_label, normal_lengths.to(device), device)
            loss_total1 += normal_loss1.item()
            normal_loss = normal_loss1
            optimizer.zero_grad()
            normal_loss.backward()
            optimizer.step()

            anomaly_simlist = []
            anomaly_idlist = []
            anomaly_logits=[]
            anomaly_visual_feat_spilts = np.split(anomaly_features, anomaly_features.shape[1] / args.attn_window, axis=1)
            for a, anomaly_visual_feat_spilt in enumerate(anomaly_visual_feat_spilts):

                # get local_similarity
                cos_sim_percentage = local_similarity(anomaly_visual_feat_spilt[:, :int(args.attn_window / 2), :].to(device),
                                                      anomaly_visual_feat_spilt[:, int(args.attn_window / 2):, :].to(device), 64)
                anomaly_simlist.append(cos_sim_percentage)
                if a > 3:
                    ajitter = calculate_jitter(anomaly_simlist[-3:])
                    if ajitter:
                        if abs(cos_sim_percentage-anomaly_simlist[-2]) > ajitter:
                            anomaly_logits1 = model(anomaly_visual_feat_spilt, 'Train', args.attn_window,
                                                   anomaly_video_names,None)
                            anomaly_logits.append(anomaly_logits1)
                            anomaly_idlist.append(a)                        
                else:
                    anomaly_logits1 = model(anomaly_visual_feat_spilt, 'Train', args.attn_window,
                                           anomaly_video_names,None)
                    anomaly_logits.append(anomaly_logits1)
                    anomaly_idlist.append(a)


            # hlj restore list
            anomaly_logits = restore_logits(anomaly_logits, anomaly_idlist, anomaly_lengths)
            anomaly_logits = torch.cat(anomaly_logits, dim=1)
            anomaly_loss1 = CLAS(anomaly_logits, text_anomaly_label, anomaly_lengths.to(device), device)
            loss_total1 += anomaly_loss1.item()
            anomaly_loss = anomaly_loss1
            optimizer.zero_grad()
            anomaly_loss.backward()
            optimizer.step()
            step = i * 2+2
  
            if step % 1280 == 0 and step != 0:
                print('epoch: ', e + 1, '| step: ', step, '| loss1: ', loss_total1 / (i + 1))
                AUC, AP = test(model, testloader,  gt, device, args)
                metrics2['AUC'] = AUC
                metrics2['AP'] = AP
                wandb.log(metrics2, step=i*2 + step_num * e)
                if AUC > auc_best:
                    auc_best = AUC
                    checkpoint = {
                        'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'ap': ap_best}
                    torch.save(checkpoint, '/root/LSHMA/model/'+project_name+'.pth')

        scheduler.step()
        AUC, AP = test(model, testloader,  gt, device, args)
        metrics2['AUC'] = AUC
        metrics2['AP'] = AP
        wandb.log(metrics2, step=step_num + step_num * e)
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
    args = ucf_option.parser.parse_args()
    setup_seed(args.seed)
    project_name='3ucf_ls_cache_'+str(args.cache_keep_max_len)+'_layer_'+str(args.visual_layers)+'_window_'+str(args.attn_window)
    wandb.init(
        project="LSHMA",
        name=project_name,

        settings=wandb.Settings(code_dir=os.path.dirname(os.path.abspath(__file__))),
        save_code=True,
    )

    normal_dataset = UCFDataset_Whole_test_train(args.train_list, False,  True,args.attn_window)
    normal_loader = DataLoader(normal_dataset, batch_size=1, shuffle=True, drop_last=True)
    anomaly_dataset = UCFDataset_Whole_test_train(args.train_list, False,  False,args.attn_window)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=1, shuffle=True, drop_last=True)

    test_dataset = UCFDataset_Whole_test_train(args.test_list, True, False,args.attn_window)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = LSHMA(args.visual_width, args.p,args.visual_layers, args.attn_window, device,args.drop_rate,args.drop_attn_rate,args.drop_qkv_rate,args.cache_keep_max_len)

    train(model, normal_loader, anomaly_loader, test_loader, args,  device,project_name)