import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from dataset import UCFDataset_Whole_test_train
from utils.tools import slide_smooth
import ucf_option as ucf_option
from model import LSHMA

def test(model, testdataloader, gt, device,args):
    model.to(device)
    model.eval()
    threshold_range=[0.6]
    best_threshold = 0
    bsetauc=0
    bestap=0
    for threshold in threshold_range:
        print(f"Evaluating with threshold: {threshold:.2f}")

        with torch.no_grad():
            for i, item in enumerate(testdataloader):
                visual = item[0].squeeze(0)
                length = item[2]
                video_names = item[3]
                length = int(length)
                len_cur = length
                visual = visual.to(device)

                if len(visual.shape) < 3:
                    visual = visual.unsqueeze(0)
                # hlj Split video clips for input into the network
                visual_feat_spilts = np.split(visual, visual.shape[1] / args.attn_window, axis=1)
                logits = []
                for visual_feat_spilt in visual_feat_spilts:
                    logits1 = model(visual_feat_spilt, 'Test',  args.attn_window, video_names,threshold)
                    logits.append(logits1)

                logits = torch.cat(logits, dim=1)
                logits = logits.reshape(logits.shape[0] * logits.shape[1], logits.shape[2])
                prob1 = logits[0:len_cur].squeeze(-1)
                prob1=slide_smooth(prob1,8)
                if i == 0:
                    ap1 = prob1
                else:
                    ap1 = torch.cat([ap1, prob1], dim=0)
 
        ap1 = ap1.cpu().numpy()
        ap1 = ap1.tolist()

        ROC1 = roc_auc_score(gt, np.repeat(ap1, 16))
        AP1 = average_precision_score(gt, np.repeat(ap1, 16))
        print("AUC1: ", ROC1, " AP1: ", AP1)
        if ROC1 > bsetauc:
            bsetauc = ROC1
            bestap=AP1
            best_threshold = threshold

    print(f"Best Threshold: {best_threshold}, Best Performance: {bsetauc}")
    return bsetauc, bestap

if __name__ == '__main__':
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    args = ucf_option.parser.parse_args()
    test_dataset = UCFDataset_Whole_test_train( args.test_list, True, False,args.attn_window)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    gt = np.load(args.gt_path)
    model = LSHMA(args.visual_width, args.p,
                    args.visual_layers, args.attn_window,device,args.drop_rate,args.drop_attn_rate,args.drop_qkv_rate,args.cache_keep_max_len)
    model_path = '/root/LSHMA/model/8744_ucf.pth'
    model_param = torch.load(model_path)
    model.load_state_dict(model_param['model_state_dict'])
    test(model, test_loader,  gt, device, args)

