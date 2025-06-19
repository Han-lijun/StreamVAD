import torch
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances




def get_batch_label(texts,dataset):
    label_vectors = torch.zeros(0)
    if dataset=='XD':
        classdict = dict(
        {'A': 'normal', 'B1': 'fighting', 'B2': 'shooting', 'B4': 'riot', 'B5': 'abuse', 'B6': 'car accident',
         'G': 'explosion'})
        clslist = []
        for v in classdict.values():
            clslist.append(v)
        for text in texts:
            label_vector = torch.zeros(len(clslist))
            labels = text.split('-')
            for label in labels:
                if label in classdict:
                    label_text = classdict[label]
                    label_vector[clslist.index(label_text)] = 1
            
            label_vector = label_vector.unsqueeze(0)
            label_vectors = torch.cat([label_vectors, label_vector], dim=0)

    else:
        clslist = []
        classdict = dict({'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson', 'Assault': 'assault',
                      'Burglary': 'burglary', 'Explosion': 'explosion', 'Fighting': 'fighting',
                      'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery', 'Shooting': 'shooting',
                      'Shoplifting': 'shoplifting', 'Stealing': 'stealing', 'Vandalism': 'vandalism'})

        for v in classdict.values():
            clslist.append(v)
        for text in texts:
            label_vector = torch.zeros(len(clslist))
            if text in classdict:
                label_text = classdict[text]
                label_vector[clslist.index(label_text)] = 1

            label_vector = label_vector.unsqueeze(0)
            label_vectors = torch.cat([label_vectors, label_vector], dim=0)
    return label_vectors



def random_extract(feat, t_max):
   r = np.random.randint(feat.shape[0] - t_max)
   return feat[r : r+t_max, :]

def uniform_extract(feat, t_max, avg: bool = True):
    new_feat = np.zeros((t_max, feat.shape[1])).astype(np.float32)
    # 均匀采样257个数
    r = np.linspace(0, len(feat), t_max+1, dtype=np.int32)
    if avg == True:
        for i in range(t_max):            
            if r[i]!=r[i+1]:
                new_feat[i,:] = np.mean(feat[r[i]:r[i+1],:], 0)
            else:
                new_feat[i,:] = feat[r[i],:]
    else:
        r = np.linspace(0, feat.shape[0]-1, t_max, dtype=np.uint16)
        new_feat = feat[r, :]
            
    return new_feat

def pad(feat, min_len):
    clip_length = feat.shape[0]
    if clip_length <= min_len:
       res= np.pad(feat, ((0, min_len - clip_length), (0, 0)), mode='constant', constant_values=0)
       return res
    else:
       return feat

def process_feat(feat, length, is_random=False):
    clip_length = feat.shape[0]
    if feat.shape[0] > length:
        if is_random:
            return random_extract(feat, length), length
        else:
            return uniform_extract(feat, length), length
    else:
        return pad(feat, length), clip_length

def process_split(feat, length):
    clip_length = feat.shape[0]
    if clip_length < length:
        return pad(feat, length), clip_length
    else:
        split_num = int(clip_length / length) + 1
        for i in range(split_num):
            if i == 0:
                split_feat = feat[i*length:i*length+length, :].reshape(1, length, feat.shape[1])
            elif i < split_num - 1:
                split_feat = np.concatenate([split_feat, feat[i*length:i*length+length, :].reshape(1, length, feat.shape[1])], axis=0)
            else:
                split_feat = np.concatenate([split_feat, pad(feat[i*length:i*length+length, :], length).reshape(1, length, feat.shape[1])], axis=0)

        return split_feat, clip_length

def complete_video(feat, length):
    clip_length = feat.shape[0]
    if clip_length < length:
        return pad(feat, length), clip_length
    else:
        split_num = int(clip_length / length) + 1
        for i in range(split_num):
            if i == 0:
                split_feat = feat[i*length:i*length+length, :].reshape(1, length, feat.shape[1])
            elif i < split_num - 1:
                split_feat = np.concatenate([split_feat, feat[i*length:i*length+length, :].reshape(1, length, feat.shape[1])], axis=0)
            else:
                split_feat = np.concatenate([split_feat, pad(feat[i*length:i*length+length, :], length).reshape(1, length, feat.shape[1])], axis=0)

        return split_feat, clip_length



def fixed_smooth(logits, t_size):
    lenth=len(logits)
    ins_preds = torch.zeros(0).to(logits.device)
    assert t_size > 1
    if len(logits) % t_size != 0:
        delta = t_size - len(logits) % t_size
        logits = F.pad(logits, (0,  delta), 'constant', 0)

    seq_len = len(logits) // t_size
    for i in range(seq_len):
        seq = logits[i * t_size: (i + 1) * t_size]
        avg = torch.mean(seq, dim=0)
        avg = avg.repeat(t_size)
        ins_preds = torch.cat((ins_preds, avg))

    return ins_preds[:lenth]


def slide_smooth(logits, t_size, mode='zero'):
    assert t_size > 1
    ins_preds = torch.zeros(0).to(logits.device)
    padding = t_size - 1
    if mode == 'zero':
        logits = F.pad(logits, (0, padding), 'constant', 0)
    elif mode == 'constant':
        logits = F.pad(logits, (0, padding), 'constant', logits[-1])

    seq_len = int(len(logits) - t_size) + 1
    for i in range(seq_len):
        seq = logits[i: i + t_size]
        avg = torch.mean(seq, dim=0).unsqueeze(dim=0)
        ins_preds = torch.cat((ins_preds, avg))

    return ins_preds

def slide_max(logits, t_size, mode='zero'):
    assert t_size > 1
    ins_preds = torch.zeros(0).to(logits.device)
    padding = t_size - 1
    if mode == 'zero':
        logits = F.pad(logits, (0, padding), 'constant', 0)
    elif mode == 'constant':
        logits = F.pad(logits, (0, padding), 'constant', logits[-1])

    seq_len = int(len(logits) - t_size) + 1
    for i in range(seq_len):
        seq = logits[i: i + t_size]
        values, _ = torch.max(seq, dim=0)

        # 对最大值使用 unsqueeze 方法
        avg = values.unsqueeze(dim=0)
        # avg = torch.max(seq, dim=0).unsqueeze(dim=0)
        ins_preds = torch.cat((ins_preds, avg))

    return ins_preds