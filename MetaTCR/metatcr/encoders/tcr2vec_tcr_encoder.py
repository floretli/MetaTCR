import torch
from .tcr2vec.model import TCR2vec
from .tcr2vec.dataset import TCRLabeledDset
from .tcr2vec.utils import get_emb
from torch.utils.data import DataLoader

import numpy as np

## load the trained TCR2vec model
def load_tcr2vec(path_to_TCR2vec = '../..pretrained_models/TCR2vec_120', device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    emb_model = TCR2vec(path_to_TCR2vec)
    emb_model = emb_model.to(device)
    return emb_model

def seqlist2ebd(seq_list, emb_model, emb_size = 120, keep_pbar = True):  ## input: a list of TCR seqs ['CAAAGGIYEQYF', 'CAAAPGINEQFF' ... ], output: the mtx of 96-dim embedding

    if len(seq_list) == 0 :
        return np.zeros((1, emb_size),dtype='float32')

    dset = TCRLabeledDset(seq_list, only_tcr=True) #input a list of TCRs
    loader = DataLoader(dset, batch_size=2048, collate_fn=dset.collate_fn, shuffle=False)
    emb = get_emb(emb_model, loader, detach=True, keep_pbar = keep_pbar) #B x emb_size

    return emb


if __name__ == "__main__":
    ## load the trained TCR2vec model
    path_to_TCR2vec = '../../pretrained_models/TCR2vec_120'
    emb_model = load_tcr2vec(path_to_TCR2vec)

    # convert list of seqs to numpy array
    seq_list = ['NAGVTQTPKFQVLKTGQSMTLQCAQDMNHNSMYWYRQDPYSASEGTTDKGEVPNGYNVSRLNKREFSLRLESAAPSQTSVYFCASSEALGTGNTIYFGEGSWLTVV',
                'NAGVTQTPKFQVLKTGQSMTLQCAQDMNHNSMYWYRQDPGMGLLLIYYSASEGTTDKGEVPNGYNVSRLNKREFSLRLESAAPSQTSVYFCASSEALGTGNTIYFGEGSWLTVV',
                'NAGVTQTPKFQVLKTGQSMTLQCAQDMNHNSMYWYRQDPGMGLRLIYYSRLNKREFSLRLESAAPSQTSVYFCASSEALGTGNTIYFGEGSWLTVV']
    embmtx = seqlist2ebd(seq_list, emb_model)
    print("example seq list = ", seq_list)
    print("embedding mtx shape = ", embmtx.shape)
    print("embedding mtx = ", embmtx)
