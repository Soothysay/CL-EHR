import time

from load_data import *
import class_network1 as lib
from class_network1 import *
import argparse
import os
from tqdm import tqdm, trange#, tqdm_notebook, tnrange
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
import math
import re
from random import *
import numpy as np
import sys
from datetime import timedelta
from datetime import datetime
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
def zero_loss():
    return [0]*10

def get_attn_pad_mask(seq_q):
    batch_size, len_q = seq_q.size()
    #batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_q.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_q)  # batch_size x len_q x len_k
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(64) # scores : [batch_size x 12 x len_q(=len_k) x len_k(=len_q)]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn 
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(50, 64 * 12)
        self.W_K = nn.Linear(50, 64 * 12)
        self.W_V = nn.Linear(50, 64 * 12)
        self.fc = nn.Linear(12 * 64, 50)
        self.norm = nn.LayerNorm(50)
    def forward(self, Q, K, V, attn_mask):
        """if schema=='joint':
            self.W_Q=self.W_Q.cuda()
            self.W_V=self.W_V.cuda()
            self.W_K=self.W_K.cuda()
            self.fc=self.fc.cuda()
            self.norm=self.norm.cuda()"""
        # q: [batch_size x len_q x 50], k: [batch_size x len_k x 50], v: [batch_size x len_k x 50]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, 12, 64).transpose(1,2)  # q_s: [batch_size x 12 x len_q x 64]
        k_s = self.W_K(K).view(batch_size, -1, 12, 64).transpose(1,2)  # k_s: [batch_size x 12 x len_k x 64]
        v_s = self.W_V(V).view(batch_size, -1, 12, 64).transpose(1,2)  # v_s: [batch_size x 12 x len_k x 64]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, 12, 1, 1) # attn_mask : [batch_size x 12 x len_q x len_k]
        # context: [batch_size x 12 x len_q x 64], attn: [batch_size x 12 x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, 12 * 64) # context: [batch_size x len_q x 12 * 64]
        output = self.fc(context)
        output = self.norm(output + residual) # output: [batch_size x len_q x 50]
        return output, attn # output: [batch_size x len_q x 50]
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(50, 200)
        self.fc2 = nn.Linear(200, 50)

    def forward(self, x):
        # (batch_size, len_seq, 50) -> (batch_size, len_seq, 200) -> (batch_size, len_seq, 50)
        return self.fc2(gelu(self.fc1(x)))
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        """if schema=='joint':
            self.enc_self_attn=self.enc_self_attn.cuda()
            self.pos_ffn=self.pos_ffn.cuda()"""
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x 50]
        return enc_outputs, attn
class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        #self.embedding = Embedding()
        self.layer1 = EncoderLayer()
        self.layer2 = EncoderLayer()
        self.layer3 = EncoderLayer()
        self.layer4 = EncoderLayer()
        self.layer5 = EncoderLayer()
        self.layer6 = EncoderLayer()
        #self.fc = nn.Linear(50, 50)
        #self.activ1 = nn.Tanh()
        self.linear = nn.Linear(50, 50)
        self.activ2 = gelu
        self.norm = nn.LayerNorm(50)
        #self.classifier = nn.Linear(50, 2)
        # decoder is shared with embedding layer
        #embed_weight = self.embedding.tok_embed.weight
        #n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(50,191929, bias=False)
        #self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(191929))
        self.l= nn.Linear(50,128).cuda()
        self.activ=nn.Tanh()

    def forward(self, output, masked_pos,input_ids,schema):
        """if schema=='joint':
            self.layer1=self.layer1.cuda()
            self.layer1.enc_self_attn=self.layer1.enc_self_attn.cuda()
            self.layer1.pos_ffn=self.layer1.pos_ffn.cuda()
            self.layer1.enc_self_attn.W_Q=self.layer1.enc_self_attn.W_Q.cuda()
            self.layer1.enc_self_attn.W_K=self.layer1.enc_self_attn.W_K.cuda()
            self.layer1.enc_self_attn.W_V=self.layer1.enc_self_attn.W_V.cuda()
            self.layer1.enc_self_attn.fc=self.layer1.enc_self_attn.fc.cuda()
            self.layer1.enc_self_attn.norm=self.layer1.enc_self_attn.norm.cuda()
            self.layer1.pos_ffn.fc1=self.layer1.pos_ffn.fc1.cuda()
            self.layer1.pos_ffn.fc2=self.layer1.pos_ffn.fc2.cuda() 
            self.layer2=self.layer2.cuda()
            self.layer2.enc_self_attn=self.layer2.enc_self_attn.cuda()
            self.layer2.pos_ffn=self.layer2.pos_ffn.cuda()
            self.layer2.enc_self_attn.W_Q=self.layer2.enc_self_attn.W_Q.cuda()
            self.layer2.enc_self_attn.W_K=self.layer2.enc_self_attn.W_K.cuda()
            self.layer2.enc_self_attn.W_V=self.layer2.enc_self_attn.W_V.cuda()
            self.layer2.enc_self_attn.fc=self.layer2.enc_self_attn.fc.cuda()
            self.layer2.enc_self_attn.norm=self.layer2.enc_self_attn.norm.cuda()
            self.layer2.pos_ffn.fc1=self.layer2.pos_ffn.fc1.cuda()
            self.layer2.pos_ffn.fc2=self.layer2.pos_ffn.fc2.cuda() 
            self.layer3=self.layer3.cuda()
            self.layer3.enc_self_attn=self.layer3.enc_self_attn.cuda()
            self.layer3.pos_ffn=self.layer3.pos_ffn.cuda()
            self.layer3.enc_self_attn.W_Q=self.layer3.enc_self_attn.W_Q.cuda()
            self.layer3.enc_self_attn.W_K=self.layer3.enc_self_attn.W_K.cuda()
            self.layer3.enc_self_attn.W_V=self.layer3.enc_self_attn.W_V.cuda()
            self.layer3.enc_self_attn.fc=self.layer3.enc_self_attn.fc.cuda()
            self.layer3.enc_self_attn.norm=self.layer3.enc_self_attn.norm.cuda()
            self.layer3.pos_ffn.fc1=self.layer3.pos_ffn.fc1.cuda()
            self.layer3.pos_ffn.fc2=self.layer3.pos_ffn.fc2.cuda() 
            self.layer4=self.layer4.cuda()
            self.layer4.enc_self_attn=self.layer4.enc_self_attn.cuda()
            self.layer4.pos_ffn=self.layer4.pos_ffn.cuda()
            self.layer4.enc_self_attn.W_Q=self.layer4.enc_self_attn.W_Q.cuda()
            self.layer4.enc_self_attn.W_K=self.layer4.enc_self_attn.W_K.cuda()
            self.layer4.enc_self_attn.W_V=self.layer4.enc_self_attn.W_V.cuda()
            self.layer4.enc_self_attn.fc=self.layer4.enc_self_attn.fc.cuda()
            self.layer4.enc_self_attn.norm=self.layer4.enc_self_attn.norm.cuda()
            self.layer4.pos_ffn.fc1=self.layer4.pos_ffn.fc1.cuda()
            self.layer4.pos_ffn.fc2=self.layer4.pos_ffn.fc2.cuda() 
            self.layer5=self.layer5.cuda()
            self.layer5.enc_self_attn=self.layer5.enc_self_attn.cuda()
            self.layer5.pos_ffn=self.layer5.pos_ffn.cuda()
            self.layer5.enc_self_attn.W_Q=self.layer5.enc_self_attn.W_Q.cuda()
            self.layer5.enc_self_attn.W_K=self.layer5.enc_self_attn.W_K.cuda()
            self.layer5.enc_self_attn.W_V=self.layer5.enc_self_attn.W_V.cuda()
            self.layer5.enc_self_attn.fc=self.layer5.enc_self_attn.fc.cuda()
            self.layer5.enc_self_attn.norm=self.layer5.enc_self_attn.norm.cuda()
            self.layer5.pos_ffn.fc1=self.layer5.pos_ffn.fc1.cuda()
            self.layer5.pos_ffn.fc2=self.layer5.pos_ffn.fc2.cuda() 
            self.layer6=self.layer6.cuda()
            self.layer6.enc_self_attn=self.layer6.enc_self_attn.cuda()
            self.layer6.pos_ffn=self.layer6.pos_ffn.cuda()
            self.layer6.enc_self_attn.W_Q=self.layer6.enc_self_attn.W_Q.cuda()
            self.layer6.enc_self_attn.W_K=self.layer6.enc_self_attn.W_K.cuda()
            self.layer6.enc_self_attn.W_V=self.layer6.enc_self_attn.W_V.cuda()
            self.layer6.enc_self_attn.fc=self.layer6.enc_self_attn.fc.cuda()
            self.layer6.enc_self_attn.norm=self.layer6.enc_self_attn.norm.cuda()
            self.layer6.pos_ffn.fc1=self.layer6.pos_ffn.fc1.cuda()
            self.layer6.pos_ffn.fc2=self.layer6.pos_ffn.fc2.cuda() 
            self.linear=self.linear.cuda()
            self.norm=self.norm.cuda()
            self.decoder=self.decoder.cuda()
            self.decoder_bias=self.decoder_bias.cuda()
            output=output.cuda()
            segment_ids=segment_ids.cuda()
            masked_pos=masked_pos.cuda()
        """
        #output = ws
        enc_self_attn_mask = get_attn_pad_mask(input_ids)
        if schema=='joint':
            enc_self_attn_mask=enc_self_attn_mask.cuda()
        output, enc_self_attn = self.layer1(output, enc_self_attn_mask)
        output, enc_self_attn = self.layer2(output, enc_self_attn_mask)
        output, enc_self_attn = self.layer3(output, enc_self_attn_mask)
        output, enc_self_attn = self.layer4(output, enc_self_attn_mask)
        output, enc_self_attn = self.layer5(output, enc_self_attn_mask)
        output, enc_self_attn = self.layer6(output, enc_self_attn_mask)
        
        
        # output : [batch_size, len, 50], attn : [batch_size, 12, d_mode, 50]
        # it will be decided by first token(CLS)
        #h_pooled = self.activ1(self.fc(output[:, 0])) # [batch_size, 50]
        #logits_clsf = self.classifier(h_pooled) # [batch_size, 2]
        masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1)) # [batch_size, max_pred, 50]
        # get masked position from final output of transformer.
        h_masked = torch.gather(output, 1, masked_pos) # masking position [batch_size, max_pred, 50]
        h_masked = self.norm(self.activ2(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias # [batch_size, max_pred, n_vocab]
        if schema=='joint':
            op=self.l(output[:, 0])
            op=self.activ(op)
        else:
            op=0
        #else:
        #op=logits_lm
        return logits_lm, op

class Train_reg:
    def __init__(self,model,data,optimizer,gpu_id):
        self.gpu_id=gpu_id
        self.model=model.to(gpu_id)
        self.train_data=data
        self.optimizer=optimizer
        self.model.layer1=model.layer1.to(gpu_id)
        self.model.layer1.enc_self_attn=model.layer1.enc_self_attn.to(gpu_id)
        self.model.layer1.pos_ffn=model.layer1.pos_ffn.to(gpu_id)
        self.model.layer1.enc_self_attn.W_Q=model.layer1.enc_self_attn.W_Q.to(gpu_id)
        self.model.layer1.enc_self_attn.W_K=model.layer1.enc_self_attn.W_K.to(gpu_id)
        self.model.layer1.enc_self_attn.W_V=model.layer1.enc_self_attn.W_V.to(gpu_id)
        self.model.layer1.enc_self_attn.fc=model.layer1.enc_self_attn.fc.to(gpu_id)
        self.model.layer1.enc_self_attn.norm=model.layer1.enc_self_attn.norm.to(gpu_id)
        self.model.layer1.pos_ffn.fc1=model.layer1.pos_ffn.fc1.to(gpu_id)
        self.model.layer1.pos_ffn.fc2=model.layer1.pos_ffn.fc2.to(gpu_id) 
        self.model.layer2=model.layer2.to(gpu_id)
        self.model.layer2.enc_self_attn=model.layer2.enc_self_attn.to(gpu_id)
        self.model.layer2.pos_ffn=model.layer2.pos_ffn.to(gpu_id)
        self.model.layer2.enc_self_attn.W_Q=model.layer2.enc_self_attn.W_Q.to(gpu_id)
        self.model.layer2.enc_self_attn.W_K=model.layer2.enc_self_attn.W_K.to(gpu_id)
        self.model.layer2.enc_self_attn.W_V=model.layer2.enc_self_attn.W_V.to(gpu_id)
        self.model.layer2.enc_self_attn.fc=model.layer2.enc_self_attn.fc.to(gpu_id)
        self.model.layer2.enc_self_attn.norm=model.layer2.enc_self_attn.norm.to(gpu_id)
        self.model.layer2.pos_ffn.fc1=model.layer2.pos_ffn.fc1.to(gpu_id)
        self.model.layer2.pos_ffn.fc2=model.layer2.pos_ffn.fc2.to(gpu_id) 
        self.model.layer3=model.layer3.to(gpu_id)
        self.model.layer3.enc_self_attn=model.layer3.enc_self_attn.to(gpu_id)
        self.model.layer3.pos_ffn=model.layer3.pos_ffn.to(gpu_id)
        self.model.layer3.enc_self_attn.W_Q=model.layer3.enc_self_attn.W_Q.to(gpu_id)
        self.model.layer3.enc_self_attn.W_K=model.layer3.enc_self_attn.W_K.to(gpu_id)
        self.model.layer3.enc_self_attn.W_V=model.layer3.enc_self_attn.W_V.to(gpu_id)
        self.model.layer3.enc_self_attn.fc=model.layer3.enc_self_attn.fc.to(gpu_id)
        self.model.layer3.enc_self_attn.norm=model.layer3.enc_self_attn.norm.to(gpu_id)
        self.model.layer3.pos_ffn.fc1=model.layer3.pos_ffn.fc1.to(gpu_id)
        self.model.layer3.pos_ffn.fc2=model.layer3.pos_ffn.fc2.to(gpu_id) 
        self.model.layer4=model.layer4.to(gpu_id)
        self.model.layer4.enc_self_attn=model.layer4.enc_self_attn.to(gpu_id)
        self.model.layer4.pos_ffn=model.layer4.pos_ffn.to(gpu_id)
        self.model.layer4.enc_self_attn.W_Q=model.layer4.enc_self_attn.W_Q.to(gpu_id)
        self.model.layer4.enc_self_attn.W_K=model.layer4.enc_self_attn.W_K.to(gpu_id)
        self.model.layer4.enc_self_attn.W_V=model.layer4.enc_self_attn.W_V.to(gpu_id)
        self.model.layer4.enc_self_attn.fc=model.layer4.enc_self_attn.fc.to(gpu_id)
        self.model.layer4.enc_self_attn.norm=model.layer4.enc_self_attn.norm.to(gpu_id)
        self.model.layer4.pos_ffn.fc1=model.layer4.pos_ffn.fc1.to(gpu_id)
        self.model.layer4.pos_ffn.fc2=model.layer4.pos_ffn.fc2.to(gpu_id) 
        self.model.layer5=model.layer5.to(gpu_id)
        self.model.layer5.enc_self_attn=model.layer5.enc_self_attn.to(gpu_id)
        self.model.layer5.pos_ffn=model.layer5.pos_ffn.to(gpu_id)
        self.model.layer5.enc_self_attn.W_Q=model.layer5.enc_self_attn.W_Q.to(gpu_id)
        self.model.layer5.enc_self_attn.W_K=model.layer5.enc_self_attn.W_K.to(gpu_id)
        self.model.layer5.enc_self_attn.W_V=model.layer5.enc_self_attn.W_V.to(gpu_id)
        self.model.layer5.enc_self_attn.fc=model.layer5.enc_self_attn.fc.to(gpu_id)
        self.model.layer5.enc_self_attn.norm=model.layer5.enc_self_attn.norm.to(gpu_id)
        self.model.layer5.pos_ffn.fc1=model.layer5.pos_ffn.fc1.to(gpu_id)
        self.model.layer5.pos_ffn.fc2=model.layer5.pos_ffn.fc2.to(gpu_id) 
        self.model.layer6=model.layer6.to(gpu_id)
        self.model.layer6.enc_self_attn=model.layer6.enc_self_attn.to(gpu_id)
        self.model.layer6.pos_ffn=model.layer6.pos_ffn.to(gpu_id)
        self.model.layer6.enc_self_attn.W_Q=model.layer6.enc_self_attn.W_Q.to(gpu_id)
        self.model.layer6.enc_self_attn.W_K=model.layer6.enc_self_attn.W_K.to(gpu_id)
        self.model.layer6.enc_self_attn.W_V=model.layer6.enc_self_attn.W_V.to(gpu_id)
        self.model.layer6.enc_self_attn.fc=model.layer6.enc_self_attn.fc.to(gpu_id)
        self.model.layer6.enc_self_attn.norm=model.layer6.enc_self_attn.norm.to(gpu_id)
        self.model.layer6.pos_ffn.fc1=model.layer6.pos_ffn.fc1.to(gpu_id)
        self.model.layer6.pos_ffn.fc2=model.layer6.pos_ffn.fc2.to(gpu_id) 
        self.model.linear=model.linear.to(gpu_id)
        self.model.norm=model.norm.to(gpu_id)
        self.model=DDP(model,device_ids=[gpu_id],find_unused_parameters=True)
    def _run_batch(self,embed_w,masked_pos,input_ids,masked_tokens):
        criterion = nn.CrossEntropyLoss()
        self.optimizer.zero_grad()
        logits_lm, _ = self.model(embed_w, masked_pos,input_ids,'pretrain')
        loss = criterion(logits_lm.transpose(1, 2), masked_tokens)
        loss.backward()
        #print(loss.item())
        self.optimizer.step()
    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = 'models/bignotes/multiGPU/Model'+str(epoch)+'.pt'
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")
    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for embed_w,masked_pos,input_ids,masked_tokens in self.train_data:
            embed_w = embed_w.to(self.gpu_id)
            masked_pos = masked_pos.to(self.gpu_id)
            input_ids = input_ids.to(self.gpu_id)
            masked_tokens = masked_tokens.to(self.gpu_id)
            self._run_batch(embed_w,masked_pos,input_ids,masked_tokens)
    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 1 and epoch % 2 == 0:
                self._save_checkpoint(epoch)

def ddp_setup(rank: int, world_size: int):
  """
  Args:
      rank: Unique identifier of each process
     world_size: Total number of processes
  """
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = "12355"
  init_process_group(backend="nccl", rank=rank, world_size=world_size)

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

def load_train_objs(embed_w,masked_pos,input_ids,masked_tokens):
    torch.save(embed_w,'embed_w1.pt')
    torch.save(masked_pos,'masked_pos1.pt')
    torch.save(input_ids,'input_ids1.pt')
    torch.save(masked_tokens,'masked_tokens1.pt')
    train_set = BERTDATA(embed_w,masked_pos,input_ids,masked_tokens)  # load your dataset
    model = BERT()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return train_set, model, optimizer

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

def main(rank: int, world_size: int, embed_w,masked_pos,input_ids,masked_tokens, total_epochs=500, batch_size=150):
    ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs(embed_w,masked_pos,input_ids,masked_tokens)
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Train_reg(model, train_data, optimizer, rank)
    trainer.train(total_epochs)
    destroy_process_group()

class BERTDATA(Dataset):
    def __init__(self,embed_w,masked_pos,input_ids,masked_tokens):
        self.input_ids=input_ids
        self.embed_w=embed_w
        self.masked_pos=masked_pos
        self.masked_tokens=masked_tokens
    def __len__(self):
        return len(self.embed_w)
    def __getitem__(self, idx):
        embw=self.embed_w[idx]
        mpos=self.masked_pos[idx]
        ipi=self.input_ids[idx]
        mtok=self.masked_tokens[idx]
        return embw,mpos,ipi,mtok
#import argparse
#parser = argparse.ArgumentParser(description='simple distributed training job')
#parser.add_argument('total_epochs', type=int, default=200, help='Total epochs to train the model')
#parser.add_argument('save_every', type=int, help='How often to save a snapshot')
#parser.add_argument('--batch_size', default=50, type=int, help='Input batch size on each device (default: 32)')
#args = parser.parse_args(args=[])
if __name__ == '__main__':
    hasmap=pd.read_csv('data/Hashmap1.csv')
    hm1=dict(hasmap.values)
    hm= {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
    number_dict={0: '[PAD]', 1: '[CLS]', 2: '[SEP]', 3: '[MASK]'}
    for key,value in hm1.items():
        hm[key]=(value+4)
        number_dict[(value+4)]=key
    #print(len(hm))
    note1=pd.read_csv('data/data/chunk4_encoded.csv')
    #note2=pd.read_csv('data/chunk2_encoded1.csv')
    #note3=pd.read_csv('data/chunk3_encoded1.csv')
    #note1=pd.concat([note,note2,note3],axis=0)
    print(len(note1))
    note1=note1.dropna(subset=['NOTE_TEXT'])
    #note1=note1[]
    note1.reset_index(drop=True)
    #inter=pd.read_csv('data/interactions.csv')
    #inter=inter[['user_id','timestamp']]
    #str_d='2007/12/01'
    #d = datetime.strptime(str_d, "%Y/%m/%d")
    #note1['timestamp']=note1['NOTE_DATETIME'].apply(lambda x: (x-d).days)
    note1=note1[['pid','timestamp','NOTE_TEXT']]
    #inter['pid']=inter['user_id']
    #inter=inter[['pid','timestamp']]
    sentences=[]
    for row, data in tqdm(note1.iterrows()):
        
        sentences.append(data['NOTE_TEXT'])
    #print(len(sentences))
    #sentences=sentences[:9000]
    token_list=list()

    ml=0
    for sentence in sentences:
        arr=[int(s) for s in sentence.split()]
        if len(arr)>ml:
            ml=len(arr)
        token_list.append(arr)
    #print(ml)
    #print(token_list[:3])
    maxlen = 325 # maximum of length
    batch_size = len(sentences) # Full Chunk Value For now. Need to think about this
    max_pred = 10  # max tokens of prediction
    n_layers = 6 # number of Encoder of Encoder Layer
    n_heads = 12 # number of heads in Multi-Head Attention
    d_model = 50 # Embedding Size
    d_ff = 50 * 4  # 4*50, FeedForward dimension
    d_k = d_v = 64  # dimension of K(=Q), V
    n_segments = 2
    batcher=6000
    vocab_size =len(hm)
    #print(vocab_size)
    batch=make_batch(sentences,hm,token_list,max_pred,maxlen,vocab_size,number_dict)
    input_ids, _, masked_tokens, masked_pos = map(torch.LongTensor, zip(*batch))
    unpickled_df = np.load("data/data/L10T50G100A1ngU_iter99.p",allow_pickle=True)
    voc1=unpickled_df[1]
    embed_w = torch.rand(((len(sentences)),325,50))
    va=np.random.rand(50)
    for i in tqdm(range(batch_size)):
        for j in range(maxlen):
            pos=int(input_ids[i][j])
            if pos<=3:
                ve=va
            else:
                #if pos+4>=vocab_size:
                    #pos=vocab_size-5
                ve=voc1[(pos-4)]
            embed_w[i,j,:]=torch.tensor(ve)
    input_ids=input_ids
    #segment_ids=segment_ids
    masked_tokens=masked_tokens
    embed_w=embed_w
    masked_pos=masked_pos
    world_size = torch.cuda.device_count()
    #args.total_epochs=200
    mp.spawn(main, args=(world_size,embed_w,masked_pos,input_ids,masked_tokens,500,150,), nprocs=world_size)








