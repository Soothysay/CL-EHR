# Main Training Script
# change args for relevant data load and epochs

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
def get_statistics(user_sequence_id, user2id, item2id, static_feature_sequence, dynamic_feature_sequence,
        D2id, M2id, R2id, N2id):
    num_interactions = len(user_sequence_id)
    num_users = len(user2id)
    num_items = len(item2id) + 1 # one extra item for "none-of-these"
    num_D = len(D2id) + 1# one extra item for "none-of-these"
    num_M = len(M2id) + 1# one extra item for "none-of-these"
    num_R = len(R2id) + 1# one extra item for "none-of-these"
    num_N= len(N2id) + 1# one extra item for "none-of-these"
    num_static_features = len(static_feature_sequence[0])
    num_dynamic_features = len(dynamic_feature_sequence[0])
    return num_interactions, num_users, num_items, num_static_features, num_dynamic_features, num_D, num_M, num_R, num_N

def print_network_statistics(num_interactions, num_users, num_items, num_static_features, num_dynamic_features, num_D, num_M, num_R,num_N):
    print("{} users, {} items ({} doctor, {} medication, {} room, {} notes), {} interactions".format(num_users, num_items-1, num_D-1, num_M-1, num_R-1, num_N-1, num_interactions))
    print("{} user static features, {} user dynamic features".format(num_static_features, num_dynamic_features))

def initialize_loss_arrays(args):
    loss_per_timestep = np.zeros((args.epochs))
    prediction_loss_per_timestep = np.zeros((args.epochs))
    user_update_loss_per_timestep = np.zeros((args.epochs))
    item_update_loss_per_timestep = np.zeros((args.epochs))
    D_loss_per_timestep = np.zeros((args.epochs))
    M_loss_per_timestep = np.zeros((args.epochs))
    R_loss_per_timestep = np.zeros((args.epochs))
    llm_loss=np.zeros((args.epochs))
    cl_loss=np.zeros((args.epochs))
    return loss_per_timestep, prediction_loss_per_timestep, user_update_loss_per_timestep, item_update_loss_per_timestep, \
            D_loss_per_timestep, M_loss_per_timestep, R_loss_per_timestep,llm_loss,cl_loss

def initialize_dictionaries_for_tbaching():
    cached_tbatches_user = {}
    cached_tbatches_item = {}
    cached_tbatches_itemtype = {}
    cached_tbatches_interactionids = {}
    cached_tbatches_static_feature = {}
    cached_tbatches_dynamic_feature = {}
    cached_tbatches_user_timediffs = {}
    cached_tbatches_item_timediffs = {}
    cached_tbatches_previous_item = {}
    return cached_tbatches_user, cached_tbatches_item, cached_tbatches_itemtype, cached_tbatches_interactionids,\
            cached_tbatches_static_feature, cached_tbatches_dynamic_feature,\
            cached_tbatches_user_timediffs, cached_tbatches_item_timediffs, cached_tbatches_previous_item

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
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.fc = nn.Linear(n_heads * d_v, d_model)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, Q, K, V, attn_mask):
        """if schema=='joint':
            self.W_Q=self.W_Q.cuda()
            self.W_V=self.W_V.cuda()
            self.W_K=self.W_K.cuda()
            self.fc=self.fc.cuda()
            self.norm=self.norm.cuda()"""
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]
        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size x len_q x n_heads * d_v]
        output = self.fc(context)
        output = self.norm(output + residual) # output: [batch_size x len_q x d_model]
        return output, attn # output: [batch_size x len_q x d_model]
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):

        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch_size, len_seq, d_model) -> (batch_size, len_seq, d_ff) -> (batch_size, len_seq, d_model)
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
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
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
        #self.fc = nn.Linear(d_model, d_model)
        #self.activ1 = nn.Tanh()
        self.linear = nn.Linear(d_model, d_model)
        self.activ2 = gelu
        self.norm = nn.LayerNorm(d_model)
        #self.classifier = nn.Linear(d_model, 2)
        # decoder is shared with embedding layer
        #embed_weight = self.embedding.tok_embed.weight
        #n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(d_model,vocab_size, bias=False)
        #self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(vocab_size))
        self.l= nn.Linear(d_model,128).cuda()
        self.activ=nn.Tanh()

    def forward(self, output, masked_pos,input_ids,schema):

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


        # output : [batch_size, len, d_model], attn : [batch_size, n_heads, d_mode, d_model]
        # it will be decided by first token(CLS)
        #h_pooled = self.activ1(self.fc(output[:, 0])) # [batch_size, d_model]
        #logits_clsf = self.classifier(h_pooled) # [batch_size, 2]
        masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1)) # [batch_size, max_pred, d_model]
        # get masked position from final output of transformer.
        h_masked = torch.gather(output, 1, masked_pos) # masking position [batch_size, max_pred, d_model]
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
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', default='whole3', help='Name of the network/dataset')
    parser.add_argument('--gpu', default=4, type=int, help='ID of the gpu. Default is 0')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs')
    parser.add_argument('--embedding_dim', default=128, type=int, help='dimension of dynamic embeddings')
    parser.add_argument('--patience', default=50, type=int, help='early stopping')
    parser.add_argument('--tbatch_timespan', default=4, type=int,
            help='timespan of the t-batch. Longer timespan requires more GPU memory, but less number of batches are created and model is updated less frequently (training is faster).')
    parser.add_argument('--laplacian', default="laplacian_DMR_big", help='Name of the file that contains laplacians')
    parser.add_argument('--doctor_static', default="doctor_embedding", help='Name of the file that contains doctor static embedding')
    parser.add_argument('--medication_static', default="medication_embedding", help='Name of the file that contains medication static embedding')
    parser.add_argument('--room_static', default="room_embedding", help='Name of the file that contains room static embedding')
    parser.add_argument('--num_user_static_features', default=2, type=int, help='number of static patient features in the dataset')
    parser.add_argument('--decent_pretrain_epochs', default=0, type=int, help='number of DECEnt Pretrain Epochs')
    parser.add_argument('--bert_pretrain_epochs', default=200, type=int, help='number of BERT Pretrain Epochs')
    args = parser.parse_args(args=[])
    #cuda=torch.device('cuda:0')
    args.datapath = "data/data/{}.csv".format(args.network)
    args.laplacian = "data/data/{}.npz".format(args.laplacian)
    args.doctor_static = "data/data/{}.npz".format(args.doctor_static)
    args.medication_static = "data/data/{}.npz".format(args.medication_static)
    args.room_static = "data/data/{}.npz".format(args.room_static)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Load data
    [user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence,
     item2id, item_sequence_id, item_timediffs_sequence, timestamp_sequence,
     static_feature_sequence, dynamic_feature_sequence, y_true,
     item2itemtype, itemtype_sequence,
     D2id, M2id, R2id,N2id] = load_network_with_label(args)

    print(len(user_sequence_id))
    print(len(item_sequence_id))
    num_interactions, num_users, num_items, num_static_features, num_dynamic_features, num_D, num_M, num_R, num_N = get_statistics(user_sequence_id, user2id, item2id, static_feature_sequence, dynamic_feature_sequence,
            D2id, M2id, R2id,N2id)
    print_network_statistics(num_interactions, num_users, num_items, num_static_features, num_dynamic_features, num_D, num_M, num_R,num_N)
    save_mappings(args, user2id, item2id, item2itemtype)
    args.network='c62'
    args.datapath = "data/{}.csv".format(args.network)
    [user2id_ovr, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence,
    item2id_ovr, item_sequence_id, item_timediffs_sequence, timestamp_sequence,
    static_feature_sequence, dynamic_feature_sequence, y_true,
    item2itemtype_ovr, itemtype_sequence,
    D2id_ovr, M2id_ovr, R2id_ovr,N2id_ovr] = load_network_with_labels(args,item2id,D2id,M2id,R2id,N2id,user2id,True)
    print(len(item_sequence_id))
    print(len(user_sequence_id))
    num_interactions, num_users, num_items, num_static_features, num_dynamic_features, num_D, num_M, num_R, num_N = get_statistics(user_sequence_id, user2id, item2id, static_feature_sequence, dynamic_feature_sequence,
            D2id, M2id, R2id,N2id)
    print_network_statistics(num_interactions, num_users, num_items, num_static_features, num_dynamic_features, num_D, num_M, num_R,num_N)
    # Load Laplacians to use for Laplacian normalizations loss
    L_D, L_M, L_R, D_index_array, M_index_array, R_index_array = load_laplacians(args)
    #L_D = torch.Tensor(L_D).cuda()
    #L_M = torch.Tensor(L_M).cuda()
    #L_R = torch.Tensor(L_R).cuda()
    # Save mappings for later use
    save_mappings(args, user2id, item2id, item2itemtype)

    # Print the statistics of the data before model training
    num_interactions, num_users, num_items, num_static_features, num_dynamic_features, num_D, num_M, num_R, num_N = get_statistics(user_sequence_id, user2id, item2id, static_feature_sequence, dynamic_feature_sequence,
            D2id, M2id, R2id,N2id)
    print_network_statistics(num_interactions, num_users, num_items, num_static_features, num_dynamic_features, num_D, num_M, num_R,num_N)
    # Last item per entity is a dummy item.
    D_idx_for_D_embeddings = np.arange(num_D - 1)
    M_idx_for_M_embeddings = np.arange(num_M - 1)
    R_idx_for_R_embeddings = np.arange(num_R - 1)

    train_end_idx = int(num_interactions)
    tbatch_timespan = args.tbatch_timespan

    # Model initialization
    initial_user_embedding = nn.Parameter(F.normalize(torch.rand(args.embedding_dim).cuda(), dim=0))
    initial_item_embedding = nn.Parameter(F.normalize(torch.rand(args.embedding_dim).cuda(), dim=0))
    initial_D_embedding = nn.Parameter(F.normalize(torch.rand(args.embedding_dim).cuda(), dim=0))
    initial_M_embedding = nn.Parameter(F.normalize(torch.rand(args.embedding_dim).cuda(), dim=0))
    initial_R_embedding = nn.Parameter(F.normalize(torch.rand(args.embedding_dim).cuda(), dim=0))

    # Load static embeddings.
    D_embedding_static, M_embedding_static, R_embedding_static = load_static_emb(args, D2id, M2id, R2id)
    D_embedding_static = Variable(torch.Tensor(D_embedding_static).cuda())
    M_embedding_static = Variable(torch.Tensor(M_embedding_static).cuda())
    R_embedding_static = Variable(torch.Tensor(R_embedding_static).cuda())
    # If entities do not have static embeddings, use onehot instead. E.g. uncomment the following lines
    #D_embedding_static = Variable(torch.eye(num_D).cuda())
    #M_embedding_static = Variable(torch.eye(num_M).cuda())
    #R_embedding_static = Variable(torch.eye(num_R).cuda())
    #N_embedding_static = Variable(torch.eye(num_N).cuda())
    model = DECENT(args, num_static_features, num_dynamic_features, num_users, num_items, D_embedding_static.shape[1], M_embedding_static.shape[1], R_embedding_static.shape[1]).cuda()
    print(model)
    PATH='models/bignotes/DECent_2/Model'+str(23)+'.pt'
    model.load_state_dict(torch.load(PATH), strict=False)
    modb=DECENT(args, num_static_features, num_dynamic_features, num_users, num_items, D_embedding_static.shape[1], M_embedding_static.shape[1], R_embedding_static.shape[1]).cuda()
    modb.load_state_dict(torch.load(PATH), strict=False)

    MSELoss = nn.MSELoss()

    # Embedding initialization
    user_embeddings = initial_user_embedding.repeat(num_users, 1) # initialize all users to the same embedding
    item_embeddings = initial_item_embedding.repeat(num_items, 1) # initialize all items to the same embedding
    D_embeddings = initial_D_embedding.repeat(num_D, 1) # initialize all doctors to the same embedding
    M_embeddings = initial_M_embedding.repeat(num_M, 1) # initialize all meds to the same embedding
    R_embeddings = initial_R_embedding.repeat(num_R, 1) # initialize all rooms to the same embedding

    #item_embedding_static = Variable(torch.eye(num_items).cuda()) # one-hot vectors for static embeddings
    user_embedding_static = Variable(torch.eye(num_users).cuda()) # one-hot vectors for static embeddings

    # Optimizer
    learning_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Model train
    print("\n Training the DECENT model for {} epochs".format(args.epochs))

    # variables to help using tbatch cache between epochs
    is_first_epoch = True

    [cached_tbatches_user, cached_tbatches_item, cached_tbatches_itemtype, cached_tbatches_interactionids,\
            cached_tbatches_static_feature, cached_tbatches_dynamic_feature,\
            cached_tbatches_user_timediffs, cached_tbatches_item_timediffs, cached_tbatches_previous_item] = initialize_dictionaries_for_tbaching()

    [loss_per_timestep, prediction_loss_per_timestep, user_update_loss_per_timestep, item_update_loss_per_timestep, \
            D_loss_per_timestep, M_loss_per_timestep, R_loss_per_timestep,llm_per_timestamp,CL] = initialize_loss_arrays(args)

    patience = args.patience


    hasmap=pd.read_csv('data/Hashmap1.csv')
    hm1=dict(hasmap.values)
    hm= {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
    number_dict={0: '[PAD]', 1: '[CLS]', 2: '[SEP]', 3: '[MASK]'}
    for key,value in hm1.items():
        hm[key]=(value+4)
        number_dict[(value+4)]=key
    #print(len(hm))
    note=pd.read_csv('data/data/chunk5_encoded.csv')
    note2=pd.read_csv('data/data/chunk4_encoded.csv')
    note3=pd.read_csv('data/data/chunk6_encoded.csv')
    note1=pd.concat([note2,note,note3],axis=0)
    print(len(note1))
    note1=note1.dropna(subset=['NOTE_TEXT'])
    note1.reset_index(drop=True)
    #inter=pd.read_csv('data/interactions.csv')
    #inter=inter[['user_id','timestamp']]
    #str_d='2007/12/01'
    #d = datetime.strptime(str_d, "%Y/%m/%d")
    #note1['timestamp']=note1['NOTE_DATETIME'].apply(lambda x: (x-d).days)
    note1=note1[['item_id','timestamp','NOTE_TEXT']]
    #inter['pid']=inter['user_id']
    #inter=inter[['pid','timestamp']]
    sentences=[]
    for row, data in tqdm(note1.iterrows()):
        s=data['NOTE_TEXT']
        ap1=s.split()
        s1=''
        if len(ap1)>323:
            lenap=323
        else:
            lenap=len(ap1)
        for i in range(lenap):
            s1=s1+ap1[i]+' '
        sentences.append(s1)
    #print(len(sentences))
    token_list=list()

    ml=0
    for i in range(len(sentences)):
        sentence=sentences[i]
        arr=[int(s) for s in sentence.split()]
        if len(arr1)>295:
            arr=arr1[0:294]
        else:
            arr=arr1
        sp=''
        for s in arr:
            sp=sp+str(s)+' '
        sentences[i]=sp
        token_list.append(arr)
    #print(ml)
    #print(token_list[:3])
    maxlen = 325 # maximum of length
    batch_size = len(sentences) # Full Chunk Value For now. Need to think about this
    max_pred = 10  # max tokens of prediction
    n_layers = 6 # number of Encoder of Encoder Layer
    n_heads = 12 # number of heads in Multi-Head Attention
    d_model = 50 # Embedding Size
    d_ff = 50 * 4  # 4*d_model, FeedForward dimension
    d_k = d_v = 64  # dimension of K(=Q), V
    n_segments = 2
    batcher=6000
    vocab_size =len(hm)
    batch=make_batch(sentences,hm,token_list,max_pred,maxlen,vocab_size,number_dict)
    input_ids, _, masked_tokens, masked_pos = map(torch.LongTensor, zip(*batch))
    unpickled_df = np.load("data/data/L10T50G100A1ngU_iter99.p",allow_pickle=True)
    voc1=unpickled_df[3]
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

    print('Pretraining BERT')
    lm =BERT()
    #print(lm)
    PATH2='models/bignotes/BERT2/Model'+str(23)+'.pt'
    #PATH2='models/bignotes/multiGPU/Model'+str(190)+'.pt'
    lm.load_state_dict(torch.load(PATH2,map_location='cuda:0'), strict=False)
    criterion = nn.CrossEntropyLoss()

    print('BERT Pretraining Complete')
    lm=lm.cuda()
    embed_w=embed_w.detach()
    masked_pos=masked_pos.detach()
    input_ids=input_ids.detach()
    optimizer = optim.Adam((list(lm.parameters())+list(model.parameters())), lr=learning_rate, weight_decay=1e-5)
    print('Pretraining DECEnt')
    ################################################################################################################################################
    # Epoch
    ################################################################################################################################################
    for ep in tqdm(range(args.decent_pretrain_epochs)):
        print("Epoch {} of {}".format(ep, args.epochs))

        epoch_start_time = time.time()
        # INITIALIZE EMBEDDING TRAJECTORY STORAGE
        user_embeddings_timeseries = Variable(torch.Tensor(num_interactions, args.embedding_dim).cuda())
        item_embeddings_timeseries = Variable(torch.Tensor(num_interactions, args.embedding_dim).cuda())

        optimizer.zero_grad()
        reinitialize_tbatches()
        total_loss, loss, total_interaction_count, prediction_loss, user_update_loss, item_update_loss, D_loss, M_loss, R_loss,_ = zero_loss()

        tbatch_start_time = None
        tbatch_to_insert = -1
        tbatch_full = False

        ################################################################################################################################################
        # Iterate over interactions. j is the index of the interactions
        ################################################################################################################################################
        for j in tqdm(range(train_end_idx)):
            if is_first_epoch:
                userid = user_sequence_id[j]
                itemid = item_sequence_id[j]
                itemtype = itemtype_sequence[j]
                static_feature = static_feature_sequence[j]
                dynamic_feature = dynamic_feature_sequence[j]
                user_timediff = user_timediffs_sequence[j]
                item_timediff = item_timediffs_sequence[j]

                ################################################################################################################################################
                # T-batching (Step1)
                ################################################################################################################################################
                # This is step 1 for preparing T-BATCHES
                # Later, we divide interactions in each batch into the number of types of entities
                tbatch_to_insert = max(lib.tbatchid_user[userid], lib.tbatchid_item[itemid]) + 1
                lib.tbatchid_user[userid] = tbatch_to_insert
                lib.tbatchid_item[itemid] = tbatch_to_insert

                lib.current_tbatches_user[tbatch_to_insert].append(userid)
                lib.current_tbatches_item[tbatch_to_insert].append(itemid)
                lib.current_tbatches_itemtype[tbatch_to_insert].append(itemtype)
                lib.current_tbatches_static_feature[tbatch_to_insert].append(static_feature)
                lib.current_tbatches_dynamic_feature[tbatch_to_insert].append(dynamic_feature)
                lib.current_tbatches_interactionids[tbatch_to_insert].append(j)
                lib.current_tbatches_user_timediffs[tbatch_to_insert].append(user_timediff)
                lib.current_tbatches_item_timediffs[tbatch_to_insert].append(item_timediff)
                lib.current_tbatches_previous_item[tbatch_to_insert].append(user_previous_itemid_sequence[j])

            timestamp = timestamp_sequence[j]
            if tbatch_start_time is None:
                tbatch_start_time = timestamp

            # Train the batches in the tbatch_timespan
            # if timestamp - tbatch_start_time > tbatch_timespan: # using this condition, instances in the later timesteps that do not meet this condition are NOT included in the batches!
            if (timestamp - tbatch_start_time > tbatch_timespan) or (j == train_end_idx-1): # Check if j is the last index
                # idx0: D, idx1: M, idx2: R, idx3: D, idx4: M, idx5: R, ...
                # Split each batch into three batches, based on the itemtype
                if is_first_epoch:
                    ################################################################################################################################################
                    # T-batching (Step2)
                    ################################################################################################################################################
                    tbatch_id = 0 # This is the actual tbatch_id.
                    # max_tbatch_to_insert = tbatch_to_insert # few batches are missed if we simply do this!
                    max_tbatch_to_insert = max(lib.tbatchid_user.values())
                    # iterate over each current_tbatch
                    for tbatch_to_insert in range(max_tbatch_to_insert+1): # max batch_id is inclusive.
                        # iterate over each item in the batch
                        for idx_of_interaction, itemtype_of_interaction in enumerate(lib.current_tbatches_itemtype[tbatch_to_insert]):
                            userid = lib.current_tbatches_user[tbatch_to_insert][idx_of_interaction]
                            itemid = lib.current_tbatches_item[tbatch_to_insert][idx_of_interaction]
                            static_feature = lib.current_tbatches_static_feature[tbatch_to_insert][idx_of_interaction]
                            dynamic_feature = lib.current_tbatches_dynamic_feature[tbatch_to_insert][idx_of_interaction]

                            j_interaction_id = lib.current_tbatches_interactionids[tbatch_to_insert][idx_of_interaction]
                            user_timediff = lib.current_tbatches_user_timediffs[tbatch_to_insert][idx_of_interaction]
                            item_timediff = lib.current_tbatches_item_timediffs[tbatch_to_insert][idx_of_interaction]
                            previous_item = lib.current_tbatches_previous_item[tbatch_to_insert][idx_of_interaction]

                            lib.DECEnt_tbatches_itemtype[tbatch_id] = 'D'
                            lib.DECEnt_tbatches_itemtype[tbatch_id+1] = 'M'
                            lib.DECEnt_tbatches_itemtype[tbatch_id+2] = 'R'

                            if itemtype_of_interaction=='D':
                                lib.DECEnt_tbatches_user[tbatch_id].append(userid)
                                lib.DECEnt_tbatches_item[tbatch_id].append(itemid)
                                lib.DECEnt_tbatches_static_feature[tbatch_id].append(static_feature)
                                lib.DECEnt_tbatches_dynamic_feature[tbatch_id].append(dynamic_feature)
                                lib.DECEnt_tbatches_interactionids[tbatch_id].append(j_interaction_id)
                                lib.DECEnt_tbatches_user_timediffs[tbatch_id].append(user_timediff)
                                lib.DECEnt_tbatches_item_timediffs[tbatch_id].append(item_timediff)
                                lib.DECEnt_tbatches_previous_item[tbatch_id].append(previous_item)

                            elif itemtype_of_interaction=='M':
                                lib.DECEnt_tbatches_user[tbatch_id+1].append(userid)
                                lib.DECEnt_tbatches_item[tbatch_id+1].append(itemid)
                                lib.DECEnt_tbatches_static_feature[tbatch_id+1].append(static_feature)
                                lib.DECEnt_tbatches_dynamic_feature[tbatch_id+1].append(dynamic_feature)
                                lib.DECEnt_tbatches_interactionids[tbatch_id+1].append(j_interaction_id)
                                lib.DECEnt_tbatches_user_timediffs[tbatch_id+1].append(user_timediff)
                                lib.DECEnt_tbatches_item_timediffs[tbatch_id+1].append(item_timediff)
                                lib.DECEnt_tbatches_previous_item[tbatch_id+1].append(previous_item)

                            elif itemtype_of_interaction=='R':
                                lib.DECEnt_tbatches_user[tbatch_id+2].append(userid)
                                lib.DECEnt_tbatches_item[tbatch_id+2].append(itemid)
                                lib.DECEnt_tbatches_static_feature[tbatch_id+2].append(static_feature)
                                lib.DECEnt_tbatches_dynamic_feature[tbatch_id+2].append(dynamic_feature)
                                lib.DECEnt_tbatches_interactionids[tbatch_id+2].append(j_interaction_id)
                                lib.DECEnt_tbatches_user_timediffs[tbatch_id+2].append(user_timediff)
                                lib.DECEnt_tbatches_item_timediffs[tbatch_id+2].append(item_timediff)
                                lib.DECEnt_tbatches_previous_item[tbatch_id+2].append(previous_item)

                        tbatch_id += 3
                # Reset the start time of the next tbatch
                tbatch_start_time = timestamp

                if not is_first_epoch:
                    lib.DECEnt_tbatches_user = cached_tbatches_user[timestamp]
                    lib.DECEnt_tbatches_item = cached_tbatches_item[timestamp]
                    lib.DECEnt_tbatches_itemtype = cached_tbatches_itemtype[timestamp]
                    lib.DECEnt_tbatches_interactionids = cached_tbatches_interactionids[timestamp]
                    lib.DECEnt_tbatches_static_feature = cached_tbatches_static_feature[timestamp]
                    lib.DECEnt_tbatches_dynamic_feature = cached_tbatches_dynamic_feature[timestamp]
                    lib.DECEnt_tbatches_user_timediffs = cached_tbatches_user_timediffs[timestamp]
                    lib.DECEnt_tbatches_item_timediffs = cached_tbatches_item_timediffs[timestamp]
                    lib.DECEnt_tbatches_previous_item = cached_tbatches_previous_item[timestamp]

                # print("\n")
                # print("Number of instances processed: {}".format(j+1))
                # print("Number of instances in lib.current_tbatches_user: {}".format(sum([len(lib.current_tbatches_user[batch_id]) for batch_id in lib.current_tbatches_user])))
                # print("Number of instances in lib.DECEnt_tbatches_user: {}".format(sum([len(lib.DECEnt_tbatches_user[batch_id]) for batch_id in lib.DECEnt_tbatches_user])))
                # print("\n")

                ################################################################################################################################################
                # For the batches in the tbatch_timespan, train the model
                ################################################################################################################################################
                # index upto max batch_id + 1 to include the instances in the last batch
                with trange(max(lib.DECEnt_tbatches_user.keys())+1) as progress_bar3:
                    # Here, i is the batch_id in teh set of batches in the current tbatch_timespan
                    for i in progress_bar3:
                        # If itemtype is 'D', 'R', there are not many interactions, so the batches correspond to these get empty early.
                        if i not in lib.DECEnt_tbatches_user:
                            continue
                        num_interaction_in_batch = len(lib.DECEnt_tbatches_interactionids[i])
                        if num_interaction_in_batch == 0:
                            continue
                        total_interaction_count += num_interaction_in_batch

                        if is_first_epoch:
                            # move the tensors to GPU
                            lib.DECEnt_tbatches_user[i] = torch.LongTensor(lib.DECEnt_tbatches_user[i]).cuda()
                            lib.DECEnt_tbatches_item[i] = torch.LongTensor(lib.DECEnt_tbatches_item[i]).cuda()
                            lib.DECEnt_tbatches_interactionids[i] = torch.LongTensor(lib.DECEnt_tbatches_interactionids[i]).cuda()
                            lib.DECEnt_tbatches_static_feature[i] = torch.Tensor(lib.DECEnt_tbatches_static_feature[i]).cuda()
                            lib.DECEnt_tbatches_dynamic_feature[i] = torch.Tensor(lib.DECEnt_tbatches_dynamic_feature[i]).cuda()
                            lib.DECEnt_tbatches_user_timediffs[i] = torch.Tensor(lib.DECEnt_tbatches_user_timediffs[i]).cuda()
                            lib.DECEnt_tbatches_item_timediffs[i] = torch.Tensor(lib.DECEnt_tbatches_item_timediffs[i]).cuda()
                            lib.DECEnt_tbatches_previous_item[i] = torch.LongTensor(lib.DECEnt_tbatches_previous_item[i]).cuda()

                        tbatch_userids = lib.DECEnt_tbatches_user[i] # Recall "lib.DECEnt_tbatches_user[i]" has unique elements
                        tbatch_itemids = lib.DECEnt_tbatches_item[i] # Recall "lib.DECEnt_tbatches_item[i]" has unique elements
                        tbatch_itemtype = lib.DECEnt_tbatches_itemtype[i] # this is one string.
                        tbatch_interactionids = lib.DECEnt_tbatches_interactionids[i]
                        static_feature_tensor = Variable(lib.DECEnt_tbatches_static_feature[i]) # Recall "lib.DECEnt_tbatches_static_feature[i]" is list of list, so "static_feature_tensor" is a 2-d tensor
                        dynamic_feature_tensor = Variable(lib.DECEnt_tbatches_dynamic_feature[i]) # Recall "lib.DECEnt_tbatches_dynamic_feature[i]" is list of list, so "dynamic_feature_tensor" is a 2-d tensor

                        user_timediffs_tensor = Variable(lib.DECEnt_tbatches_user_timediffs[i]).unsqueeze(1)
                        item_timediffs_tensor = Variable(lib.DECEnt_tbatches_item_timediffs[i]).unsqueeze(1)
                        tbatch_itemids_previous = lib.DECEnt_tbatches_previous_item[i]


                        # item_embedding_previous = item_embeddings[tbatch_itemids_previous,:]

                        ###############################################################################################################

                        # Step1: project user embedding
                        user_embedding_input = user_embeddings[tbatch_userids,:]
                        user_projected_embedding = model.forward(user_embedding_input, None, user_timediffs_tensor, None, None, select='project')

                        # Use the current interaction entity (e.g., med, doc, room) as both input and the vector to compute loss from.
                        # Concatenate user embedding and item embedding
                        # Get the batch of embeddings of current item
                        if tbatch_itemtype == 'D':
                            item_embedding_static_in_batch = D_embedding_static[tbatch_itemids,:]
                            item_embedding_input = D_embeddings[tbatch_itemids,:]
                        elif tbatch_itemtype == 'M':
                            item_embedding_static_in_batch = M_embedding_static[tbatch_itemids,:]
                            item_embedding_input = M_embeddings[tbatch_itemids,:]
                        elif tbatch_itemtype == 'R':
                            item_embedding_static_in_batch = R_embedding_static[tbatch_itemids,:]
                            item_embedding_input = R_embeddings[tbatch_itemids,:]

                        user_item_embedding = torch.cat(
                                [
                                    user_projected_embedding,
                                    item_embedding_input,
                                    item_embedding_static_in_batch,
                                    user_embedding_static[tbatch_userids,:],
                                    static_feature_tensor
                                ],
                                dim=1)

                        # Step2: predict the users' current item interaction
                        predicted_item_embedding = model.predict_item_embedding(user_item_embedding, itemtype=tbatch_itemtype)

                        # Loss1: prediction loss
                        loss_temp = MSELoss(predicted_item_embedding, torch.cat([item_embedding_input, item_embedding_static_in_batch], dim=1).detach())
                        loss += loss_temp
                        prediction_loss += loss_temp

                        # Step3: update dynamic embeddings based on the interaction
                        user_embedding_output = model.forward(user_embedding_input, item_embedding_input, user_timediffs_tensor, static_feature_tensor, dynamic_feature_tensor, select='user{}_update'.format(tbatch_itemtype))
                        item_embedding_output = model.forward(user_embedding_input, item_embedding_input, item_timediffs_tensor, static_feature_tensor, dynamic_feature_tensor, select='item{}_update'.format(tbatch_itemtype))

                        # Step4: Update embedding arrays
                        # item_embeddings[tbatch_itemids,:] = item_embedding_output
                        if tbatch_itemtype == 'D':
                            D_embeddings[tbatch_itemids,:] = item_embedding_output
                        elif tbatch_itemtype == 'M':
                            M_embeddings[tbatch_itemids,:] = item_embedding_output
                        elif tbatch_itemtype == 'R':
                            R_embeddings[tbatch_itemids,:] = item_embedding_output

                        user_embeddings[tbatch_userids,:] = user_embedding_output
                        user_embeddings_timeseries[tbatch_interactionids,:] = user_embedding_output
                        item_embeddings_timeseries[tbatch_interactionids,:] = item_embedding_output # no need to fix this.

                        # Loss2: item update loss (we do not want embeddings to change dramatically per interaction)
                        loss_temp = MSELoss(item_embedding_output, item_embedding_input.detach())
                        loss += loss_temp
                        item_update_loss += loss_temp

                        # Loss3: user update loss (we don not want embeddings to change dramatically per interaction)
                        loss_temp = MSELoss(user_embedding_output, user_embedding_input.detach())
                        loss += loss_temp
                        user_update_loss += loss_temp

                        ##############
                        # Modification: do the laplacian normalization once per epoch!
                        # Loss4-6: items in the same group (e.g. doctors with same specialty) to have similar embeddings
                        #if tbatch_itemtype == 'D':
                            #loss_temp = torch.sum(torch.mm(torch.mm(D_embeddings[D_idx_for_D_embeddings, :].T, L_D), D_embeddings[D_idx_for_D_embeddings, :]))
                            #loss += loss_temp
                            #D_loss += loss_temp
                        #elif tbatch_itemtype == 'M':
                            #loss_temp = torch.sum(torch.mm(torch.mm(M_embeddings[M_idx_for_M_embeddings, :].T, L_M), M_embeddings[M_idx_for_M_embeddings, :]))
                            #loss += loss_temp
                            #M_loss += loss_temp
                        #elif tbatch_itemtype == 'R':
                            #loss_temp = torch.sum(torch.mm(torch.mm(R_embeddings[R_idx_for_R_embeddings, :].T, L_R), R_embeddings[R_idx_for_R_embeddings, :]))
                            #loss += loss_temp
                            #R_loss += loss_temp





                # At the end of t-batch, backpropagate error
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # Reset loss
                loss = 0
                # item_embeddings.detach_() # Detachment is needed to prevent double propagation of gradient
                D_embeddings.detach_() # Detachment is needed to prevent double propagation of gradient
                M_embeddings.detach_() # Detachment is needed to prevent double propagation of gradient
                R_embeddings.detach_() # Detachment is needed to prevent double propagation of gradient
                user_embeddings.detach_()
                item_embeddings_timeseries.detach_()
                user_embeddings_timeseries.detach_()

                # Reinitialize tbatches
                if is_first_epoch:
                    cached_tbatches_user[timestamp] = lib.DECEnt_tbatches_user
                    cached_tbatches_item[timestamp] = lib.DECEnt_tbatches_item
                    cached_tbatches_itemtype[timestamp] = lib.DECEnt_tbatches_itemtype
                    cached_tbatches_interactionids[timestamp] = lib.DECEnt_tbatches_interactionids
                    cached_tbatches_static_feature[timestamp] = lib.DECEnt_tbatches_static_feature
                    cached_tbatches_dynamic_feature[timestamp] = lib.DECEnt_tbatches_dynamic_feature
                    cached_tbatches_user_timediffs[timestamp] = lib.DECEnt_tbatches_user_timediffs
                    cached_tbatches_item_timediffs[timestamp] = lib.DECEnt_tbatches_item_timediffs
                    cached_tbatches_previous_item[timestamp] = lib.DECEnt_tbatches_previous_item

                    reinitialize_tbatches()
                    tbatch_to_insert = -1

        is_first_epoch = False # as first epoch ends here
        print("Last epoch took {} minutes".format((time.time()-epoch_start_time)/60))
        # End of the epoch
        print("\nTotal loss in this epoch = %f" % (total_loss))
        print("\nPrediction loss in this epoch = %f" % (prediction_loss))

        #loss_per_timestep[ep] = total_loss
        #prediction_loss_per_timestep[ep] = prediction_loss
        #user_update_loss_per_timestep[ep] = user_update_loss
        #item_update_loss_per_timestep[ep] = item_update_loss
        #D_loss_per_timestep[ep] = D_loss
        #M_loss_per_timestep[ep] = M_loss
        #R_loss_per_timestep[ep] = R_loss

        # Save D, M, R embeddings in item_embeddings at exact locations
        item_embeddings[D_index_array] = D_embeddings[D_idx_for_D_embeddings]
        item_embeddings[M_index_array] = M_embeddings[M_idx_for_M_embeddings]
        item_embeddings[R_index_array] = R_embeddings[R_idx_for_R_embeddings]
        # print(item_embeddings)

        item_embeddings_dystat = item_embeddings
        user_embeddings_dystat = torch.cat([user_embeddings, user_embedding_static], dim=1)
        # Save model
        # Uncomment the following line if want to save models for each epoch
        # save_model(model, optimizer, args, ep, user_embeddings_dystat, item_embeddings_dystat, train_end_idx, user_embeddings_timeseries, item_embeddings_timeseries)

        # Revert to initial embeddings at the end of each epoch
        # user_embeddings = initial_user_embedding.clone()
        # item_embeddings = initial_item_embedding.clone()
        user_embeddings = initial_user_embedding.repeat(num_users, 1)
        # item_embeddings = initial_item_embedding.repeat(num_items, 1)
        D_embeddings = initial_D_embedding.repeat(num_D, 1)
        M_embeddings = initial_M_embedding.repeat(num_M, 1)
        R_embeddings = initial_R_embedding.repeat(num_R, 1)
        PATHx='models/bignotes/DECent_2/Model'+str(ep)+'.pt'
        torch.save(model.state_dict(), PATHx)


        # user_embeddings = initial_user_embedding.repeat(num_users, 1)
        # item_embeddings = initial_item_embedding.repeat(num_items, 1)

        # Save the loss at every epoch. (not necessary for training. Monitor loss over time.
        #save_loss_arrays(args, loss_per_timestep, prediction_loss_per_timestep, user_update_loss_per_timestep, item_update_loss_per_timestep, D_loss_per_timestep, M_loss_per_timestep, R_loss_per_timestep)

        #if ep > patience and np.argmin(loss_per_timestep[ep-patience: ep])==0:
            #print("Early stopping!")
            #break

        #if ep % 100 == 0:
        #    save_model(model, optimizer, args, ep, user_embeddings_dystat, item_embeddings_dystat, train_end_idx, user_embeddings_timeseries, item_embeddings_timeseries)

    batcher=16
    embed_w=embed_w
    masked_tokens=masked_tokens
    lm.layer1=lm.layer1.cuda()
    lm.layer1.enc_self_attn=lm.layer1.enc_self_attn.cuda()
    lm.layer1.pos_ffn=lm.layer1.pos_ffn.cuda()
    lm.layer1.enc_self_attn.W_Q=lm.layer1.enc_self_attn.W_Q.cuda()
    lm.layer1.enc_self_attn.W_K=lm.layer1.enc_self_attn.W_K.cuda()
    lm.layer1.enc_self_attn.W_V=lm.layer1.enc_self_attn.W_V.cuda()
    lm.layer1.enc_self_attn.fc=lm.layer1.enc_self_attn.fc.cuda()
    lm.layer1.enc_self_attn.norm=lm.layer1.enc_self_attn.norm.cuda()
    lm.layer1.pos_ffn.fc1=lm.layer1.pos_ffn.fc1.cuda()
    lm.layer1.pos_ffn.fc2=lm.layer1.pos_ffn.fc2.cuda()
    lm.layer2=lm.layer2.cuda()
    lm.layer2.enc_self_attn=lm.layer2.enc_self_attn.cuda()
    lm.layer2.pos_ffn=lm.layer2.pos_ffn.cuda()
    lm.layer2.enc_self_attn.W_Q=lm.layer2.enc_self_attn.W_Q.cuda()
    lm.layer2.enc_self_attn.W_K=lm.layer2.enc_self_attn.W_K.cuda()
    lm.layer2.enc_self_attn.W_V=lm.layer2.enc_self_attn.W_V.cuda()
    lm.layer2.enc_self_attn.fc=lm.layer2.enc_self_attn.fc.cuda()
    lm.layer2.enc_self_attn.norm=lm.layer2.enc_self_attn.norm.cuda()
    lm.layer2.pos_ffn.fc1=lm.layer2.pos_ffn.fc1.cuda()
    lm.layer2.pos_ffn.fc2=lm.layer2.pos_ffn.fc2.cuda()
    lm.layer3=lm.layer3.cuda()
    lm.layer3.enc_self_attn=lm.layer3.enc_self_attn.cuda()
    lm.layer3.pos_ffn=lm.layer3.pos_ffn.cuda()
    lm.layer3.enc_self_attn.W_Q=lm.layer3.enc_self_attn.W_Q.cuda()
    lm.layer3.enc_self_attn.W_K=lm.layer3.enc_self_attn.W_K.cuda()
    lm.layer3.enc_self_attn.W_V=lm.layer3.enc_self_attn.W_V.cuda()
    lm.layer3.enc_self_attn.fc=lm.layer3.enc_self_attn.fc.cuda()
    lm.layer3.enc_self_attn.norm=lm.layer3.enc_self_attn.norm.cuda()
    lm.layer3.pos_ffn.fc1=lm.layer3.pos_ffn.fc1.cuda()
    lm.layer3.pos_ffn.fc2=lm.layer3.pos_ffn.fc2.cuda()
    lm.layer4=lm.layer4.cuda()
    lm.layer4.enc_self_attn=lm.layer4.enc_self_attn.cuda()
    lm.layer4.pos_ffn=lm.layer4.pos_ffn.cuda()
    lm.layer4.enc_self_attn.W_Q=lm.layer4.enc_self_attn.W_Q.cuda()
    lm.layer4.enc_self_attn.W_K=lm.layer4.enc_self_attn.W_K.cuda()
    lm.layer4.enc_self_attn.W_V=lm.layer4.enc_self_attn.W_V.cuda()
    lm.layer4.enc_self_attn.fc=lm.layer4.enc_self_attn.fc.cuda()
    lm.layer4.enc_self_attn.norm=lm.layer4.enc_self_attn.norm.cuda()
    lm.layer4.pos_ffn.fc1=lm.layer4.pos_ffn.fc1.cuda()
    lm.layer4.pos_ffn.fc2=lm.layer4.pos_ffn.fc2.cuda()
    lm.layer5=lm.layer5.cuda()
    lm.layer5.enc_self_attn=lm.layer5.enc_self_attn.cuda()
    lm.layer5.pos_ffn=lm.layer5.pos_ffn.cuda()
    lm.layer5.enc_self_attn.W_Q=lm.layer5.enc_self_attn.W_Q.cuda()
    lm.layer5.enc_self_attn.W_K=lm.layer5.enc_self_attn.W_K.cuda()
    lm.layer5.enc_self_attn.W_V=lm.layer5.enc_self_attn.W_V.cuda()
    lm.layer5.enc_self_attn.fc=lm.layer5.enc_self_attn.fc.cuda()
    lm.layer5.enc_self_attn.norm=lm.layer5.enc_self_attn.norm.cuda()
    lm.layer5.pos_ffn.fc1=lm.layer5.pos_ffn.fc1.cuda()
    lm.layer5.pos_ffn.fc2=lm.layer5.pos_ffn.fc2.cuda()
    lm.layer6=lm.layer6.cuda()
    lm.layer6.enc_self_attn=lm.layer6.enc_self_attn.cuda()
    lm.layer6.pos_ffn=lm.layer6.pos_ffn.cuda()
    lm.layer6.enc_self_attn.W_Q=lm.layer6.enc_self_attn.W_Q.cuda()
    lm.layer6.enc_self_attn.W_K=lm.layer6.enc_self_attn.W_K.cuda()
    lm.layer6.enc_self_attn.W_V=lm.layer6.enc_self_attn.W_V.cuda()
    lm.layer6.enc_self_attn.fc=lm.layer6.enc_self_attn.fc.cuda()
    lm.layer6.enc_self_attn.norm=lm.layer6.enc_self_attn.norm.cuda()
    lm.layer6.pos_ffn.fc1=lm.layer6.pos_ffn.fc1.cuda()
    lm.layer6.pos_ffn.fc2=lm.layer6.pos_ffn.fc2.cuda()
    lm.linear=lm.linear.cuda()
    lm.norm=lm.norm.cuda()
    is_first_epoch = True
    epx=range(args.epochs)
    #tbatch_timespan=
    [cached_tbatches_user, cached_tbatches_item, cached_tbatches_itemtype, cached_tbatches_interactionids,\
            cached_tbatches_static_feature, cached_tbatches_dynamic_feature,\
            cached_tbatches_user_timediffs, cached_tbatches_item_timediffs, cached_tbatches_previous_item] = initialize_dictionaries_for_tbaching()

    [loss_per_timestep, prediction_loss_per_timestep, user_update_loss_per_timestep, item_update_loss_per_timestep, \
            D_loss_per_timestep, M_loss_per_timestep, R_loss_per_timestep,llm_per_timestamp,CL] = initialize_loss_arrays(args)

    #lm.decoder=lm.decoder.cuda()
    #lm.decoder_bias=lm.decoder_bias.cuda()
    #output=output.cuda()
    #segment_ids=segment_ids.cuda()
    #masked_pos=masked_pos.cuda()
    plo=0
    ################################################################################################################################################
    # Joint Training
    ################################################################################################################################################
    print('Joint Training Start')
    #N_embeddings=torch.zeros((batch_size, args.embedding_dim))
    for ep in tqdm(epx):
        print("Epoch {} of {}".format(ep, args.epochs))
        #torch.cuda.empty_cache()

        epoch_start_time = time.time()
        # INITIALIZE EMBEDDING TRAJECTORY STORAGE
        user_embeddings_timeseries = Variable(torch.Tensor(num_interactions, args.embedding_dim).cuda())
        item_embeddings_timeseries = Variable(torch.Tensor(num_interactions, args.embedding_dim).cuda())

        optimizer.zero_grad()
        #optimizer_BERT.zero_grad()
        reinitialize_tbatches()
        total_loss, loss, total_interaction_count, prediction_loss, user_update_loss, item_update_loss, D_loss, M_loss, R_loss,llm = zero_loss()
        loss_cl=0
        #llm=0
        #i=0
        #loss_b=0
        """while (i+batcher)<batch_size:
            _, embi = lm(embed_w[i:i+batcher], segment_ids[i:i+batcher], masked_pos[i:i+batcher],input_ids[i:i+batcher],'joint')
            #h_masked=h_masked.detach()
            N_embeddings[i:i+batcher]=embi
            i=i+batcher

        if i<batch_size-1:
            _, embi = lm(embed_w[i:batch_size], segment_ids[i:batch_size], masked_pos[i:batch_size],input_ids[i:batch_size],'joint')
            #h_masked=h_masked.detach()
            N_embeddings[i:batch_size]=embi
        N_embeddings=N_embeddings.cuda()
        """
        tbatch_start_time = None
        tbatch_to_insert = -1
        tbatch_full = False

        ################################################################################################################################################
        # Iterate over interactions. j is the index of the interactions
        ################################################################################################################################################
        for j in tqdm(range(train_end_idx)):
            if is_first_epoch:
                userid = user_sequence_id[j]
                itemid = item_sequence_id[j]
                itemtype = itemtype_sequence[j]
                static_feature = static_feature_sequence[j]
                dynamic_feature = dynamic_feature_sequence[j]
                user_timediff = user_timediffs_sequence[j]
                item_timediff = item_timediffs_sequence[j]

                ################################################################################################################################################
                # T-batching (Step1)
                ################################################################################################################################################
                # This is step 1 for preparing T-BATCHES
                # Later, we divide interactions in each batch into the number of types of entities
                tbatch_to_insert = max(lib.tbatchid_user[userid], lib.tbatchid_item[itemid]) + 1
                lib.tbatchid_user[userid] = tbatch_to_insert
                lib.tbatchid_item[itemid] = tbatch_to_insert

                lib.current_tbatches_user[tbatch_to_insert].append(userid)
                lib.current_tbatches_item[tbatch_to_insert].append(itemid)
                lib.current_tbatches_itemtype[tbatch_to_insert].append(itemtype)
                lib.current_tbatches_static_feature[tbatch_to_insert].append(static_feature)
                lib.current_tbatches_dynamic_feature[tbatch_to_insert].append(dynamic_feature)
                lib.current_tbatches_interactionids[tbatch_to_insert].append(j)
                lib.current_tbatches_user_timediffs[tbatch_to_insert].append(user_timediff)
                lib.current_tbatches_item_timediffs[tbatch_to_insert].append(item_timediff)
                lib.current_tbatches_previous_item[tbatch_to_insert].append(user_previous_itemid_sequence[j])

            timestamp = timestamp_sequence[j]
            if tbatch_start_time is None:
                tbatch_start_time = timestamp

            # Train the batches in the tbatch_timespan
            # if timestamp - tbatch_start_time > tbatch_timespan: # using this condition, instances in the later timesteps that do not meet this condition are NOT included in the batches!
            if (timestamp - tbatch_start_time > tbatch_timespan) or (j == train_end_idx-1): # Check if j is the last index
                # idx0: D, idx1: M, idx2: R, idx3: D, idx4: M, idx5: R, ...
                # Split each batch into three batches, based on the itemtype
                if is_first_epoch:
                    ################################################################################################################################################
                    # T-batching (Step2)
                    ################################################################################################################################################
                    tbatch_id = 0 # This is the actual tbatch_id.
                    # max_tbatch_to_insert = tbatch_to_insert # few batches are missed if we simply do this!
                    max_tbatch_to_insert = max(lib.tbatchid_user.values())
                    # iterate over each current_tbatch
                    for tbatch_to_insert in range(max_tbatch_to_insert+1): # max batch_id is inclusive.
                        # iterate over each item in the batch
                        xi=3
                        for idx_of_interaction, itemtype_of_interaction in enumerate(lib.current_tbatches_itemtype[tbatch_to_insert]):
                            userid = lib.current_tbatches_user[tbatch_to_insert][idx_of_interaction]
                            itemid = lib.current_tbatches_item[tbatch_to_insert][idx_of_interaction]
                            static_feature = lib.current_tbatches_static_feature[tbatch_to_insert][idx_of_interaction]
                            dynamic_feature = lib.current_tbatches_dynamic_feature[tbatch_to_insert][idx_of_interaction]

                            j_interaction_id = lib.current_tbatches_interactionids[tbatch_to_insert][idx_of_interaction]
                            user_timediff = lib.current_tbatches_user_timediffs[tbatch_to_insert][idx_of_interaction]
                            item_timediff = lib.current_tbatches_item_timediffs[tbatch_to_insert][idx_of_interaction]
                            previous_item = lib.current_tbatches_previous_item[tbatch_to_insert][idx_of_interaction]

                            lib.DECEnt_tbatches_itemtype[tbatch_id] = 'D'
                            lib.DECEnt_tbatches_itemtype[tbatch_id+1] = 'M'
                            lib.DECEnt_tbatches_itemtype[tbatch_id+2] = 'R'
                            lib.DECEnt_tbatches_itemtype[tbatch_id+xi] = 'N'

                            if itemtype_of_interaction=='D':
                                lib.DECEnt_tbatches_user[tbatch_id].append(userid)
                                lib.DECEnt_tbatches_item[tbatch_id].append(itemid)
                                lib.DECEnt_tbatches_static_feature[tbatch_id].append(static_feature)
                                lib.DECEnt_tbatches_dynamic_feature[tbatch_id].append(dynamic_feature)
                                lib.DECEnt_tbatches_interactionids[tbatch_id].append(j_interaction_id)
                                lib.DECEnt_tbatches_user_timediffs[tbatch_id].append(user_timediff)
                                lib.DECEnt_tbatches_item_timediffs[tbatch_id].append(item_timediff)
                                lib.DECEnt_tbatches_previous_item[tbatch_id].append(previous_item)

                            elif itemtype_of_interaction=='M':
                                lib.DECEnt_tbatches_user[tbatch_id+1].append(userid)
                                lib.DECEnt_tbatches_item[tbatch_id+1].append(itemid)
                                lib.DECEnt_tbatches_static_feature[tbatch_id+1].append(static_feature)
                                lib.DECEnt_tbatches_dynamic_feature[tbatch_id+1].append(dynamic_feature)
                                lib.DECEnt_tbatches_interactionids[tbatch_id+1].append(j_interaction_id)
                                lib.DECEnt_tbatches_user_timediffs[tbatch_id+1].append(user_timediff)
                                lib.DECEnt_tbatches_item_timediffs[tbatch_id+1].append(item_timediff)
                                lib.DECEnt_tbatches_previous_item[tbatch_id+1].append(previous_item)

                            elif itemtype_of_interaction=='R':
                                lib.DECEnt_tbatches_user[tbatch_id+2].append(userid)
                                lib.DECEnt_tbatches_item[tbatch_id+2].append(itemid)
                                lib.DECEnt_tbatches_static_feature[tbatch_id+2].append(static_feature)
                                lib.DECEnt_tbatches_dynamic_feature[tbatch_id+2].append(dynamic_feature)
                                lib.DECEnt_tbatches_interactionids[tbatch_id+2].append(j_interaction_id)
                                lib.DECEnt_tbatches_user_timediffs[tbatch_id+2].append(user_timediff)
                                lib.DECEnt_tbatches_item_timediffs[tbatch_id+2].append(item_timediff)
                                lib.DECEnt_tbatches_previous_item[tbatch_id+2].append(previous_item)

                            elif itemtype_of_interaction=='N':
                                if len(lib.DECEnt_tbatches_user[tbatch_id+xi])>130:
                                    #print(tbatch_id+xi)
                                    xi=xi+1
                                    lib.DECEnt_tbatches_itemtype[tbatch_id+xi] = 'N'
                                lib.DECEnt_tbatches_user[tbatch_id+xi].append(userid)
                                lib.DECEnt_tbatches_item[tbatch_id+xi].append(itemid)
                                lib.DECEnt_tbatches_static_feature[tbatch_id+xi].append(static_feature)
                                lib.DECEnt_tbatches_dynamic_feature[tbatch_id+xi].append(dynamic_feature)
                                lib.DECEnt_tbatches_interactionids[tbatch_id+xi].append(j_interaction_id)
                                lib.DECEnt_tbatches_user_timediffs[tbatch_id+xi].append(user_timediff)
                                lib.DECEnt_tbatches_item_timediffs[tbatch_id+xi].append(item_timediff)
                                lib.DECEnt_tbatches_previous_item[tbatch_id+xi].append(previous_item)

                        tbatch_id += xi+1
                # Reset the start time of the next tbatch
                tbatch_start_time = timestamp
                #print("\n")
                #print("Number of instances processed: {}".format(j+1))
                #print("Number of instances in lib.current_tbatches_user: {}".format(sum([len(lib.current_tbatches_user[batch_id]) for batch_id in lib.current_tbatches_user])))
                #print("Number of instances in lib.DECEnt_tbatches_user: {}".format(sum([len(lib.DECEnt_tbatches_user[batch_id]) for batch_id in lib.DECEnt_tbatches_user])))
                #print("\n")
                if not is_first_epoch:
                    lib.DECEnt_tbatches_user = cached_tbatches_user[timestamp]
                    lib.DECEnt_tbatches_item = cached_tbatches_item[timestamp]
                    lib.DECEnt_tbatches_itemtype = cached_tbatches_itemtype[timestamp]
                    lib.DECEnt_tbatches_interactionids = cached_tbatches_interactionids[timestamp]
                    lib.DECEnt_tbatches_static_feature = cached_tbatches_static_feature[timestamp]
                    lib.DECEnt_tbatches_dynamic_feature = cached_tbatches_dynamic_feature[timestamp]
                    lib.DECEnt_tbatches_user_timediffs = cached_tbatches_user_timediffs[timestamp]
                    lib.DECEnt_tbatches_item_timediffs = cached_tbatches_item_timediffs[timestamp]
                    lib.DECEnt_tbatches_previous_item = cached_tbatches_previous_item[timestamp]

                # print("\n")
                # print("Number of instances processed: {}".format(j+1))
                # print("Number of instances in lib.current_tbatches_user: {}".format(sum([len(lib.current_tbatches_user[batch_id]) for batch_id in lib.current_tbatches_user])))
                # print("Number of instances in lib.DECEnt_tbatches_user: {}".format(sum([len(lib.DECEnt_tbatches_user[batch_id]) for batch_id in lib.DECEnt_tbatches_user])))
                # print("\n")

                ################################################################################################################################################
                # For the batches in the tbatch_timespan, train the model
                ################################################################################################################################################
                # index upto max batch_id + 1 to include the instances in the last batch

                with trange(max(lib.DECEnt_tbatches_user.keys())+1) as progress_bar3:
                    # Here, i is the batch_id in teh set of batches in the current tbatch_timespan
                    for i in progress_bar3:
                        # If itemtype is 'D', 'R', there are not many interactions, so the batches correspond to these get empty early.
                        if i not in lib.DECEnt_tbatches_user:
                            continue
                        num_interaction_in_batch = len(lib.DECEnt_tbatches_interactionids[i])

                        if num_interaction_in_batch == 0:
                            continue
                        total_interaction_count += num_interaction_in_batch

                        if is_first_epoch:
                            # move the tensors to GPU
                            lib.DECEnt_tbatches_user[i] = torch.LongTensor(lib.DECEnt_tbatches_user[i]).cuda()
                            lib.DECEnt_tbatches_item[i] = torch.LongTensor(lib.DECEnt_tbatches_item[i]).cuda()
                            lib.DECEnt_tbatches_interactionids[i] = torch.LongTensor(lib.DECEnt_tbatches_interactionids[i]).cuda()
                            lib.DECEnt_tbatches_static_feature[i] = torch.Tensor(lib.DECEnt_tbatches_static_feature[i]).cuda()
                            lib.DECEnt_tbatches_dynamic_feature[i] = torch.Tensor(lib.DECEnt_tbatches_dynamic_feature[i]).cuda()
                            lib.DECEnt_tbatches_user_timediffs[i] = torch.Tensor(lib.DECEnt_tbatches_user_timediffs[i]).cuda()
                            lib.DECEnt_tbatches_item_timediffs[i] = torch.Tensor(lib.DECEnt_tbatches_item_timediffs[i]).cuda()
                            lib.DECEnt_tbatches_previous_item[i] = torch.LongTensor(lib.DECEnt_tbatches_previous_item[i]).cuda()

                        tbatch_userids = lib.DECEnt_tbatches_user[i] # Recall "lib.DECEnt_tbatches_user[i]" has unique elements
                        tbatch_itemids = lib.DECEnt_tbatches_item[i] # Recall "lib.DECEnt_tbatches_item[i]" has unique elements
                        tbatch_itemtype = lib.DECEnt_tbatches_itemtype[i] # this is one string.
                        tbatch_interactionids = lib.DECEnt_tbatches_interactionids[i]
                        static_feature_tensor = Variable(lib.DECEnt_tbatches_static_feature[i]) # Recall "lib.DECEnt_tbatches_static_feature[i]" is list of list, so "static_feature_tensor" is a 2-d tensor
                        dynamic_feature_tensor = Variable(lib.DECEnt_tbatches_dynamic_feature[i]) # Recall "lib.DECEnt_tbatches_dynamic_feature[i]" is list of list, so "dynamic_feature_tensor" is a 2-d tensor

                        user_timediffs_tensor = Variable(lib.DECEnt_tbatches_user_timediffs[i]).unsqueeze(1)
                        item_timediffs_tensor = Variable(lib.DECEnt_tbatches_item_timediffs[i]).unsqueeze(1)
                        tbatch_itemids_previous = lib.DECEnt_tbatches_previous_item[i]


                        # item_embedding_previous = item_embeddings[tbatch_itemids_previous,:]

                        ###############################################################################################################

                        # Step1: project user embedding
                        user_embedding_input = user_embeddings[tbatch_userids,:]
                        user_projected_embedding = model.forward1(user_embedding_input, None, user_timediffs_tensor, None, None, select='project')

                        # Use the current interaction entity (e.g., med, doc, room) as both input and the vector to compute loss from.
                        # Concatenate user embedding and item embedding
                        # Get the batch of embeddings of current item
                        if tbatch_itemtype == 'D':
                            item_embedding_static_in_batch = D_embedding_static[tbatch_itemids,:]
                            item_embedding_input = D_embeddings[tbatch_itemids,:]
                        elif tbatch_itemtype == 'M':
                            item_embedding_static_in_batch = M_embedding_static[tbatch_itemids,:]
                            item_embedding_input = M_embeddings[tbatch_itemids,:]
                        elif tbatch_itemtype == 'R':
                            item_embedding_static_in_batch = R_embedding_static[tbatch_itemids,:]
                            item_embedding_input = R_embeddings[tbatch_itemids,:]
                        elif tbatch_itemtype == 'N':
                            #item_embedding_static_in_batch = N_embedding_static[tbatch_itemids,:]
                            N_inputs = embed_w[tbatch_itemids,:].cuda()
                            mt=masked_tokens[tbatch_itemids,:].cuda()
                            mpos=masked_pos[tbatch_itemids,:].cuda()
                            ipi=input_ids[tbatch_itemids,:].cuda()
                            #N_embeddings=torch.zeros((N_inputs.shape[0],128)).cuda()
                            px=0
                            llm1=0

                            predis, item_embedding_static_in_batch = lm(N_inputs, mpos,ipi,'joint')
                            llm1 += (criterion(predis.transpose(1, 2), mt).float()).mean()
                            loss+=llm1
                            llm=llm1.item()

                            #N_embeddings=embi

                            #item_embedding_input=embi
                        if tbatch_itemtype != 'N':

                            user_item_embedding = torch.cat(
                                [
                                    user_projected_embedding,
                                    item_embedding_input,
                                    item_embedding_static_in_batch,
                                    user_embedding_static[tbatch_userids,:],
                                    static_feature_tensor
                                ],
                                dim=1)

                        # Step2: predict the users' current item interaction
                        #predi=torch.cat([item_embedding_input, item_embedding_static_in_batch], dim=1).detach()
                        if tbatch_itemtype!='N':
                            predicted_item_embedding = model.predict_item_embedding1(user_item_embedding, itemtype=tbatch_itemtype)

                            # Loss1: prediction loss
                            loss_temp = MSELoss(predicted_item_embedding, torch.cat([item_embedding_input, item_embedding_static_in_batch], dim=1).detach())
                            loss += loss_temp
                            prediction_loss += loss_temp

                        # Step3: update dynamic embeddings based on the interaction
                        #pm=item_embedding_input.detach()
                        #if tbatch_itemtype == 'N':
                        #    print('N')
                        if tbatch_itemtype != 'N':
                            user_embedding_output = model.forward1(user_embedding_input, item_embedding_input, user_timediffs_tensor, static_feature_tensor, dynamic_feature_tensor, select='user{}_update'.format(tbatch_itemtype))
                        else:
                            user_embedding_output = model.forward1(user_embedding_input, item_embedding_static_in_batch, user_timediffs_tensor, static_feature_tensor, dynamic_feature_tensor, select='user{}_update'.format(tbatch_itemtype))
                        if tbatch_itemtype != 'N':
                            item_embedding_output = model.forward1(user_embedding_input, item_embedding_input, item_timediffs_tensor, static_feature_tensor, dynamic_feature_tensor, select='item{}_update'.format(tbatch_itemtype))

                        # Step4: Update embedding arrays
                        # item_embeddings[tbatch_itemids,:] = item_embedding_output
                        if tbatch_itemtype == 'D':
                            D_embeddings[tbatch_itemids,:] = item_embedding_output
                        elif tbatch_itemtype == 'M':
                            M_embeddings[tbatch_itemids,:] = item_embedding_output
                        elif tbatch_itemtype == 'R':
                            R_embeddings[tbatch_itemids,:] = item_embedding_output

                        user_embeddings[tbatch_userids,:] = user_embedding_output
                        user_embeddings_timeseries[tbatch_interactionids,:] = user_embedding_output
                        if tbatch_itemtype != 'N':
                            item_embeddings_timeseries[tbatch_interactionids,:] = item_embedding_output # no need to fix this.

                        # Loss2: item update loss (we don not want embeddings to change dramatically per interaction)
                        if tbatch_itemtype != 'N':
                            loss_temp = MSELoss(item_embedding_output, item_embedding_input.detach())
                            loss += loss_temp
                            item_update_loss += loss_temp

                        # Loss3: user update loss (we don not want embeddings to change dramatically per interaction)
                        loss_temp = MSELoss(user_embedding_output, user_embedding_input.detach())
                        loss += loss_temp
                        user_update_loss += loss_temp

                        ##############
                        # Modification: do the laplacian normalization once per epoch!
                        # Loss4-6: items in the same group (e.g. doctors with same specialty) to have similar embeddings
                        #if tbatch_itemtype == 'D':
                            #loss_temp = torch.sum(torch.mm(torch.mm(D_embeddings[D_idx_for_D_embeddings, :].T, L_D), D_embeddings[D_idx_for_D_embeddings, :]))
                            #loss += loss_temp
                            #D_loss += loss_temp
                        #elif tbatch_itemtype == 'M':
                            #loss_temp = torch.sum(torch.mm(torch.mm(M_embeddings[M_idx_for_M_embeddings, :].T, L_M), M_embeddings[M_idx_for_M_embeddings, :]))
                            #loss += loss_temp
                            #M_loss += loss_temp
                        #elif tbatch_itemtype == 'R':
                            #loss_temp = torch.sum(torch.mm(torch.mm(R_embeddings[R_idx_for_R_embeddings, :].T, L_R), R_embeddings[R_idx_for_R_embeddings, :]))
                            #loss += loss_temp
                            #R_loss += loss_temp
                        reg=0.1
                        loss+=reg*sum((x - y).abs().sum() for x, y in zip(model.state_dict().values(), modb.state_dict().values()))
                        loss_cl+=reg*sum((x - y).abs().sum() for x, y in zip(model.state_dict().values(), modb.state_dict().values()))

                        total_loss += loss.item()
                        loss.backward()
                        optimizer.step()
                        #optimizer_BERT.step()
                        optimizer.zero_grad()

                        # Reset loss
                        loss = 0
                        # item_embeddings.detach_() # Detachment is needed to prevent double propagation of gradient
                        D_embeddings.detach_() # Detachment is needed to prevent double propagation of gradient
                        M_embeddings.detach_() # Detachment is needed to prevent double propagation of gradient
                        R_embeddings.detach_() # Detachment is needed to prevent double propagation of gradient
                        #N_embeddings.detach_() # Detachment is needed to prevent double propagation of gradient
                        user_embeddings.detach_()
                        item_embeddings_timeseries.detach_()
                        user_embeddings_timeseries.detach_()

                        user_embeddings = initial_user_embedding.repeat(num_users, 1)
                        # item_embeddings = initial_item_embedding.repeat(num_items, 1)
                        D_embeddings = initial_D_embedding.repeat(num_D, 1)
                        M_embeddings = initial_M_embedding.repeat(num_M, 1)
                        R_embeddings = initial_R_embedding.repeat(num_R, 1)
                CL[ep]=CL[ep]+loss_cl.item()
                # At the end of t-batch, backpropagate error
                # CL Loss
                #loss_cl=0
                #reg=0.1
                #loss+=reg*sum((x - y).abs().sum() for x, y in zip(model.state_dict().values(), modb.state_dict().values()))
                #loss_cl+=reg*sum((x - y).abs().sum() for x, y in zip(model.state_dict().values(), modb.state_dict().values()))
                #print('CL Loss')
                #print(loss_cl)


                # Reinitialize tbatches
                if is_first_epoch:
                    cached_tbatches_user[timestamp] = lib.DECEnt_tbatches_user
                    cached_tbatches_item[timestamp] = lib.DECEnt_tbatches_item
                    cached_tbatches_itemtype[timestamp] = lib.DECEnt_tbatches_itemtype
                    cached_tbatches_interactionids[timestamp] = lib.DECEnt_tbatches_interactionids
                    cached_tbatches_static_feature[timestamp] = lib.DECEnt_tbatches_static_feature
                    cached_tbatches_dynamic_feature[timestamp] = lib.DECEnt_tbatches_dynamic_feature
                    cached_tbatches_user_timediffs[timestamp] = lib.DECEnt_tbatches_user_timediffs
                    cached_tbatches_item_timediffs[timestamp] = lib.DECEnt_tbatches_item_timediffs
                    cached_tbatches_previous_item[timestamp] = lib.DECEnt_tbatches_previous_item

                    reinitialize_tbatches()
                    tbatch_to_insert = -1

        is_first_epoch = False # as first epoch ends here
        print("Last epoch took {} minutes".format((time.time()-epoch_start_time)/60))
        # End of the epoch
        print("\nTotal loss in this epoch = %f" % (total_loss))
        print("\nPrediction loss in this epoch = %f" % (prediction_loss))

        loss_per_timestep[ep] = total_loss
        prediction_loss_per_timestep[ep] = prediction_loss
        user_update_loss_per_timestep[ep] = user_update_loss
        item_update_loss_per_timestep[ep] = item_update_loss
        D_loss_per_timestep[ep] = D_loss
        M_loss_per_timestep[ep] = M_loss
        R_loss_per_timestep[ep] = R_loss
        llm_per_timestamp[ep]=llm

        # Save D, M, R embeddings in item_embeddings at exact locations
        item_embeddings[D_index_array] = D_embeddings[D_idx_for_D_embeddings]
        item_embeddings[M_index_array] = M_embeddings[M_idx_for_M_embeddings]
        item_embeddings[R_index_array] = R_embeddings[R_idx_for_R_embeddings]
        # print(item_embeddings)

        item_embeddings_dystat = item_embeddings
        user_embeddings_dystat = torch.cat([user_embeddings, user_embedding_static], dim=1)
        # Save model
        # Uncomment the following line if want to save models for each epoch
        # save_model(model, optimizer, args, ep, user_embeddings_dystat, item_embeddings_dystat, train_end_idx, user_embeddings_timeseries, item_embeddings_timeseries)

        # Revert to initial embeddings at the end of each epoch
        # user_embeddings = initial_user_embedding.clone()
        # item_embeddings = initial_item_embedding.clone()
        user_embeddings = initial_user_embedding.repeat(num_users, 1)
        # item_embeddings = initial_item_embedding.repeat(num_items, 1)
        D_embeddings = initial_D_embedding.repeat(num_D, 1)
        M_embeddings = initial_M_embedding.repeat(num_M, 1)
        R_embeddings = initial_R_embedding.repeat(num_R, 1)

        # user_embeddings = initial_user_embedding.repeat(num_users, 1)
        # item_embeddings = initial_item_embedding.repeat(num_items, 1)

        # Save the loss at every epoch. (not necessary for training. Monitor loss over time.
        save_loss_arrays(args, loss_per_timestep, prediction_loss_per_timestep, user_update_loss_per_timestep, item_update_loss_per_timestep, D_loss_per_timestep, M_loss_per_timestep, R_loss_per_timestep, llm_per_timestamp,CL)

        #if ep > patience and np.argmin(loss_per_timestep[ep-patience: ep])==0:
            #print("Early stopping!")
            #break
        PATHx='models/bignotes/DECent_3/Model'+str(ep)+'.pt'
        torch.save(model.state_dict(), PATHx)
        PATHy='models/bignotes/BERT3/Model'+str(ep)+'.pt'
        torch.save(lm.state_dict(), PATHy)
        #if ep % 5 == 0:
        save_model(model, optimizer, args, ep, user_embeddings_dystat, item_embeddings_dystat, train_end_idx, user_embeddings_timeseries, item_embeddings_timeseries)
        #if ep // 50 == 49:
            #save_model(model, optimizer, args, ep, user_embeddings_dystat, item_embeddings_dystat, train_end_idx, user_embeddings_timeseries, item_embeddings_timeseries)
    # Training end.
    print("\nTraining complete. Save final model")
    save_model(model, optimizer, args, ep, user_embeddings_dystat, item_embeddings_dystat, train_end_idx, user_embeddings_timeseries, item_embeddings_timeseries)
