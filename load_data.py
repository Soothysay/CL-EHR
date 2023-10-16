'''
Author: -
Email: -
Last Modified: Sep, 2021

This code contains functions to load data for training
'''

from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import scale
import pandas as pd
from tqdm import tqdm
import math
import re
from random import *
import numpy as np
import sys
from datetime import timedelta
from datetime import datetime


def load_laplacians(args):
    npzfile = np.load(args.laplacian)
    L_D = npzfile["L_D"]
    L_M = npzfile["L_M"]
    L_R = npzfile["L_R"]
    D_index_array = npzfile["D_index_array"]
    M_index_array = npzfile["M_index_array"]
    R_index_array = npzfile["R_index_array"]
    npzfile.close()
    return L_D, L_M, L_R, D_index_array, M_index_array, R_index_array

def load_static_emb(args, D2id, M2id, R2id):
    npzfile = np.load(args.doctor_static, allow_pickle=True)
    D_node_mapping = npzfile["node_mapping"].item()
    D_index_mapping = npzfile["index_mapping"].item()
    D_emb = npzfile["embedding"]
    npzfile.close()

    npzfile = np.load(args.medication_static, allow_pickle=True)
    M_node_mapping = npzfile["node_mapping"].item()
    M_index_mapping = npzfile["index_mapping"].item()
    M_emb = npzfile["embedding"]
    npzfile.close()

    npzfile = np.load(args.room_static, allow_pickle=True)
    R_node_mapping = npzfile["node_mapping"].item()
    R_index_mapping = npzfile["index_mapping"].item()
    R_emb = npzfile["embedding"]
    npzfile.close()

    #################
    # Preprocess bourgain Doc embeddings
    D_emb = minmaxnorm(D_emb)

    did_node_mapping = dict()
    for D, idx in D_node_mapping.items():
        did_node_mapping[str(D)] = idx

    # Add one more room (dummy)
    D_embedding_static = np.zeros((len(D2id)+1, D_emb.shape[1]))

    cnt_nonexistent_did = 0
    for did, idx_of_embedding_static in D2id.items():
        if did in did_node_mapping:
            idx_of_bourgain = did_node_mapping[did]
            D_embedding_static[idx_of_embedding_static] = D_emb[idx_of_bourgain]
        else:
            D_embedding_static[idx_of_embedding_static] = np.random.rand(D_emb.shape[1])
            cnt_nonexistent_did += 1
    D_embedding_static[-1] = np.random.rand(D_emb.shape[1]) # dummy item

    # D_emb = minmaxnorm(D_emb)
    #################
    # Preprocess bourgain Room embeddings
    R_emb = minmaxnorm(R_emb)

    rid_node_mapping = dict()
    for R, idx in R_node_mapping.items():
        rid_node_mapping[str(R)] = idx

    # Add one more room (dummy)
    R_embedding_static = np.zeros((len(R2id)+1, R_emb.shape[1]))

    cnt_nonexistent_rid = 0
    for rid, idx_of_embedding_static in R2id.items():
        if rid in rid_node_mapping:
            idx_of_bourgain = rid_node_mapping[rid]
            R_embedding_static[idx_of_embedding_static] = R_emb[idx_of_bourgain]
        else:
            R_embedding_static[idx_of_embedding_static] = np.random.rand(R_emb.shape[1])
            cnt_nonexistent_rid += 1
    R_embedding_static[-1] = np.random.rand(R_emb.shape[1]) # dummy item

    #################
    # Preprocess bourgain Med embeddings
    M_emb = minmaxnorm(M_emb)

    mid_node_mapping = dict()
    for M in M_node_mapping:
        if M.startswith("mid"):
            mid = M[4:] # mid_480480 -> 480480
            mid_node_mapping[mid] = M_node_mapping[M]
    M_embedding_static = np.zeros((len(M2id)+1, M_emb.shape[1]))

    cnt_nonexistent_mid = 0
    for mid, idx_of_embedding_static in M2id.items():
        if mid in mid_node_mapping:
            idx_of_bourgain = mid_node_mapping[mid]
            M_embedding_static[idx_of_embedding_static] = M_emb[idx_of_bourgain]
        else:
            M_embedding_static[idx_of_embedding_static] = np.random.rand(M_emb.shape[1])
            cnt_nonexistent_mid += 1
    M_embedding_static[-1] = np.random.rand(M_emb.shape[1]) # dummy item

    return D_embedding_static, M_embedding_static, R_embedding_static

def minmaxnorm(array):
    minimum = np.min(array)
    maximum = np.max(array)
    return (array - minimum) / (maximum - minimum)

# D2id: order of the index of doctors.
def item2id_by_entity(item2id, item2itemtype):
    item_array = list(item2id.keys())
    D2id = {}
    M2id = {}
    R2id = {}
    N2id = {}
    D_cnt, M_cnt, R_cnt, N_cnt= 0, 0, 0, 0
    for item in item_array:
        if item2itemtype[item] == 'D':
            D2id[item] = D_cnt
            D_cnt += 1
        elif item2itemtype[item] == 'M':
            M2id[item] = M_cnt
            M_cnt += 1
        elif item2itemtype[item] == 'R':
            R2id[item] = R_cnt
            R_cnt += 1
        elif item2itemtype[item] == 'N':
            N2id[item] = N_cnt
            N_cnt += 1
    return D2id, M2id, R2id, N2id
def my_filtering_function(pair):
    key, value = pair
    if value == 'N':
        return True  # keep pair in the filtered dictionary
    else:
        return False  # filter pair out of the dictionary
def item2id_by_entity1(item2id, item2itemtype):
    item_array = list(item2id.keys())
    D2id = {}
    M2id = {}
    R2id = {}
    N2id = {}
    D_cnt, M_cnt, R_cnt, N_cnt= 0, 0, 0, 0
    for item in item_array:
        if item2itemtype[item] == 'N':
            N2id[item] = N_cnt
            N_cnt += 1
    return D2id, M2id, R2id, N2id
def load_network_with_label(args, time_scaling=True):
    '''
    This function loads three sets of interaction, where the interactions are sorted by time

    Each line corresponds to one interaction (e.g., patient to doctor or patient to medication or patient to room), which corresponds to a timestamped edge
    Columns must be shaped as the following:
    ['patient', 'entity', 'time', 'y_lable (not used)', 'itemtype', 'pf_s1', 'pf_s2', 'pf_d1', ... 'pf_dx']
    Here, 'patient' is the patient id, 'entity' is the id of the doctor, medication, or room, 'time' is a integer value starting with 0 (initial interaction time), 'itemtype' is in {'D', 'M', 'R'} that denote the type of the entity (D:doctor, M:medication, R:room), 'pf_s1' and 'pf_s2' are two static features of the patient, and the rest are dynamic features of the patient
    '''

    network = args.network
    datapath = args.datapath

    user_sequence = []
    item_sequence = []
    itemtype_sequence = []
    static_feature_sequence = []
    dynamic_feature_sequence = []
    timestamp_sequence = []
    start_timestamp = None
    y_true_labels = [] # This is not used in the current setup

    print("\nLoading %s data from file: %s" % (network, datapath))
    f = open(datapath,"r")
    f.readline()
    idx_static_feature_start = 5
    idx_dynamic_feature_start = 5 + args.num_user_static_features
    for cnt, l in enumerate(f):
        ls = l.strip().split(",")
        user_sequence.append(ls[0])
        item_sequence.append(ls[1])
        # Using floating point timestamp causes problem when constructing cached objects!
        # if start_timestamp is None:
            # start_timestamp = float(ls[2])
        # timestamp_sequence.append(float(ls[2]) - start_timestamp) 
        # Use interger as timesteps
        if start_timestamp is None:
            start_timestamp = int(ls[2])
        timestamp_sequence.append(int(ls[2]) - start_timestamp) 
        y_true_labels.append(int(ls[3]))
        itemtype_sequence.append(str(ls[4])) 
        static_feature_sequence.append(list(map(float,ls[idx_static_feature_start: idx_dynamic_feature_start])))
        dynamic_feature_sequence.append(list(map(float,ls[idx_dynamic_feature_start:])))
    f.close()

    user_sequence = np.array(user_sequence) 
    item_sequence = np.array(item_sequence)
    timestamp_sequence = np.array(timestamp_sequence)

    print("Formating item sequence")
    nodeid = 0
    item2id = {}
    item2itemtype = {}

    item_timedifference_sequence = []
    item_current_timestamp = defaultdict(float)
    for cnt, (item, itemtype) in enumerate(zip(item_sequence, itemtype_sequence)):
        if item not in item2id:
            item2id[item] = nodeid
            item2itemtype[item] = itemtype
            nodeid += 1
        timestamp = timestamp_sequence[cnt]
        item_timedifference_sequence.append(timestamp - item_current_timestamp[item])
        item_current_timestamp[item] = timestamp
    num_items = len(item2id)

    ################
    D2id, M2id, R2id, N2id = item2id_by_entity(item2id, item2itemtype)
    num_D = len(D2id)
    num_M = len(M2id)
    num_R = len(R2id)
    num_N = len(N2id)
    # item_sequence_id = [item2id[item] for item in item_sequence]
    item_sequence_id = []
    for item, itemtype in zip(item_sequence, itemtype_sequence):
        #print(item)
        #print(itemtype)
        if itemtype=='D':
            item_sequence_id.append(D2id[item])
        elif itemtype=='M':
            item_sequence_id.append(M2id[item])
        elif itemtype=='R':
            item_sequence_id.append(R2id[item])
        elif itemtype=='N':
            item_sequence_id.append(N2id[item])
        
    # latest_itemtype = {'D': defaultdict(lambda: num_items), 'M': defaultdict(lambda: num_items), 'R': defaultdict(lambda: num_items)}

    print("Formating user sequence")
    nodeid = 0
    user2id = {}
    user_timedifference_sequence = []
    user_current_timestamp = defaultdict(float)
    user_previous_itemid_sequence = []
    user_latest_itemtype_itemid = {}
    for user in user_sequence:
        if user in user_latest_itemtype_itemid:
            pass
        else:
            user_latest_itemtype_itemid[user] = {'D':num_D, 'M':num_M, 'R':num_R, 'N':num_N}

    for cnt, user in enumerate(user_sequence):
        if user not in user2id:
            user2id[user] = nodeid
            nodeid += 1
        timestamp = timestamp_sequence[cnt]
        user_timedifference_sequence.append(timestamp - user_current_timestamp[user])
        user_current_timestamp[user] = timestamp

        current_item = item_sequence[cnt]
        current_item_type = itemtype_sequence[cnt]

        previous_itemid = user_latest_itemtype_itemid[user][current_item_type]
        user_previous_itemid_sequence.append(previous_itemid)

        # user_latest_itemtype_itemid[user][current_item_type] = item2id[current_item]
        if current_item_type == 'D':
            user_latest_itemtype_itemid[user][current_item_type] = D2id[current_item]
        elif current_item_type == 'M':
            user_latest_itemtype_itemid[user][current_item_type] = M2id[current_item]
        elif current_item_type == 'R':
            user_latest_itemtype_itemid[user][current_item_type] = R2id[current_item]
        elif current_item_type == 'N':
            user_latest_itemtype_itemid[user][current_item_type] = N2id[current_item]

    num_users = len(user2id)
    user_sequence_id = [user2id[user] for user in user_sequence]

    if time_scaling:
        print("Scaling timestamps")
        user_timedifference_sequence = scale(np.array(user_timedifference_sequence) + 1)
        item_timedifference_sequence = scale(np.array(item_timedifference_sequence) + 1)

    print("*** Network loading completed ***\n\n")

    return [user2id, user_sequence_id, user_timedifference_sequence, user_previous_itemid_sequence, \
        item2id, item_sequence_id, item_timedifference_sequence, \
        timestamp_sequence, \
        static_feature_sequence, dynamic_feature_sequence, \
        y_true_labels,
        item2itemtype, itemtype_sequence,
        D2id, M2id, R2id, N2id]

# LOAD PREVIOUSLY TRAINED AND SAVED MODEL
def load_model(model, optimizer, args, epoch):
    filename = "saved_models/%s/checkpoint.ep%d.pth.tar" % (args.trained_network, epoch)
    checkpoint = torch.load(filename)
    print("Loading saved embeddings and model: %s" % filename)
    args.start_epoch = checkpoint['epoch']
    user_embeddings = Variable(torch.from_numpy(checkpoint['user_embeddings']).cuda())
    item_embeddings = Variable(torch.from_numpy(checkpoint['item_embeddings']).cuda())
    try:
        train_end_idx = checkpoint['train_end_idx'] 
    except KeyError:
        train_end_idx = None

    try:
        user_embeddings_time_series = Variable(torch.from_numpy(checkpoint['user_embeddings_time_series']).cuda())
        item_embeddings_time_series = Variable(torch.from_numpy(checkpoint['item_embeddings_time_series']).cuda())
    except:
        user_embeddings_time_series = None
        item_embeddings_time_series = None

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return [model, optimizer, user_embeddings, item_embeddings, user_embeddings_time_series, item_embeddings_time_series, train_end_idx]

# SET USER AND ITEM EMBEDDINGS TO THE END OF THE TRAINING PERIOD 
def set_embeddings_training_end(user_embeddings, item_embeddings, user_embeddings_time_series, item_embeddings_time_series, user_data_id, item_data_id, train_end_idx):
    userid2lastidx = {}
    for cnt, userid in enumerate(user_data_id[:train_end_idx]):
        userid2lastidx[userid] = cnt
    itemid2lastidx = {}
    for cnt, itemid in enumerate(item_data_id[:train_end_idx]):
        itemid2lastidx[itemid] = cnt

    try:
        embedding_dim = user_embeddings_time_series.size(1)
    except:
        embedding_dim = user_embeddings_time_series.shape[1]
    for userid in userid2lastidx:
        user_embeddings[userid, :embedding_dim] = user_embeddings_time_series[userid2lastidx[userid]]
    for itemid in itemid2lastidx:
        item_embeddings[itemid, :embedding_dim] = item_embeddings_time_series[itemid2lastidx[itemid]]

    user_embeddings.detach_()
    item_embeddings.detach_()
def make_batch(sentences,hm,token_list,max_pred,maxlen,vocab_size,number_dict):
        batch = []
        positive = negative = 0
        i=0
        while i<=(len(sentences)-1):
        
            tokens_a_index, tokens_b_index= i, randrange(len(sentences))
            i=i+1
            tokens_a, tokens_b= token_list[tokens_a_index], token_list[tokens_b_index]

            input_ids = [hm['[CLS]']] + tokens_a + [hm['[SEP]']]

            segment_ids = [0] * (1 + len(tokens_a) + 1)

            #MASK LM
            n_pred =  min(max_pred, max(1, int(round(len(input_ids) * 0.25)))) # 25% of tokens in one sentence

            cand_maked_pos = [i for i, token in enumerate(input_ids)
                            if token != hm['[CLS]'] and token !=hm['[SEP]']]
            shuffle(cand_maked_pos)
            masked_tokens, masked_pos = [], []
            for pos in cand_maked_pos[:n_pred]:
                masked_pos.append(pos)
                masked_tokens.append(input_ids[pos])
                if random() < 0.8:  # 80%
                    input_ids[pos] = hm['[MASK]'] # make mask
                elif random() < 0.5:  # 10%
                    index = randint(0, vocab_size - 1) # random index in vocabulary
                    while index < 4: # cause {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3} are all  meanless
                        index = randint(0, vocab_size - 1)
                    input_ids[pos] = hm[number_dict[index]] # replace

            # Zero Paddings
            n_pad = maxlen - len(input_ids)
            input_ids.extend([0] * n_pad)
            segment_ids.extend([0] * n_pad)

            # Zero Padding (100% - 15%) tokens
            if max_pred > n_pred:
                n_pad = max_pred - n_pred
                masked_tokens.extend([0] * n_pad)
                masked_pos.extend([0] * n_pad)

        
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos]) # IsNext
        
        return batch

def load_network_with_labels(args,item2id,D2id,M2id,R2id,N2id,user2id, time_scaling=True):
    '''
    This function loads three sets of interaction, where the interactions are sorted by time

    Each line corresponds to one interaction (e.g., patient to doctor or patient to medication or patient to room), which corresponds to a timestamped edge
    Columns must be shaped as the following:
    ['patient', 'entity', 'time', 'y_lable (not used)', 'itemtype', 'pf_s1', 'pf_s2', 'pf_d1', ... 'pf_dx']
    Here, 'patient' is the patient id, 'entity' is the id of the doctor, medication, or room, 'time' is a integer value starting with 0 (initial interaction time), 'itemtype' is in {'D', 'M', 'R'} that denote the type of the entity (D:doctor, M:medication, R:room), 'pf_s1' and 'pf_s2' are two static features of the patient, and the rest are dynamic features of the patient
    '''

    network = args.network
    datapath = args.datapath

    user_sequence = []
    item_sequence = []
    itemtype_sequence = []
    static_feature_sequence = []
    dynamic_feature_sequence = []
    timestamp_sequence = []
    start_timestamp = None
    y_true_labels = [] # This is not used in the current setup

    print("\nLoading %s data from file: %s" % (network, datapath))
    f = open(datapath,"r")
    f.readline()
    idx_static_feature_start = 5
    idx_dynamic_feature_start = 5 + args.num_user_static_features
    for cnt, l in enumerate(f):
        ls = l.strip().split(",")
        user_sequence.append(ls[0])
        item_sequence.append(ls[1])
        # Using floating point timestamp causes problem when constructing cached objects!
        # if start_timestamp is None:
            # start_timestamp = float(ls[2])
        # timestamp_sequence.append(float(ls[2]) - start_timestamp) 
        # Use interger as timesteps
        if start_timestamp is None:
            start_timestamp = int(ls[2])
        timestamp_sequence.append(int(ls[2]) - start_timestamp) 
        y_true_labels.append(int(ls[3]))
        itemtype_sequence.append(str(ls[4])) 
        static_feature_sequence.append(list(map(float,ls[idx_static_feature_start: idx_dynamic_feature_start])))
        dynamic_feature_sequence.append(list(map(float,ls[idx_dynamic_feature_start:])))
    f.close()

    user_sequence = np.array(user_sequence) 
    item_sequence = np.array(item_sequence)
    timestamp_sequence = np.array(timestamp_sequence)

    print("Formating item sequence")
    nodeid = 0
    #item2id = {}
    item2itemtype = {}

    item_timedifference_sequence = []
    item_current_timestamp = defaultdict(float)
    for cnt, (item, itemtype) in enumerate(zip(item_sequence, itemtype_sequence)):
        if item not in item2id:
            item2id[item] = nodeid
            item2itemtype[item] = itemtype
            nodeid += 1
        timestamp = timestamp_sequence[cnt]
        item_timedifference_sequence.append(timestamp - item_current_timestamp[item])
        item_current_timestamp[item] = timestamp
    num_items = len(item2id)

    ################
    #_,_,_,N2id = item2id_by_entity1(item2id, item2itemtype)
    num_D = len(D2id)
    num_M = len(M2id)
    num_R = len(R2id)
    num_N = len(N2id)
    maxval=max(list(N2id.values()))
    # item_sequence_id = [item2id[item] for item in item_sequence]
    item_sequence_id = []
    for item, itemtype,timestamp in zip(item_sequence, itemtype_sequence,timestamp_sequence):
        if itemtype=='D':
            item_sequence_id.append(D2id[item])
        elif itemtype=='M':
            item_sequence_id.append(M2id[item])
        elif itemtype=='R':
            item_sequence_id.append(R2id[item])
        elif itemtype=='N':
            #if item in N2id.keys():
            item_sequence_id.append(N2id[item])
            #else:
                #N2id[item]=maxval
                #maxval=maxval+1
                #sitem_sequence_id.append(N2id[item])

        
    # latest_itemtype = {'D': defaultdict(lambda: num_items), 'M': defaultdict(lambda: num_items), 'R': defaultdict(lambda: num_items)}

    print("Formating user sequence")
    nodeid = 0
    #user2id = {}
    user_timedifference_sequence = []
    user_current_timestamp = defaultdict(float)
    user_previous_itemid_sequence = []
    user_latest_itemtype_itemid = {}
    for user in user_sequence:
        if user in user_latest_itemtype_itemid:
            pass
        else:
            user_latest_itemtype_itemid[user] = {'D':num_D, 'M':num_M, 'R':num_R,'N':num_N}

    for cnt, user in enumerate(user_sequence):
        #if user not in user2id:
            #user2id[user] = nodeid
            #nodeid += 1
        timestamp = timestamp_sequence[cnt]
        user_timedifference_sequence.append(timestamp - user_current_timestamp[user])
        user_current_timestamp[user] = timestamp

        current_item = item_sequence[cnt]
        current_item_type = itemtype_sequence[cnt]

        previous_itemid = user_latest_itemtype_itemid[user][current_item_type]
        user_previous_itemid_sequence.append(previous_itemid)

        # user_latest_itemtype_itemid[user][current_item_type] = item2id[current_item]
        if current_item_type == 'D':
            user_latest_itemtype_itemid[user][current_item_type] = D2id[current_item]
        elif current_item_type == 'M':
            user_latest_itemtype_itemid[user][current_item_type] = M2id[current_item]
        elif current_item_type == 'R':
            user_latest_itemtype_itemid[user][current_item_type] = R2id[current_item]
        elif current_item_type == 'N':
            user_latest_itemtype_itemid[user][current_item_type] = N2id[current_item]

    num_users = len(user2id)
    user_sequence_id = [user2id[user] for user in user_sequence]

    if time_scaling:
        print("Scaling timestamps")
        user_timedifference_sequence = scale(np.array(user_timedifference_sequence) + 1)
        item_timedifference_sequence = scale(np.array(item_timedifference_sequence) + 1)

    print("*** Network loading completed ***\n\n")

    return [user2id, user_sequence_id, user_timedifference_sequence, user_previous_itemid_sequence, \
        item2id, item_sequence_id, item_timedifference_sequence, \
        timestamp_sequence, \
        static_feature_sequence, dynamic_feature_sequence, \
        y_true_labels,
        item2itemtype, itemtype_sequence,
        D2id, M2id, R2id,N2id]