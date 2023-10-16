'''
Author: -
Email: -
Last Modified: Oct 2021

This script contains the class for the deep learning model
and the functions to save the learned model and mappings
'''

from __future__ import division
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import math
from collections import defaultdict
import os
import pickle
import numpy as np

PATH = "./"

total_reinitialization_count = 0

# Normalization layer
class NormalLinear(nn.Linear):
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
            self.bias.data.normal_(0, stdv)

# DECENT class
class DECENT(nn.Module):
    def __init__(self, args, num_static_features, num_dynamic_features, num_users, num_items, num_D, num_M, num_R):
        super(DECENT,self).__init__()
        #self.BERT=BERT_model
        print("Initialize the DECENT model")
        self.embedding_dim = args.embedding_dim
        self.num_users = num_users
        self.num_items = num_items
        self.user_static_embedding_size = num_users
        # self.item_static_embedding_size = num_items
        self.D_static_embedding_size = num_D
        self.M_static_embedding_size = num_M
        self.R_static_embedding_size = num_R
        #self.N_static_embedding_size = num_N
        self.num_static_features = num_static_features
        self.num_dynamic_features = num_dynamic_features

        nn_input_size_users = 2 * self.embedding_dim + 1 + num_static_features + num_dynamic_features
        nn_input_size_items = 2 * self.embedding_dim + 1 + num_static_features + num_dynamic_features

        print("Initializing user and item update neural networks")
        self.itemD_nn = nn.Linear(nn_input_size_users, self.embedding_dim)
        self.itemM_nn = nn.Linear(nn_input_size_users, self.embedding_dim)
        self.itemR_nn = nn.Linear(nn_input_size_users, self.embedding_dim)
        self.userD_nn = nn.Linear(nn_input_size_items, self.embedding_dim)
        self.userM_nn = nn.Linear(nn_input_size_items, self.embedding_dim)
        self.userR_nn = nn.Linear(nn_input_size_items, self.embedding_dim)
        self.userN_nn = nn.Linear(nn_input_size_items, self.embedding_dim)

        self.predictionD_layer = nn.Linear( \
                in_features = self.num_static_features + self.user_static_embedding_size + self.D_static_embedding_size + self.embedding_dim * 2, \
                out_features = self.D_static_embedding_size + self.embedding_dim)

        self.predictionM_layer = nn.Linear( \
                in_features = self.num_static_features + self.user_static_embedding_size + self.M_static_embedding_size + self.embedding_dim * 2, \
                out_features = self.M_static_embedding_size + self.embedding_dim)

        self.predictionR_layer = nn.Linear( \
                in_features = self.num_static_features + self.user_static_embedding_size + self.R_static_embedding_size + self.embedding_dim * 2, \
                out_features = self.R_static_embedding_size + self.embedding_dim)
        
        #self.predictionN_layer = nn.Linear( \
        #        in_features = self.num_static_features + self.user_static_embedding_size + self.N_static_embedding_size + self.embedding_dim * 2, \
        #        out_features = self.N_static_embedding_size + self.embedding_dim)

        self.embedding_layer = NormalLinear(1, self.embedding_dim)
        print("DECENT initialization complete\n")
    
    def forward(self, user_embeddings, item_embeddings, timediffs=None, static_features=None, dynamic_features=None, select=None):
        if select == 'itemD_update':
            inputD = torch.cat([user_embeddings, item_embeddings, timediffs, static_features, dynamic_features], dim=1)
            item_embedding_output = self.itemD_nn(inputD)
            return F.normalize(nn.Tanh()(item_embedding_output))
        elif select == 'itemM_update':
            inputM = torch.cat([user_embeddings, item_embeddings, timediffs, static_features, dynamic_features], dim=1)
            item_embedding_output = self.itemM_nn(inputM)
            return F.normalize(nn.Tanh()(item_embedding_output))
        elif select == 'itemR_update':
            inputR = torch.cat([user_embeddings, item_embeddings, timediffs, static_features, dynamic_features], dim=1)
            item_embedding_output = self.itemR_nn(inputR)
            return F.normalize(nn.Tanh()(item_embedding_output))
        

        elif select == 'userD_update':
            inputD = torch.cat([user_embeddings, item_embeddings, timediffs, static_features, dynamic_features], dim=1)
            user_embedding_output = self.userD_nn(inputD)
            return F.normalize(nn.Tanh()(user_embedding_output))
        elif select == 'userM_update':
            inputM = torch.cat([user_embeddings, item_embeddings, timediffs, static_features, dynamic_features], dim=1)
            user_embedding_output = self.userM_nn(inputM)
            return F.normalize(nn.Tanh()(user_embedding_output))
        elif select == 'userR_update':
            inputR = torch.cat([user_embeddings, item_embeddings, timediffs, static_features, dynamic_features], dim=1)
            user_embedding_output = self.userR_nn(inputR)
            return F.normalize(nn.Tanh()(user_embedding_output))

        elif select == 'project':
            user_projected_embedding = self.context_convert(user_embeddings, timediffs)
            return user_projected_embedding    
    def forward1(self, user_embeddings, item_embeddings, timediffs=None, static_features=None, dynamic_features=None, select=None):
        if select == 'itemD_update':
            inputD = torch.cat([user_embeddings, item_embeddings, timediffs, static_features, dynamic_features], dim=1)
            item_embedding_output = self.itemD_nn(inputD)
            return F.normalize(nn.Tanh()(item_embedding_output))
        elif select == 'itemM_update':
            inputM = torch.cat([user_embeddings, item_embeddings, timediffs, static_features, dynamic_features], dim=1)
            item_embedding_output = self.itemM_nn(inputM)
            return F.normalize(nn.Tanh()(item_embedding_output))
        elif select == 'itemR_update':
            inputR = torch.cat([user_embeddings, item_embeddings, timediffs, static_features, dynamic_features], dim=1)
            item_embedding_output = self.itemR_nn(inputR)
            return F.normalize(nn.Tanh()(item_embedding_output))
        #elif select == 'itemN_update':
            #return predi

        elif select == 'userD_update':
            inputD = torch.cat([user_embeddings, item_embeddings, timediffs, static_features, dynamic_features], dim=1)
            user_embedding_output = self.userD_nn(inputD)
            return F.normalize(nn.Tanh()(user_embedding_output))
        elif select == 'userM_update':
            inputM = torch.cat([user_embeddings, item_embeddings, timediffs, static_features, dynamic_features], dim=1)
            user_embedding_output = self.userM_nn(inputM)
            return F.normalize(nn.Tanh()(user_embedding_output))
        elif select == 'userR_update':
            inputR = torch.cat([user_embeddings, item_embeddings, timediffs, static_features, dynamic_features], dim=1)
            user_embedding_output = self.userR_nn(inputR)
            return F.normalize(nn.Tanh()(user_embedding_output))
        elif select == 'userN_update':
            inputN = torch.cat([user_embeddings, item_embeddings, timediffs, static_features, dynamic_features], dim=1)
            user_embedding_output = self.userN_nn(inputN)
            return F.normalize(nn.Tanh()(user_embedding_output))

        elif select == 'project':
            user_projected_embedding = self.context_convert(user_embeddings, timediffs)
            return user_projected_embedding

    def context_convert(self, embeddings, timediffs):
        new_embeddings = embeddings * (1 + self.embedding_layer(timediffs))
        return new_embeddings

    def predict_item_embedding1(self, user_item_embedding, itemtype):
        if itemtype == 'D':
            X_out = self.predictionD_layer(user_item_embedding)
        elif itemtype == 'M':
            X_out = self.predictionM_layer(user_item_embedding)
        elif itemtype == 'R':
            X_out = self.predictionR_layer(user_item_embedding)
        #elif itemtype == 'N':
            #X_out = predi
        return X_out

    def predict_item_embedding(self, user_item_embedding, itemtype):
        if itemtype == 'D':
            X_out = self.predictionD_layer(user_item_embedding)
        elif itemtype == 'M':
            X_out = self.predictionM_layer(user_item_embedding)
        elif itemtype == 'R':
            X_out = self.predictionR_layer(user_item_embedding)
        return X_out

# Re-Initialize dictionaries
def reinitialize_tbatches():
    global current_tbatches_interactionids, current_tbatches_user, current_tbatches_item, current_tbatches_timestamp, current_tbatches_static_feature, current_tbatches_dynamic_feature, current_tbatches_label, current_tbatches_previous_item, current_tbatches_itemtype
    global tbatchid_user, tbatchid_item, current_tbatches_user_timediffs, current_tbatches_item_timediffs, current_tbatches_user_timediffs_next
    global DECEnt_tbatches_interactionids, DECEnt_tbatches_user, DECEnt_tbatches_item, DECEnt_tbatches_timestamp, DECEnt_tbatches_static_feature, DECEnt_tbatches_dynamic_feature, DECEnt_tbatches_label, DECEnt_tbatches_previous_item, DECEnt_tbatches_itemtype
    global DECEnt_tbatches_user_timediffs, DECEnt_tbatches_item_timediffs, DECEnt_tbatches_user_timediffs_next

    # list of users of each tbatch up to now
    current_tbatches_interactionids = defaultdict(list)
    current_tbatches_user = defaultdict(list)
    current_tbatches_item = defaultdict(list)
    current_tbatches_itemtype = defaultdict(list)
    current_tbatches_timestamp = defaultdict(list)
    current_tbatches_static_feature = defaultdict(list)
    current_tbatches_dynamic_feature = defaultdict(list)
    current_tbatches_label = defaultdict(list)
    current_tbatches_previous_item = defaultdict(list)
    current_tbatches_user_timediffs = defaultdict(list)
    current_tbatches_item_timediffs = defaultdict(list)
    current_tbatches_user_timediffs_next = defaultdict(list)

    DECEnt_tbatches_interactionids = defaultdict(list)
    DECEnt_tbatches_user = defaultdict(list)
    DECEnt_tbatches_item = defaultdict(list)
    DECEnt_tbatches_itemtype = defaultdict(list)
    DECEnt_tbatches_timestamp = defaultdict(list)
    DECEnt_tbatches_static_feature = defaultdict(list)
    DECEnt_tbatches_dynamic_feature = defaultdict(list)
    DECEnt_tbatches_label = defaultdict(list)
    DECEnt_tbatches_previous_item = defaultdict(list)
    DECEnt_tbatches_user_timediffs = defaultdict(list)
    DECEnt_tbatches_item_timediffs = defaultdict(list)
    DECEnt_tbatches_user_timediffs_next = defaultdict(list)

    # the latest tbatch a user is in
    tbatchid_user = defaultdict(lambda: -1)

    # the latest tbatch a item is in
    tbatchid_item = defaultdict(lambda: -1)

    global total_reinitialization_count
    total_reinitialization_count +=1

# Save the model
def save_model(model, optimizer, args, epoch, user_embeddings, item_embeddings, train_end_idx, user_embeddings_time_series=None, item_embeddings_time_series=None, path=PATH):
    print("*** Saving embeddings and model ***")
    state = {
            'user_embeddings': user_embeddings.data.cpu().numpy(),
            'item_embeddings': item_embeddings.data.cpu().numpy(),
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'train_end_idx': train_end_idx
            }
    #args.network='DEC1'
    if user_embeddings_time_series is not None:
        state['user_embeddings_time_series'] = user_embeddings_time_series.data.cpu().numpy()
        state['item_embeddings_time_series'] = item_embeddings_time_series.data.cpu().numpy()

    directory = os.path.join(path, 'saved_models/{}'.format(args.network))
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = os.path.join(directory, "checkpoint.ep{}.pth.tar".format(epoch))
    torch.save(state, filename)
    print("*** Saved embeddings and model to file: {} ***\n\n".format(filename))

# Save the mappings for purpose of later evaluations
def save_mappings(args, user2id, item2id, item2itemtype):
    with open("pickle/{}_user2id.pickle".format(args.network), "wb") as handle:
        pickle.dump(user2id, handle)
    with open("pickle/{}_item2id.pickle".format(args.network), "wb") as handle:
        pickle.dump(item2id, handle)
    with open("pickle/{}_item2itemtype.pickle".format(args.network), "wb") as handle:
        pickle.dump(item2itemtype, handle)

def save_loss_arrays(args, loss_per_timestep, prediction_loss_per_timestep, user_update_loss_per_timestep, item_update_loss_per_timestep, D_loss_per_timestep, M_loss_per_timestep, R_loss_per_timestep,llm_loss,clos):
    np.savez("loss/loss1_{}".format(args.network), \
            loss_per_timestep = loss_per_timestep, \
            prediction_loss_per_timestep = prediction_loss_per_timestep, \
            user_update_loss_per_timestep = user_update_loss_per_timestep, \
            item_update_loss_per_timestep = item_update_loss_per_timestep, \
            D_loss_per_timestep = D_loss_per_timestep, \
            M_loss_per_timestep = M_loss_per_timestep, \
            R_loss_per_timestep = R_loss_per_timestep, \
            llm=llm_loss, \
            CL_loss=clos
            
            )
