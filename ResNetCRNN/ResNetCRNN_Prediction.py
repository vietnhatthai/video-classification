import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import matplotlib.pyplot as plt
from functions import *
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle

# set path
action_name_path = "./UCF101actions.pkl"
save_model_path = "./ResNetCRNN_ckpt/"

# use same encoder CNN saved!
CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
CNN_embed_dim = 512   # latent dim extracted by 2D CNN
res_size = 224        # ResNet image size
dropout_p = 0.0       # dropout probability

# use same decoder RNN saved!
RNN_hidden_layers = 3
RNN_hidden_nodes = 512
RNN_FC_dim = 256

# training parameters
k = 101             # number of target category

# data loading parameters
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

transform = transforms.Compose([transforms.Resize([res_size, res_size]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# reload CRNN model
cnn_encoder = ResCNNEncoder(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device)
rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes, 
                         h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=k).to(device)

cnn_encoder.load_state_dict(torch.load(os.path.join(save_model_path, 'cnn_encoder_epoch120.pth')))
rnn_decoder.load_state_dict(torch.load(os.path.join(save_model_path, 'rnn_decoder_epoch120.pth')))
print('CRNN model reloaded!')

cnn_encoder.eval()
rnn_decoder.eval()

with open(action_name_path, 'rb') as f:
    action_names = pickle.load(f)   # load UCF101 actions names

# convert labels -> category
le = LabelEncoder()
le.fit(action_names)

# show how many classes there are
list(le.classes_)

X = []

frames = range(1, 75)

path = 'D:\\Projects\\video-classification\ResNetCRNN\\v_BaseballPitch_g04_c02'

for i in frames:
    image = Image.open(os.path.join(path, 'frame{:06d}.jpg'.format(i)))
    image = transform(image)

    X.append(image)

X = torch.stack(X, dim=0)
X = X.reshape((-1, 74, 3, 224, 224))

with torch.no_grad():
    # distribute data to device
    X = X.to(device)
    output = rnn_decoder(cnn_encoder(X))
    y_pred = output.max(1, keepdim=True)[1]  # location of max log-probability as prediction
    y_pred = y_pred.cpu().data.squeeze().numpy().tolist()

print(le.classes_[y_pred])