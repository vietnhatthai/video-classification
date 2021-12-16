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


class Predictor(nn.Module):
    def __init__(self, action_name_path, model_path):
        super(Predictor, self).__init__()

        self.action_name_path = action_name_path
        self.model_path = model_path

        with open(action_name_path, 'rb') as f:
            action_names = pickle.load(f)   # load UCF101 actions names

        # convert labels -> category
        self.le = LabelEncoder()
        self.le.fit(action_names)

        # show how many classes there are
        #list(self.le.classes_)

        # use same encoder CNN saved!
        self.CNN_fc_hidden1, self.CNN_fc_hidden2 = 1024, 768
        self.CNN_embed_dim = 512   # latent dim extracted by 2D CNN
        self.res_size = 224        # ResNet image size
        self.dropout_p = 0.0       # dropout probability

        # use same decoder RNN saved!
        self.RNN_hidden_layers = 3
        self.RNN_hidden_nodes = 512
        self.RNN_FC_dim = 256

        # training parameters
        self.k = 101             # number of target category

        # data loading parameters
        self.use_cuda = torch.cuda.is_available()                   # check if GPU exists
        self.device = torch.device("cuda" if self.use_cuda else "cpu")   # use CPU or GPU

        self.transform = transforms.Compose([transforms.Resize([self.res_size, self.res_size]),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        # reload CRNN model
        self.cnn_encoder = ResCNNEncoder(fc_hidden1=self.CNN_fc_hidden1, fc_hidden2=self.CNN_fc_hidden2,
                                        drop_p=self.dropout_p, CNN_embed_dim=self.CNN_embed_dim).to(self.device)
        self.rnn_decoder = DecoderRNN(CNN_embed_dim=self.CNN_embed_dim, h_RNN_layers=self.RNN_hidden_layers,
                                        h_RNN=self.RNN_hidden_nodes, 
                                        h_FC_dim=self.RNN_FC_dim, drop_p=self.dropout_p, num_classes=self.k).to(self.device)

        self.cnn_encoder.load_state_dict(torch.load(os.path.join(self.model_path, 'cnn_encoder_epoch120.pth')))
        self.rnn_decoder.load_state_dict(torch.load(os.path.join(self.model_path, 'rnn_decoder_epoch120.pth')))
        #print('CRNN model reloaded!')
        self.cnn_encoder.eval()
        self.rnn_decoder.eval()

    def forward(self, X):
        with torch.no_grad():
            # distribute data to device
            X = X.to(self.device)
            output = self.rnn_decoder(self.cnn_encoder(X))
            y_pred = output.max(1, keepdim=True)[1]  # location of max log-probability as prediction
            y_pred = y_pred.cpu().data.squeeze().numpy().tolist()
            return self.le.classes_[y_pred]


if __name__ == '__main__':
    res_size = 224 
    transform = transforms.Compose([transforms.Resize([res_size, res_size]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # set path
    action_name_path = "./UCF101actions.pkl"
    save_model_path = "./ResNetCRNN_ckpt/"

    model = Predictor(action_name_path, save_model_path)

    X = []

    frames = range(1, 75)

    path = 'D:\\Projects\\video-classification\ResNetCRNN\\v_BaseballPitch_g04_c02'

    for i in frames:
        image = Image.open(os.path.join(path, 'frame{:06d}.jpg'.format(i)))
        image = transform(image)

        X.append(image)

    X = torch.stack(X, dim=0)
    X = X.reshape((-1, 74, 3, 224, 224))

    Y = model(X)

    print(Y)