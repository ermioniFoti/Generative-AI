import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils import weight_norm as weight_normal                                                                                                            


class Decoder(nn.Module):
    def __init__(
        self,
        dims,
        dropout=None,
        dropout_prob=0.1,
        norm_layers=(),
        latent_in=(),
        weight_norm=True,   
        use_tanh=True
    ):
        super(Decoder, self).__init__()
        
        ##########################################################
        # <================START MODIFYING CODE<================>
        ##########################################################
        # **** YOU SHOULD IMPLEMENT THE MODEL ARCHITECTURE HERE ****
        # Define the network architecture based on the figure shown in the assignment page.
        # Read the instruction carefully for layer details.
        # Pay attention that your implementation should include FC layers, weight_norm layers,
        # Leaky ReLU layers, Dropout layers and a tanh layer.
        # self.dropout_prob = dropout_prob
        self.dropout_prob = dropout_prob
        
        self.fc1 = weight_normal(nn.Linear(3,512))
        self.fc2 = weight_normal(nn.Linear(512,512))
        self.fc3 = weight_normal(nn.Linear(512,509))
        self.fc4 = weight_normal(nn.Linear(512,512))
        self.fc5 = weight_normal(nn.Linear(512,512))
        self.fc6 = weight_normal(nn.Linear(512,512))
        self.fc7 = weight_normal(nn.Linear(512,512))
        self.fc_final = nn.Linear(512,1)
        self.th = nn.Tanh()
        self.activation_layer = nn.LeakyReLU(0.01) # Leaky ReLU with negative slope = 0.01
        self.dropout_layer = nn.Dropout(self.dropout_prob)

        # ***********************************************************************
        ##########################################################
        # <================END MODIFYING CODE<================>
        ##########################################################
    
    # input: N x 3
    def forward(self, input):

        ##########################################################
        # <================START MODIFYING CODE<================>
        ##########################################################
        # **** YOU SHOULD IMPLEMENT THE FORWARD PASS HERE ****
        # Based on the architecture defined above, implement the feed forward procedure

        x = self.dropout_layer(self.activation_layer(self.fc1(input)))
        x = self.dropout_layer(self.activation_layer(self.fc2(x)))
        x = self.dropout_layer(self.activation_layer(self.fc3(x)))
        
        # Concatenates the given sequence of tensors (x,input) along dimension = 1 
        # x = [N,509] + input = [N,3] -> [N,512]
        x = torch.cat((x,input),1) 

        x = self.dropout_layer(self.activation_layer(self.fc4(x)))
        x = self.dropout_layer(self.activation_layer(self.fc5(x)))
        x = self.dropout_layer(self.activation_layer(self.fc6(x)))
        x = self.dropout_layer(self.activation_layer(self.fc7(x)))
        x = self.fc_final(x)
        x = self.th(x)

        # ***********************************************************************
        ##########################################################  
        # <================END MODIFYING CODE<================>
        ##########################################################

        return x
