import numpy as np
import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import os
import sys
import torch
from torch.autograd import Variable
from torch.nn import Sequential
from torch import nn
import torchvision.models as models





# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/model.py#L10
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/model.py#L82
class NNBase(nn.Module):

    def __init__(self, recurrent, recurrent_input_size, hidden_size):

        super(NNBase, self).__init__()
        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRUCell(recurrent_input_size, hidden_size)
            nn.init.orthogonal_(self.gru.weight_ih.data)
            nn.init.orthogonal_(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def rec_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x = hxs = self.gru(x, hxs * masks[:, None])
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N, 1)

            outputs = []
            for i in range(T):
                hx = hxs = self.gru(x[i], hxs * masks[i])
                outputs.append(hx)

            # x is a (T, N, -1) tensor
            x = torch.stack(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)

        return x, hxs

class PoseEstimationModel_baseline_NoMask_maskAzChannel_MaskOut(torch.nn.Module):
  def __init__(self, in_channels , num_bins_az , mask_size , num_bins_el, flag = None ):
    super().__init__()

    self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2)

    self.Rel = nn.ReLU()
    self.flat = nn.Flatten()
    self.fc1 = nn.Linear(32 * 7 * 7, 512)
    self.fc2 = nn.Linear(512, 128)
    self.fc3 = nn.Linear(128, num_bins_az)
    self.fc4 = nn.Linear(128, num_bins_el)
    self.flag = flag

  def forward(self, x, mask):

    if (self.flag == 1):
        x = torch.cat((x, mask),dim = 1)

    elif(self.flag == 2):
        x = mask.T*x.T
        x = x.T

    x = self.conv1(x)
    x = self.Rel(x)
    x = self.flat(x)
    x = self.fc1(x)
    x = self.fc2(x)
    x_az = self.fc3(x)
    x_el = self.fc4(x)

    return x_az, x_el

 


  # self.main = nn.Sequential(
  #       nn.Conv2d(3, 5, 7, stride=1, padding=1),
  #       # nn.ReLU(),
  #       nn.BatchNorm2d(5),
  #       # nn.MaxPool2d(2),
  #       nn.Conv2d(5, 8, 3, stride=1, padding=1),
  #       # nn.ReLU(),
  #       nn.BatchNorm2d(8),
  #       # nn.MaxPool2d(2),
  #       nn.Conv2d(8, 16, 3, stride=1, padding=1),
  #       # nn.ReLU(),
  #       nn.BatchNorm2d(16),
  #       # nn.MaxPool2d(2),
  #       # nn.Conv2d(128, 64, 3, stride=1, padding=1),
  #       # nn.MaxPool2d(2),
  #       # nn.BatchNorm2d()
  #       # # nn.ReLU(),
  #       # nn.Conv2d(64, 32, 3, stride=1, padding=1),
  #       # nn.MaxPool2d(2),
  #       # nn.BatchNorm2d()
  #       # nn.ReLU(),
  #       Flatten()
  #   )


class policy_model(torch.nn.Module):
  def __init__(self):
    super().__init__()

    out_size = int(20 / 16.) * int(20 / 16.)
    hidden_size =10
    num_sem_categories = 40

    self.main = nn.Sequential(
        nn.Conv2d(3, 8, 3, stride=2, padding=1),
        # nn.ReLU(),
        nn.BatchNorm2d(8),
        nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, stride=2, padding=1),
        # # nn.ReLU(),
        nn.BatchNorm2d(16),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, stride=2, padding=1),
        # nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 32, 1, stride=1, padding=1),
        # nn.MaxPool2d(2),
        nn.BatchNorm2d(32),
        nn.MaxPool2d(2),

        # # nn.ReLU(),
        # nn.Conv2d(64, 32, 3, stride=1, padding=1),
        # nn.MaxPool2d(2),
        # nn.BatchNorm2d()
        # nn.ReLU(),
        Flatten()
    )
    
    self.dp = nn.Dropout(p=0.5)
    self.classifier_fc1 = nn.Linear(256,64)
    self.classifier_fc2 = nn.Linear(64,5)

    self.linear_cnn = nn.Linear(2048, 128)
    self.rnn = nn.GRU(256, 128, 1 ,  bidirectional=True)
    self.hidden = self.init_hidden_gru()
    self.last_input = None

    self.goal_emb = nn.Linear(40, 128)

    self.src_emb = nn.Linear(2, 128)
    self.sf = nn.Softmax()

  def init_hidden_gru(self, bs=64):
    return torch.randn(2, 128, device = "cuda:0")

  def init_input_images(self, mini_batch ,im_size):
    self.last_input = torch.zeros(( mini_batch ,3,im_size,im_size),device = "cuda:0",dtype= torch.float32)


  def forward(self, inputs, src_point, goal_point , goal_cat , itr, shortest_path, dim = 30, mini_batch = 10, h0= None, last_batch= False):    
   
    # print(inputs.shape)
    # print(shortest_path)
    input_patches =  self.last_input[-1,:,:,:].repeat(mini_batch,1,1,1)

    for patch in range(mini_batch):

        # print(shortest_path.shape, "itr*mini_batch", itr, mini_batch, patch, "__", 0,itr*mini_batch+patch)
        if last_batch:
            current_loc = shortest_path[0, -mini_batch + patch]

        else:
            current_loc = shortest_path[0,itr*mini_batch + patch]

        # print(inputs.shape)
        input_patches[patch:,:,max(0,current_loc[0]-dim):min(current_loc[0]+dim, inputs.shape[2]-1),
                                max(0,current_loc[1]-dim):min(current_loc[1]+dim, inputs.shape[3]-1)] = inputs[0,:,max(0,current_loc[0]-dim):min(current_loc[0]+dim, inputs.shape[2]-1) , 
                                            max(0,current_loc[1]-dim):min(current_loc[1]+dim, inputs.shape[3]-1)].repeat(mini_batch-patch,1,1,1)

    # print(input_patches.shape)
    x = self.linear_cnn(self.main(input_patches))
    # print(x.shape)

    if last_batch:
        current_loc = shortest_path[0,-mini_batch :]
    else:
        current_loc = shortest_path[0,itr*mini_batch :itr*mini_batch +mini_batch,:]

    #print(current_loc)
    goal_emb = self.goal_emb(goal_cat).repeat(mini_batch, 1)
    src_emb = self.src_emb(current_loc.float())
    # print("goal embd", goal_emb.shape)
 

    #x = torch.cat((x, goal_emb,src_emb), 1)
    x = torch.cat((x, goal_emb), 1)

    # print(x.shape)

    # x = nn.ReLU()(self.linear1(x))


    x, self.hidden = self.rnn(x, h0)

    x = self.classifier_fc2(self.dp(self.classifier_fc1(x)))
    # print(x.shape)
    self.last_input = input_patches

    return x
