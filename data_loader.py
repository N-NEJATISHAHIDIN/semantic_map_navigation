import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms.functional as TF
import subprocess
from PIL import Image, ImageOps
import networkx as nx


class semantic_navigation_dl(torch.utils.data.Dataset):
    
    def __init__(self, episods ):

        self.episods = episods
        self.cats = {"sofa" : 10, "chair": 3, "table":5, "bed": 11}

        # self.transform = transforms.Compose([
        # transforms.ToTensor(),
        # transforms.Normalize()
        #     ])

    def __len__(self):
        
        return len(self.episods)

    def convert_to_actions(self, shortest_path):

        diff = shortest_path[1:,:]-shortest_path[:-1,:]
        semi_label = (diff+2)[:,0]+3*(diff+2)[:,1]
        label = np.select([semi_label == 9, semi_label == 5,semi_label == 11, semi_label == 7], [0, 1, 2, 3], semi_label)
        return np.append(label,4)

    def __getitem__(self, index):

        src_point = eval(self.episods.iloc[index]["src_point"])
        goal_point = eval(self.episods.iloc[index]["goal_point"])

        goal_cat = self.cats[self.episods.iloc[index]["graph_name"].split("npy_")[1].split(".gpickle")[0]]        

        # map_graph_name = self.episods.iloc[index]["graph_name"]

        map_graph = nx.read_gpickle(self.episods.iloc[index]["graph_name"])
        shortest_path = np.array(nx.shortest_path(map_graph, source=src_point, target=goal_point))
        actions = self.convert_to_actions(shortest_path[:-1])
        gt_map_path = "./dataset/"+self.episods.iloc[index]["graph_name"].split("/")[2].split(".npy")[0].split("_BEV_semantic_map")[0]+"/BEV_semantic_map.npy"
        map_img = np.load(gt_map_path,allow_pickle=True).item()["semantic_map"]
        map_img = torch.tensor(np.stack((map_img,)*3, axis=0))
        one_hot_vect = np.zeros((40,))
        one_hot_vect[goal_cat]=1

        # Normalize your data here
        # if self.transform:
        # #     x = self.transform(shortest_path)
        # print(goal_point)
        # print(np.array(goal_point,dtype= np.float64))
        return np.array(src_point), np.array(goal_point), shortest_path[:-1], map_img, actions, one_hot_vect , self.episods.iloc[index]["graph_name"]
        
        
        
        
        
        
        
