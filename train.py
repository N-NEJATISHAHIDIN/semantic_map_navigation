from model import *
# import json
import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
import os
import matplotlib as plt
from utils import *
import networkx as nx
from data_loader import * 


num_workers = 0
num_epochs = 100
batch_size = 1
learning_rate = 0.0001
num_bins_az = 8
num_bins_el = 5
in_channels = 24
step_size = 30
input_path = "./dataset"
MODEL_PATH = "./model.pth"
mask_size =64
dim = 10
mini_batch = 10
max_steps = 200
graph_dict = {}


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print ('device', device)


all_maps = []
for root, dirs, files in os.walk(input_path):
        for name in files:
            if name == 'BEV_semantic_map.npy':                
                all_maps.append(root+"/"+name)  

with open('episods.csv', mode ='r') as file:
   
  # reading the CSV file
  episods = pd.read_csv(file)[:22001]

test_indx = np.random.choice(np.arange(0,22001), size = 8000,replace = False)
print(np.unique(test_indx).shape)
train_episods =  episods.iloc[np.delete(np.arange(0,22001),test_indx)]

test_episods = episods.iloc[test_indx]
print("train_episods:", train_episods.shape)
print("test_episods:", test_episods[:500].shape)

train_dataset = semantic_navigation_dl(train_episods)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_dataset = semantic_navigation_dl(test_episods[:500])
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

model = policy_model()

model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(model.parameters()), lr=learning_rate)

n_total_steps = len(train_dataloader)
scheduler = StepLR(optimizer, step_size=step_size)
accuracy_top = 0

for epoch in range(num_epochs):
        model.train()
        train_loss_total_f = 0
        for i, (src_point, goal_point, shortest_path, map_img, actions, goal_cat,_) in enumerate(train_dataloader):

                
                train_loss_total = 0
                # map_graph = nx.read_gpickle(self.episods.iloc[index]["graph_name"])

                # shortest_path = nx.shortest_path(map_graph, src_point, goal_point)
                # actions = path_to_action(shortest_path)

                # generate input patch
                # input_img = map_img[src_point[0]-dim:src_point[0]+dim, src_point[1]-dim:src_point[1]+dim]
                # print("shortest_path" , shortest_path.shape)
                itr=0
                actions = actions.long().to(device)[0]
                number_of_iter = shortest_path.shape[1] // mini_batch
                extra = shortest_path.shape[1] % mini_batch
                h0 = None 
                model.init_input_images(mini_batch,1000)

                for itr in range(number_of_iter):

                        mini_batch_actions = actions[itr*mini_batch:(itr+1)*mini_batch]
                        # print("mini", mini_batch_actions.shape)
                        optimizer.zero_grad()

                        # compute the model output
                        predicted_acctions = model(map_img.float().to(device), src_point.to(device), goal_point.float().to(device) , goal_cat.float().to(device) , itr, shortest_path.int().to(device) ,dim = dim, h0= h0, last_batch= False)
                        # print("shortest_path", shortest_path.shape)
                        # h0 = h0.detach()
                        # h0.requires_grad = True

                        # print("pred", predicted_acctions.shape)
                        # print(torch.argmax(predicted_acctions, axis =1), mini_batch_actions)

                        # calculate loss
                        train_loss = criterion(predicted_acctions, mini_batch_actions) 
                        train_loss_total += train_loss
                        # credit assignment
                        train_loss.backward()

                        # update model weights
                        optimizer.step()

                if(extra>0):
                        # print("extra",extra)

                        mini_batch_actions = actions[-extra:]

                        # print("mini_extra", mini_batch_actions.shape)
                        optimizer.zero_grad()

                        # compute the model output
                        predicted_acctions = model(map_img.float().to(device), src_point.float().to(device), goal_point.float().to(device) , goal_cat.float().to(device) , itr+1, shortest_path.int().to(device) ,dim = dim, mini_batch = extra, h0= h0, last_batch= True)

                        # h0 = h0.detach()
                        # h0.requires_grad = True

                        # print(torch.argmax(predicted_acctions, axis =1), mini_batch_actions)

                        # calculate loss
                        train_loss = criterion(predicted_acctions, mini_batch_actions) 
                        train_loss_total += train_loss
                        # print(train_loss)
                        # credit assignment
                        train_loss.backward()

                        # update model weights
                        optimizer.step()

                
                train_loss_total_f += train_loss_total/shortest_path.shape[1]
                # print(i)
                if((i+1)%100 == 0):

                        print("Epoch : ", epoch)
                        print("train Loss : ", train_loss_total_f/100)
                        train_loss_total_f = 0


        #testing
        model.eval()
        mini_batch_size = 1
        success = 0 
        ratio = 0

        for i, (src_point, goal_point, shortest_path , map_img, actions, goal_cat, graph_name) in enumerate(test_dataloader):

                model.init_input_images(mini_batch_size,1000)
                visited_path = torch.tensor([])

                if(graph_name not in graph_dict.keys()):
                        graph_dict[graph_name[0]] = nx.read_gpickle(graph_name[0])

                current_loc = src_point
                visited_path = torch.reshape(torch.cat((visited_path,src_point[0])), (1,1,-1))
                # print(visited_path)
                steps = 0
                while(steps < max_steps):

                        # print(steps)
                        # compute the model output
                        predicted_acctions = model(map_img.float().to(device), src_point.to(device), goal_point.float().to(device) , goal_cat.float().to(device) , steps, visited_path.int().to(device) , mini_batch= mini_batch_size, dim = dim, h0= h0, last_batch= False)
                        pred_action = torch.argmax(predicted_acctions, axis =1)

                        current_loc = convert_current_base_on_action(current_loc,pred_action)

                        # print(current_loc)
                        visited_path = torch.cat((visited_path,torch.unsqueeze(current_loc ,dim=0)),dim = 1)
                        # print("visited_path" , visited_path.shape)
# 
                        steps += 1
                        ret_success, len_path = compute_shortest_path(graph_dict[graph_name[0]], current_loc, goal_point)

                        if(ret_success == 0):

                                success += 1
                                break

                        elif(ret_success == 2):

                                break
                if(len_path != None and ret_success == 0):
                        
                        ratio += (len_path + len(visited_path[0]))/ len(shortest_path)
                        print("yes !!!", (len_path + len(visited_path[0])) )
                else:
                        print("Hit obstacle !!!")


        
        
        accuracy = success/len(test_dataloader)*100
        error_ration = ratio/ len(test_dataloader)

        print("success rate: ", accuracy)
        print("error ratio rate: ", error_ration)

        model_input_name = "Policy_network_Epoch"+ str(epoch+1)
        if (accuracy_top < accuracy) :
                accuracy_top = accuracy
                #save top model
                PATH = 'model_info/best_models/{}.pth'.format(model_input_name)
                
                if not os.path.exists('model_info/best_models/'):                  
                  # Create a new directory because it does not exist 
                  os.makedirs('model_info/best_models/')
                  os.makedirs('model_info/accuracy_info/')
                torch.save(model.state_dict(), PATH)

                #save top model accuracy info
                Top_model_accuracy_info = open("model_info/accuracy_info/"+model_input_name + ".txt","w")
                Top_model_accuracy_info.write("####################################### ** "+ model_input_name +" ** #######################################")
                Top_model_accuracy_info.write("\n")
                Top_model_accuracy_info.write(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss_total_f}, Test Success Rate: {str(accuracy)}, Test Error Ratio: {error_ration}')

        scheduler.step()
