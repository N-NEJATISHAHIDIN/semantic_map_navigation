import numpy as np 
import pandas as pd 
import scipy
from scipy import ndimage
import matplotlib.pyplot as plt
# from pathplanninglib.a_star import *
# from pathplanninglib.gridmap import *
import networkx as nx
import os
import random
import torch
import csv


def multidim_intersect(arr1, arr2):
    arr1_view = arr1.view([('',arr1.dtype)]*arr1.shape[1])
    arr2_view = arr2.view([('',arr2.dtype)]*arr2.shape[1])
    intersected = numpy.intersect1d(arr1_view, arr2_view)
    return intersected.view(arr1.dtype).reshape(-1, arr1.shape[1])


def compute_shortest_path(map_graph, current_loc, goal_loc, min_path_lenght= 30 ):
    try:

        goal_loc = (goal_loc[0][0].item(),goal_loc[0][1].item())
        current_loc = (current_loc[0][0].item(),current_loc[0][1].item())
        
        shortest_path = np.array(nx.shortest_path(map_graph, source=current_loc, target=goal_loc))
        return 0 if len(shortest_path) < min_path_lenght else 1 , len(shortest_path)
    except:
        return 2, None 



def cleane_goal_points(src_point, boundries_coordinates_messy, gt_map, occ_map):

    G = nx.grid_2d_graph(occ_map.shape[0],occ_map.shape[1])
    valid_coordinates = list(zip(np.where(occ_map == 1)[0],np.where(occ_map == 1)[1]))
    G.remove_nodes_from(valid_coordinates)
    final_goal_points=[]
    for goal_point in boundries_coordinates_messy:
        if nx.has_path(G, src_point, goal_point):
            final_goal_points.append(goal_point)

    return np.asarray(final_goal_points)




def convert_current_base_on_action(current_loc,pred_action):
    if(pred_action == 0 ):
        current_loc += torch.tensor([1,0])

    elif(pred_action == 1 ):
        current_loc += torch.tensor([0,-1])

    elif(pred_action == 2 ):
        current_loc += torch.tensor([0,1])

    elif(pred_action == 3 ):
        current_loc += torch.tensor([-1,0])

    return current_loc




def make_episods( all_maps, per_scene_num = 2000 , objs = {"sofa" : 10, "chair": 3, "table":5, "bed": 11}):


    f = open('episods.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(["graph_name","src_point","goal_point"])

    all_graphs = []
    for gt_map_path in all_maps:

        gt_map = np.load(gt_map_path,allow_pickle=True).item()["semantic_map"]

        # make occupancy map
        occ_map = np.where(gt_map ==2 , 0, 1) *  np.where(gt_map == 16, 0, 1)

        # make the occupancy graph 
        G = nx.grid_2d_graph(occ_map.shape[0],occ_map.shape[1])
        occlusion_coordinates = list(zip(np.where(occ_map == 1)[0],np.where(occ_map == 1)[1]))
        G.remove_nodes_from(occlusion_coordinates)

        # for i in range(per_scene_num):
        for cat in objs.keys():

            

            # check if object exists
            if(np.sum(np.where(gt_map==objs[cat], 1, 0))>10):
                
                # try:
                print(gt_map_path, cat)
                
                # # get random starting point
                # valid_coordinates = list(zip(np.where(occ_map == 0)[0],np.where(occ_map == 0)[1]))
                # src_point = random.choice(valid_coordinates)
                
                # goal locations on map
                goal_map = np.where(gt_map==objs[cat], 1, 0)
                goal_map = scipy.ndimage.binary_fill_holes(goal_map).astype(int)

                # find center of all objects, add to the map graph

                # obj_center = ndimage.measurements.center_of_mass(goal_map)
                # G.add_node((obj_center))
                obj_center = (-1,-1)
                G.add_node((-1,-1))


                # shift map to find boundries
                shifted_map = np.roll(goal_map, -2, axis = 0) + np.roll(goal_map, 2, axis = 0)+ np.roll(goal_map, -2, axis = 1)+ np.roll(goal_map, 2, axis = 1)+ goal_map
                
                # find boundries from shifted map
                messy_boundries = np.where(shifted_map == 1, 1, 0)

                # only bounderies which are not occupied with other objects 
                final_bounderies = messy_boundries * (occ_map^(occ_map&1==occ_map))


                # check if all goal locations are reachable or not
                boundries_coordinates_messy = list(zip(np.where(final_bounderies == 1)[0],np.where(final_bounderies == 1)[1]))

                for edge in boundries_coordinates_messy:
                    G.add_edge(obj_center, edge)

                nx.write_gpickle(G, "./graphs/{}_{}_{}.gpickle".format(gt_map_path.split("/")[2],gt_map_path.split("/")[3], cat))

                #generate episods
                counter = 0
                while counter < per_scene_num:

                    # get random starting point
                    valid_coordinates = list(zip(np.where(occ_map == 0)[0],np.where(occ_map == 0)[1]))
                    src_point = random.choice(valid_coordinates)

                    if(nx.has_path(G ,src_point, obj_center)) :
                        counter += 1
                        writer.writerow(["./graphs/{}_{}_{}.gpickle".format(gt_map_path.split("/")[2],gt_map_path.split("/")[3], cat), str(src_point), str(obj_center)])
                        # writer.writerow( )




                all_graphs.append(G)
                G.remove_node(obj_center)


    f.close()               
    return all_graphs


def plot_path(src, dest , map , G):

    gt_map = np.load(map,allow_pickle=True).item()["semantic_map"]
    gt_map = np.where(gt_map == 99, 0, gt_map)
    path = np.array(nx.shortest_path(G, source=src, target=dest))[:-1]

    plt.imshow(gt_map)
    # print(path[:-1])
    plt.plot(path[0][1], path[0][0], '-ro', markersize=4)
    plt.plot(path[-1][1], path[-1][0], '-ro', markersize=4)

    plt.plot(path[:, 1], path[:, 0], color='green')

    plt.show()

# G = nx.read_gpickle("./graphs/ur6pFq6Qu1A_0_BEV_semantic_map.npy_chair.gpickle")
# plot_path( (362, 141) , (265.8343701399689, 206.76205287713842), "dataset/ur6pFq6Qu1A_0/BEV_semantic_map.npy" ,G )



def plot_path_patches(src, dest , map , G, mini_batch=10, dim = 10, itr =0):

    gt_map = np.load(map,allow_pickle=True).item()["semantic_map"]
    gt_map = np.where(gt_map == 99, 0, gt_map)

    shortest_path = np.array(nx.shortest_path(G, source=src, target=dest),dtype=np.int32)[:-1]

    # print(shortest_path)
    
    plt.imshow(gt_map)
    plt.show()    
    gt_map = torch.tensor(np.stack((gt_map,)*3, axis=0))
    print(torch.sum(gt_map))

    input_patches = np.zeros(( len(shortest_path) ,3,1000,1000))

    for patch in range(len(shortest_path)):

        current_loc = shortest_path[patch]
        print(current_loc)
        print(max(0,current_loc[0]-dim),min(current_loc[0]+dim, gt_map.shape[0]-1))
        # print(gt_map.shape)
        # curr_patch = gt_map[max(0,current_loc[0]-dim):min(current_loc[0]+dim, gt_map.shape[0]-1) , max(0,current_loc[1]-dim):min(current_loc[1]+dim, gt_map.shape[1]-1)]
        print(gt_map[:,max(0,current_loc[0]-dim):min(current_loc[0]+dim, gt_map.shape[1]-1) , 
                                            max(0,current_loc[1]-dim):min(current_loc[1]+dim, gt_map.shape[2]-1)])



        input_patches[patch:,:,max(0,current_loc[0]-dim):min(current_loc[0]+dim, gt_map.shape[1]-1),
                                max(0,current_loc[1]-dim):min(current_loc[1]+dim, gt_map.shape[2]-1)] = gt_map[:,max(0,current_loc[0]-dim):min(current_loc[0]+dim, gt_map.shape[1]-1) , 
                                            max(0,current_loc[1]-dim):min(current_loc[1]+dim, gt_map.shape[2]-1)].repeat(len(shortest_path)-patch,1,1,1)
    for curr_patch in input_patches:
        print(np.unique(curr_patch))
        plt.imshow((np.transpose(curr_patch,(2, 1, 0))*255).astype(np.uint8))
        plt.show()


G = nx.read_gpickle("./graphs/ur6pFq6Qu1A_0_BEV_semantic_map.npy_chair.gpickle")
# plot_path( (362, 141) , (265.8343701399689, 206.76205287713842), "dataset/ur6pFq6Qu1A_0/BEV_semantic_map.npy" ,G )

# plot_path_patches((362, 141) , (265.8343701399689, 206.76205287713842), "dataset/ur6pFq6Qu1A_0/BEV_semantic_map.npy" ,G , mini_batch=10, dim = 1000)

input_path = "./dataset"


all_maps = []
for root, dirs, files in os.walk(input_path):
        for name in files:
            if name == 'BEV_semantic_map.npy':                
                all_maps.append(root+"/"+name)  