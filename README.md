# semantic_map_navigation

In this project we addressed the object-goal navigation in unknown enviroment. This task has been a challenging task since no map and no reletive pose of the object is avalabel.

Contributions in this project is three fold:
1) We proposed a novel cNN-RNN based model to address the object goal navigation on semantic map.
2) We poposed a geound truth annotation on top of Matterport3D semantic_map for navigation episods.
3) We proposed a novel loss function to avoid robot from colision to obstackels.

# Dataset:
1)Matterport3D 
2)Generate ground truth semantic map from the labeled data
  - Depth, 2D ground-truth semantic segmentation, camera poses 
  - Build the 3D semantic segmentation, voxelized, project to the ground
3) Build Graph of scenes using Networkx
  - Every pixel is a node
  - Remove occupied nodes from the graph
4) Compute shortest path as ground truth annotation for robot path
  - Center of mass of the target object as goal point
  - Use A* to calculate the shortest path
5) 61 scenes, 2000 sample trajectory for each target object
  - Total of 480K trajectories


# Problem formulation:

We formulatr the problem as a classification task (⬅️, ⬆️, ⬇️, ➡️, 🛑). For the loss function we used an aulternate version of the cross entropty in-order to give a high negetive reward on Collision. The models train in form of Imitation Learning considering that the ground-truth paths are the shortest path computed using A*. 
The input to the model is the incremental map built using the observed 360˚  view  of the current location of robot(for simplisity we considerd the 20*20 patch around the robot as a current view).

## Model:

- CNN-GRU architecture
- 30 Epoch, learning rate=0.001 
-
![Picture1](https://user-images.githubusercontent.com/60449580/166802460-e669b667-acc5-468f-902c-dc76dac393dd.png)
