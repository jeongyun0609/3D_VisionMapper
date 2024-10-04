import os
import glob
import numpy as np
import argparse
import cv2
import json
import pyrender
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--seq', required=True,
                    default="./seq_all_01",
                    help="path of data sequence folder")
parser.add_argument('-m', '--models', required=True,
                    default="./models",
                    help="path of 3d models folder")                                       
args = parser.parse_args()

# How to use:
# python3 visualize_data.py -s=./sequences/seq_all_01 -m=./models


## draw camera trajectory ##
with open("%s/World2EEs.json"%(args.seq), 'r') as f:
    T_world2EEs = json.load(f)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for idx in T_world2EEs.keys():

    position = np.array(T_world2EEs[idx]["tra"]).reshape(3)
    orientation_vectors = np.array(T_world2EEs[idx]["rot"]).reshape(3, 3)
    
    for j, color in enumerate(['r', 'g', 'b']):
        ax.quiver(*position, *orientation_vectors[:, j]*10.0, color=color, alpha=1, linewidths=1)


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Camera Trajectory Plot')
plt.show()


## load data ##
colors = np.array([
  [0.89, 0.28, 0.13],
  [0.45, 0.38, 0.92],
  [0.35, 0.73, 0.63],
  [0.62, 0.28, 0.91],
  [0.65, 0.71, 0.22],
  [0.8, 0.29, 0.89],
  [0.27, 0.55, 0.22],
  [0.37, 0.46, 0.84],
  [0.84, 0.63, 0.22],
  [0.68, 0.29, 0.71],
  [0.48, 0.75, 0.48],
  [0.88, 0.27, 0.75],
  [0.82, 0.45, 0.2],
  [0.86, 0.27, 0.27],
  [0.52, 0.49, 0.18],
  [0.33, 0.67, 0.25],
  [0.67, 0.42, 0.29],
  [0.67, 0.46, 0.86],
  [0.36, 0.72, 0.84],
  [0.85, 0.29, 0.4],
  [0.24, 0.53, 0.55],
  [0.85, 0.55, 0.8],
  [0.4, 0.51, 0.33],
  [0.56, 0.38, 0.63],
  [0.78, 0.66, 0.46],
  [0.33, 0.5, 0.72],
  [0.83, 0.31, 0.56],
  [0.56, 0.61, 0.85],
  [0.89, 0.58, 0.57],
  [0.67, 0.4, 0.49]
])    

Axis_align = np.array([[1, 0, 0, 0],
                       [0, -1, 0, 0],
                       [0, 0, -1, 0],
                       [0, 0, 0, 1]])


cam_models = ["R", "L", "T"]

model_ids = open(args.models + "/object_label.json")    
model_ids = json.load(model_ids)

cam_K = dict()
cam_size = dict()
renderer = dict()
render_cam = dict()
scene = dict()
for cam in cam_models:
    with open("%s/cam_%s/camera_info.json"%(args.seq, cam), 'r') as f:
        cam_info = json.load(f)
        cam_K[cam] = np.array(cam_info["intrinsic"]).reshape(3, 3)
        cam_size[cam] = cam_info["size"]
        
        scene[cam] = pyrender.Scene()     
        renderer[cam] = pyrender.OffscreenRenderer(cam_info["size"][0], cam_info["size"][1])
        render_cam[cam] = pyrender.camera.IntrinsicsCamera(cam_info["intrinsic"][0],
                                                   cam_info["intrinsic"][4],
                                                   cam_info["intrinsic"][2],
                                                   cam_info["intrinsic"][5],
                                                   znear=0.01, zfar=100.0, name=None)
          
        cam_node = pyrender.Node(camera=render_cam[cam], matrix=np.eye(4))
        scene[cam].add_node(cam_node)                                                    
        

## draw masked images ##
for idx in T_world2EEs.keys():

    masked_imgs = dict()
    for cam in cam_models:
    
        objectmap = dict()       
        with open("%s/cam_%s/pose/%s.json"%(args.seq, cam, idx), 'r') as f:
                pose_data = json.load(f)                    
        for k in pose_data:    
            tm = trimesh.load("%s/%s/%s.obj"%(args.models, pose_data[k]["obj_name"], pose_data[k]["obj_name"]), force="mesh")   
            mesh = pyrender.Mesh.from_trimesh(tm, smooth = False) 
            node = pyrender.Node(mesh=mesh, matrix=np.eye(4))        
            objectmap[k] = {"node":node, "name":pose_data[k]["obj_name"]}        
            scene[cam].add_node(node)   
            
        if cam == "T":
            img = cv2.imread("%s/cam_T/8bit/%s.png"%(args.seq, idx))
        else:
            img = cv2.imread("%s/cam_%s/rgb/%s.png"%(args.seq, cam, idx))
            
        with open("%s/cam_%s/pose/%s.json"%(args.seq, cam, idx), 'r') as f:
            pose_data = json.load(f)        
        for k in pose_data:                            
            T_Cam2Obj = np.eye(4)            
            T_Cam2Obj[:3, :3] = np.array(pose_data[k]["rot"]).reshape(3, 3)
            T_Cam2Obj[:3, 3] = np.array(pose_data[k]["tra"]).reshape(3)*0.001                          
            scene[cam].set_pose(objectmap[k]["node"], pose= Axis_align @ T_Cam2Obj)

        full_depth = renderer[cam].render(scene[cam], flags = pyrender.constants.RenderFlags.DEPTH_ONLY)
    
        for k in pose_data:  
            objectmap[k]["node"].mesh.is_visible = False
            
        for k in pose_data: 
            objectmap[k]["node"].mesh.is_visible = True           
            depth = renderer[cam].render(scene[cam], flags = pyrender.constants.RenderFlags.DEPTH_ONLY)
            mask = np.logical_and((np.abs(depth - full_depth) < 1e-6), np.abs(full_depth) > 0)
            objectmap[k]["node"].mesh.is_visible = False 
            
            color_mask = np.zeros((cam_size[cam][1], cam_size[cam][0], 3))
            color_mask[:, :, 0] = colors[int(k)-1, 0]*255
            color_mask[:, :, 1] = colors[int(k)-1, 1]*255
            color_mask[:, :, 2] = colors[int(k)-1, 2]*255
            
            img[mask!=0] = img[mask!=0].astype(np.float32) * 0.7 + color_mask[mask!=0].astype(np.float32) * 0.3
            
        for k in pose_data:
            scene[cam]._remove_node(objectmap[k]["node"])   
            
        masked_imgs[cam] = img
            
    merged_img = np.zeros((480, 640*3, 3), dtype=np.uint8)
    merged_img[:, :640, :] = masked_imgs["L"]
    merged_img[:, 640:640*2, :] = masked_imgs["T"][16:512-16]
    merged_img[:, 640*2:640*3, :] = masked_imgs["R"]
    
    cv2.imshow("masked img", merged_img)
    cv2.waitKey(1)

