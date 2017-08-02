import numpy as np
import cv2
import sys, os

color_annot_dir = sys.argv[1]
gray_annot_dir = sys.argv[2]

animal = [64, 128, 64]	
archway = [192, 0, 128]	
bicyclist = [0, 128, 192]	
bridge = [0, 128, 64]
building = [128, 0, 0]
car = [64, 0, 128]	
cart_luggage_pram = [64, 0, 192]	
child = [192, 128, 64]	
column_pole = [192, 192, 128]	
fence =  [64, 64, 128]	
lane_mkgs_driv = [128, 0, 192]	
lane_mkgs_nondriv =  [192, 0, 64]	
misc_text = [128, 128, 64]	
motorcycle_scooter = [192, 0, 192]	
other_moving = [128, 64, 64]
parking_block = [64, 192, 128]	
pedestrian = [64, 64, 0]		
road = [128, 64, 128]	
road_shoulder = [128, 128, 192]	
sidewalk = [0, 0, 192]		
sign_symbol = [192, 128, 128]	
sky = [128, 128, 128]	
suv_pickup_truck = [64, 128, 192]	
traffic_cone = [0, 0, 64]		
traffic_light = [0, 64, 64]		
train = [192, 64, 128]	
tree = [128, 128, 0]	
truck_tus = [192, 128, 192]	
tunnel = [64, 0, 64]		
vegetation_misc = [192, 192, 0]	
void = [0, 0, 0]		
wall = [64, 192, 0]

label_colors = [animal, archway, bicyclist, bridge, building, car, cart_luggage_pram, child, column_pole, fence, lane_mkgs_driv, misc_text, motorcycle_scooter, other_moving, parking_block, pedestrian, road, road_shoulder, sidewalk, sign_symbol, sky, suv_pickup_truck, traffic_cone, traffic_light, train, tree, truck_tus, tunnel, vegetation_misc, void, wall]	


for _, _, files in os.walk(color_annot_dir):
    for f in files:
	if f.index('.png') > 0:
	    color_annot = cv2.imread(color_annot_dir + '/' + f)
	    gray_annot = np.zeros((color_annot.shape[0:2]))
	    for i, label_color in enumerate(label_colors):
		gray_annot[(color_annot[:,:,0] == label_color[0]) & (color_annot[:,:,1] == label_color[1]) & (color_annot[:,:,2] == label_color[2])] = i
            cv2.imwrite(gray_annot_dir + '/' + f, gray_annot)
            print(str(f) + ' processed!')    
