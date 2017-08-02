import sys,os
import cv2
import numpy as np

video_path = sys.argv[1]
video_name = sys.argv[2]
video_suffix = sys.argv[3]
video_start_idx = int(sys.argv[4])
img_start_idx = int(sys.argv[5])
img_end_idx = int(sys.argv[6])

img_path = video_path + '/' + video_name + "_images"
print("image path: " + img_path)
if not os.path.exists(img_path):
    os.makedirs(img_path)

cap = cv2.VideoCapture(video_path + '/' + video_name + '.' + video_suffix)
ret, frame = cap.read()

i = video_start_idx
while(ret):
    if i >= img_start_idx:
        img_name = str(i) + '.png' #video_name + '_' + str(i).zfill(6) +'.png'
        cv2.imwrite(img_path + '/'+ img_name, frame)
    
    if i % 100 == 0:
        print(str(i) + " processed!")
    i = i + 1
    
    if i <= img_end_idx:
        ret, frame = cap.read()
    else:
        break

cap.release()



