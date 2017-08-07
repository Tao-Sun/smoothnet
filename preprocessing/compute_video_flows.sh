# /bin/bash

# Loop all the files in a folder in order and compute the flow between two consecutive files.

# $1: weights of a flow net model (.caffemodel)
# $2: architecture of a flow net model (.prototxt)
# $3: folder of the images
# $4: folder of the result 
# $5: folder of the flownet scripts
# $6: image start index

model_weights=$1
model_arch=$2
images_folder=$3
result_folder=$4
flownet_scripts_folder=$5
img_start_idx=$6

file_number="$(ls -l $images_folder | grep .png | wc -l)"
echo "There are $file_number images in the images folder."

for (( i=$img_start_idx; i<$((img_start_idx+file_number-1)); i++ )); do
  echo "computing: $i.png"
  $flownet_scripts_folder/run-flownet.py $model_weights $model_arch $images_folder/$i.png $images_folder/$((i+1)).png $result_folder/$i\_$((i+1)).flo
done


