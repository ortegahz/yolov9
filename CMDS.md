# ssh

cd /tmp/pycharm_project_923

# data

https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels-segments.zip
https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip

http://images.cocodataset.org/zips/train2017.zip
http://images.cocodataset.org/zips/val2017.zip
http://images.cocodataset.org/zips/test2017.zip

# infer

export CUDA_VISIBLE_DEVICES=0

python detect.py --source ./data/images/horses.jpg --img 640 --device 0 --weights /home/Huangzhe/Test/yolov9-c-converted.pt --name yolov9_c_c_640_detect

python detect_dual.py --source /home/manu/tmp/8ECFA448-B884-4bbf-949F-5406AE198994.png --img 1280 --device 0 --weights /run/user/1000/gvfs/smb-share:server=172.20.254.200,share=sharedfolder/Test/yolov9-s-fire-12809/weights/best.pt --name yolov9_s_c_1280_detect --view-img --conf-thres 0.25

# evaluation

export CUDA_VISIBLE_DEVICES=0

python val.py --data data/coco.yaml --img 640 --batch 32 --conf 0.001 --iou 0.7 --device 0 --weights '/home/Huangzhe/Test/yolov9-c-converted.pt' --save-json --name /home/Huangzhe/Test/yolov9_c_c_640_val

python val.py --data data/coco.yaml --img 640 --batch 32 --conf 0.001 --iou 0.7 --device 0 --weights '/home/Huangzhe/Test/yolov9-s-converted.pt' --save-json --name /home/Huangzhe/Test/yolov9_s_c_640_val

# train

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nproc_per_node 8 --master_port 9527 train_dual.py --workers 8 --device 0,1,2,3,4,5,6,7 --sync-bn --batch 128 --data data/coco.yaml --img 640 --cfg models/detect/yolov9-c.yaml --weights '' --name yolov9-c --hyp hyp.scratch-high.yaml --min-items 0 --epochs 500 --close-mosaic 15

# finetune

torchrun --nproc_per_node 8 --master_port 9527 train_dual.py --workers 8 --device 0,1,2,3,4,5,6,7 --sync-bn --batch 64 --data data/fire.yaml --img 1280 --cfg models/detect/yolov9-s.yaml --weights '/home/Huangzhe/Test/yolov9-s-converted.pt' --name yolov9-s-fire-1280 --hyp hyp.scratch-high.yaml --min-items 0 --epochs 500 --close-mosaic 15

# tensorboard

tensorboard --logdir runs/train/yolov9-s-fire-12809 --bind_all
http://172.20.254.200:6007/