# install

pip install numpy==1.26.4

# ssh

cd /tmp/pycharm_project_923
cd /tmp/pycharm_project_140
cd /tmp/pycharm_project_585

# data

https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels-segments.zip
https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip

http://images.cocodataset.org/zips/train2017.zip
http://images.cocodataset.org/zips/val2017.zip
http://images.cocodataset.org/zips/test2017.zip

# infer

export CUDA_VISIBLE_DEVICES=0

python detect.py --source ./data/images/horses.jpg --img 640 --device 0 --weights /home/Huangzhe/Test/yolov9-c-converted.pt --name yolov9_c_c_640_detect

python detect_dual.py --source /home/manu/tmp/BOSH-FM数据采集/xiang/X-170m-002.mp4 --img 1280 --device 0 --weights /run/user/1000/gvfs/smb-share:server=172.20.254.200,share=sharedfolder/Test/yolov9-s-fire-12809/weights/best.pt --name yolov9_s_c_1280_detect --view-img --conf-thres 0.25

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python detect_dual.py --source /home/Huangzhe/test/coco/coco/train2017 --img 1280 --device 7 --weights /home/Huangzhe/test/runs/train/yolov9-s-fire-s1280_6/weights/last.pt --name manu_detect --save-txt --save-conf --conf-thres 0.001 --project /home/Huangzhe/test/runs/test

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python detect_dual.py --source /home/Huangzhe/test/manu-pc/ST8000DM004/jb_raw/03数据标注-samples-merge --img 1280 --device 7 --weights /home/Huangzhe/test/runs/train/yolov9-s-fire-s1280_11/weights/last.pt --name detect --save-txt --save-conf --conf-thres 0.01 --project /home/Huangzhe/test/runs/fire --nosave

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python detect_dual.py --source /home/Huangzhe/test/manu-pc/tmp/点火视频.mp4 --img 1280 --device 7 --weights /home/Huangzhe/test/runs/train/yolov9-s-fire-s1280_6/weights/last.pt --name detect --save-txt --save-conf --conf-thres 0.1 --project /home/Huangzhe/test/manu-pc/tmp/runs/fire

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python detect_dual.py --source /home/Huangzhe/test/manu-pc/ST8000DM004/jb_raw/03数据标注-samples-merge --img 640 --device 7 --weights /home/Huangzhe/test/runs/train/yolov9-s-smoke-s640_13/weights/best.pt --name detect --save-txt --save-conf --conf-thres 0.01 --project /home/Huangzhe/test/runs/smoke

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python detect.py --source /home/Huangzhe/test/manu-pc/tmp/点火视频.mp4 --img 1280 --device 7 --weights /home/Huangzhe/test/runs/train/yolov9-s-fire-s1280_11/weights/yolov9-s-converted.pt --name detect_cvd --save-txt --save-conf --conf-thres 0.1 --project /home/Huangzhe/test/manu-pc/tmp/runs/fire_cvt

# evaluation

export CUDA_VISIBLE_DEVICES=0

python val.py --data data/coco.yaml --img 640 --batch 32 --conf 0.001 --iou 0.7 --device 0 --weights '/home/Huangzhe/Test/yolov9-c-converted.pt' --save-json --name /home/Huangzhe/Test/yolov9_c_c_640_val

python val.py --data data/coco.yaml --img 640 --batch 32 --conf 0.001 --iou 0.7 --device 0 --weights '/home/Huangzhe/Test/yolov9-s-converted.pt' --save-json --name /home/Huangzhe/Test/yolov9_s_c_640_val

# train

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nproc_per_node 8 --master_port 9527 train_dual.py --workers 8 --device 0,1,2,3,4,5,6,7 --sync-bn --batch 128 --data data/coco.yaml --img 640 --cfg models/detect/yolov9-c.yaml --weights '' --name yolov9-c --hyp hyp.scratch-high.yaml --min-items 0 --epochs 500 --close-mosaic 15

# finetune

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nproc_per_node 8 --master_port 9527 train_dual.py --workers 8 --device 0,1,2,3,4,5,6,7 --sync-bn --batch 64 --data data/fire.yaml --img 1280 --cfg models/detect/yolov9-s.yaml --weights '/home/Huangzhe/Test/yolov9-s-converted.pt' --name yolov9-s-fire-1280 --hyp hyp.scratch-high.yaml --min-items 0 --epochs 500 --close-mosaic 15

torchrun --nproc_per_node 8 --master_port 9527 train_dual.py --workers 8 --device 0,1,2,3,4,5,6,7 --sync-bn --batch 32 --data data/fire.yaml --img 1280 --cfg models/detect/yolov9-c.yaml --weights '/home/Huangzhe/Test/yolov9-c-converted.pt' --name yolov9-c-fire-s1280 --hyp hyp.scratch-high.yaml --min-items 0 --epochs 500 --close-mosaic 15

torchrun --nproc_per_node 8 --master_port 9527 train_dual.py --workers 8 --device 0,1,2,3,4,5,6,7 --sync-bn --batch 32 --data data/fire.yaml --img 1280 --cfg models/detect/yolov9-m.yaml --weights '/home/Huangzhe/test/yolov9-m.pt' --name yolov9-m-fire-s1280_ --hyp hyp.scratch-high.yaml --min-items 0 --epochs 500 --close-mosaic 15 --project '/home/Huangzhe/test/runs/train'

torchrun --nproc_per_node 8 --master_port 9527 train_dual.py --workers 8 --device 0,1,2,3,4,5,6,7 --sync-bn --batch 120 --data data/fire.yaml --img 1280 --cfg models/detect/yolov9-s.yaml --weights '/home/Huangzhe/test/yolov9-s.pt' --name yolov9-s-fire-s1280_ --hyp hyp.scratch-high.yaml --min-items 0 --epochs 500 --close-mosaic 15 --project '/home/Huangzhe/test/runs/train'

screen torchrun --nproc_per_node 8 --master_port 9527 train_dual.py --workers 8 --device 0,1,2,3,4,5,6,7 --sync-bn --batch 120 --data data/fire.yaml --img 1280 --cfg models/detect/yolov9-s.yaml --weights '/home/Huangzhe/test/runs/train/yolov9-s-fire-s1280_11/weights/last.pt' --name yolov9-s-fire-s1280_ --hyp hyp.scratch-high.yaml --min-items 0 --epochs 500 --close-mosaic 15 --project '/home/Huangzhe/test/runs/train' --patience 0

screen torchrun --nproc_per_node 8 --master_port 9527 train_dual.py --workers 8 --device 0,1,2,3,4,5,6,7 --sync-bn --batch 512 --data data/smoke.yaml --img 640 --cfg models/detect/yolov9-s.yaml --weights '/home/Huangzhe/test/yolov9-s.pt' --name yolov9-s-smoke-s640_ --hyp hyp.scratch-high.yaml --min-items 0 --epochs 500 --close-mosaic 15 --project '/home/Huangzhe/test/runs/train'

# tensorboard

tensorboard --logdir /home/Huangzhe/test/runs/train/yolov9-s-fire-s1280_6 --bind_all
tensorboard --logdir /home/Huangzhe/test/runs/train/yolov9-s-fire-s1280_10 --bind_all
tensorboard --logdir /home/Huangzhe/test/runs/train/yolov9-s-fire-s1280_11 --bind_all
tensorboard --logdir /home/Huangzhe/test/runs/train/yolov9-s-fire-s1280_27 --bind_all

tensorboard --logdir /home/Huangzhe/test/runs/train/yolov9-s-smoke-s640_13 --bind_all

http://172.20.254.200:6007/
http://172.20.254.132:6006/