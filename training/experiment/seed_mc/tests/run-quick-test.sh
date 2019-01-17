PASCAL_DIR=/media/ssd1/austin/datasets/VOC/VOCdevkit/VOC2012
GPU=0

CUDA_VISIBLE_DEVICES=0

# train step 1 (DSRG training)
echo "UNIQUESTRING 1"
python ../../../tools/train.py --solver solver-qs.prototxt --weights ../../../vgg16_20M_mc.caffemodel --gpu ${GPU}
echo "UNIQUESTRING 2"
python ../../../tools/quick-test-ms.py --model models/model-qs_iter_40.caffemodel --images list/train_aug_id.txt --dir ${PASCAL_DIR} --output ${PASCAL_DIR}/quick_DSRGOutput --gpu ${GPU} --smooth true

# train step 2 (retrain)
echo "UNIQUESTRING 3"
python ../../../tools/train.py --solver solver-qf.prototxt --weights models/model-qs_iter_40.caffemodel --gpu ${GPU}
echo "UNIQUESTRING 4"
python ../../../tools/quick-test-ms-f.py --model models/model-qf_iter_100.caffemodel --images list/val_id.txt --dir ${PASCAL_DIR} --output quick_DSRG_final_output --gpu 0 --smooth true
echo "UNIQUESTRING 5"
python ../../../tools/quick-evaluate.py --pred quick_DSRG_final_output --gt ${PASCAL_DIR}/SegmentationClass --test_ids list/val_id.txt --save_path quick_DSRG_result_final.txt --class_num 21
echo "UNIQUESTRING 6"
