PASCAL_DIR=/media/ssd1/austin/datasets/VOC/VOCdevkit/VOC2012
GPU=0

CUDA_VISIBLE_DEVICES=0

# train step 1 (DSRG training)
echo "UNIQUESTRING 1"
python ../../tools/train.py --solver solver-s.prototxt --weights ../../vgg16_20M_mc.caffemodel --gpu ${GPU}
echo "UNIQUESTRING 2"
python ../../tools/test-ms.py --model models/model-s_iter_8000.caffemodel --images list/train_aug_id.txt --dir ${PASCAL_DIR} --output ${PASCAL_DIR}/DSRGOutput --gpu ${GPU} --smooth true

# train step 2 (retrain)
echo "UNIQUESTRING 3"
python ../../tools/train.py --solver solver-f.prototxt --weights models/model-s_iter_8000.caffemodel --gpu ${GPU}
echo "UNIQUESTRING 4"
python ../../tools/test-ms-f.py --model models/model-f_iter_20000.caffemodel --images list/val_id.txt --dir ${PASCAL_DIR} --output DSRG_final_output --gpu 0 --smooth true
echo "UNIQUESTRING 5"
python ../../tools/evaluate.py --pred DSRG_final_output --gt ${PASCAL_DIR}/SegmentationClass --test_ids list/val_id.txt --save_path DSRG_result_final.txt --class_num 21
echo "UNIQUESTRING 6"
