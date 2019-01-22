PASCAL_DIR=/media/ssd1/austin/datasets/VOC/VOCdevkit/VOC2012

python ../../tools/test-ms-f.py --model models/model-f_iter_20000.caffemodel --images list/val_id.txt --dir ${PASCAL_DIR} --gpu 0 --smooth true
