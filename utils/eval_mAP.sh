#!/usr/bin/env bash
python reval_voc_py3.py --year 2007 --classes voc.names \
				--image_set test --voc_dir /home/zhaoxiandong/datasets/VOCdevkit/ \
				--result_dir ../predictions/results --thresh_conf $1 \
				mAP_output