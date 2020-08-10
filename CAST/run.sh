#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python NeuralStyleTransfer.py --content_dir images/contents_0425_l1 --content_src datasets/04191521_1000_100_1/warp   --content_list images/contents_list/l1.txt  --style_dir images/styles_104 --kl 50 --cw 1 --lw 30 --save_dir exp/0425_cw1_lw30_k50_l1 --gbp
