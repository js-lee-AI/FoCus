#(c) 2021 NCSOFT Corporation & Korea University. All rights reserved.

#CUDA_VISIBLE_DEVICES=0 python train_focus.py --n_epochs 2 --lm_coef 10 --flag E2_L10 --model_name BART --incontext --retrieval tfidf
#CUDA_VISIBLE_DEVICES=1 python train_focus.py --n_epochs 2 --lm_coef 10 --flag E2_L10 --model_name BART --incontext --retrieval bm25
#CUDA_VISIBLE_DEVICES=2 python train_focus.py --n_epochs 2 --lm_coef 10 --flag E2_L10 --model_name BART --incontext --retrieval dpr
CUDA_VISIBLE_DEVICES=3 python train_focus.py --n_epochs 2 --lm_coef 10 --flag E2_L10 --model_name BART --incontext --retrieval sts




