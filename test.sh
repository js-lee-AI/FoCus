#(c) 2021 NCSOFT Corporation & Korea University. All rights reserved.
###bart MTL
echo "tfidf"
CUDA_VISIBLE_DEVICES=5 python evaluate_test.py --model_name BART --model_checkpoint ./models/train_focus_BART_E2_L10_tfidf
CUDA_VISIBLE_DEVICES=6 python evaluate_test_ppl.py --model_name BART --model_checkpoint ./models/train_focus_BART_E2_L10_tfidf
echo "finish tfidf"

echo "bm25"
CUDA_VISIBLE_DEVICES=5 python evaluate_test.py --model_name BART --model_checkpoint ./models/train_focus_BART_E2_L10_bm25
CUDA_VISIBLE_DEVICES=6 python evaluate_test_ppl.py --model_name BART --model_checkpoint ./models/train_focus_BART_E2_L10_bm25
echo "finish bm25"

echo "sts"
CUDA_VISIBLE_DEVICES=5 python evaluate_test.py --model_name BART --model_checkpoint ./models/train_focus_BART_E2_L10_sts
CUDA_VISIBLE_DEVICES=6 python evaluate_test_ppl.py --model_name BART --model_checkpoint ./models/train_focus_BART_E2_L10_sts
echo "finish sts"

echo
