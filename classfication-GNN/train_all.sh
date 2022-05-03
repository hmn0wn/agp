
python3 multiclass.py --dataset cora_full_semi  --agp_alg appnp_agp --alpha 0.1 --rmax 1e-7 --L 20 --lr 0.0001 --dropout 0.3 --hidden 2048   --batch 512  --layer 1 --epochs 200 --patience 200 --rep_num 1 |& tee ./logs/cora_full_semi_2048_512_200_200_1.log
python3 multiclass.py --dataset reddit_semi     --agp_alg appnp_agp --alpha 0.1 --rmax 1e-7 --L 20 --lr 0.0001 --dropout 0.3 --hidden 2048   --batch 512  --layer 1 --epochs 200 --patience 200 --rep_num 1 |& tee ./logs/reddit_semi_2048_512_200_200_1.log

python3 multiclass.py --dataset pubmed_semi     --agp_alg appnp_agp --alpha 0.1 --rmax 1e-7 --L 20 --lr 0.0001 --dropout 0.3 --hidden 2048   --batch 512  --layer 1 --epochs 200 --patience 200 --rep_num 1 |& tee ./logs/pubmed_semi_2048_512_200_200_5.log
python3 multiclass.py --dataset yelp_semi        --agp_alg appnp_agp --alpha 0.1 --rmax 1e-7 --L 20 --lr 0.0001 --dropout 0.3 --hidden 2048   --batch 512  --layer 1 --epochs 200 --patience 200 --rep_num 1 |& tee ./logs/yelp_2048_512_200_200_5.log
python3 multiclass.py --dataset ogbn-products_semi   --agp_alg appnp_agp --alpha 0.1 --rmax 1e-7 --L 20 --lr 0.0001 --dropout 0.3 --hidden 2048   --batch 512  --layer 1 --epochs 200 --patience 200 --rep_num 1 |& tee ./logs/ogbn-products_2048_512_200_200_5.log

python3 multilabel.py --dataset yelp_semi          --agp_alg appnp_agp --alpha 0.9 --L 20 --rmax 1e-8 --lr 0.01   --dropout 0.1 --hidden 2048   --batch 512            --epochs 200 --patience 200 --rep_num 5 |& tee ./logs/yelp_2048_512_200_200_5_multilabel.log
python3 multilabel.py --dataset ogbn-products_semi          --agp_alg appnp_agp --alpha 0.9 --L 20 --rmax 1e-8 --lr 0.01   --dropout 0.1 --hidden 2048   --batch 512            --epochs 200 --patience 200 --rep_num 5 |& tee ./logs/ogbn-products_2048_512_200_200_5_multilabel.log
