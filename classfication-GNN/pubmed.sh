
python3 multiclass.py --dataset pubmed_semi   --agp_alg appnp_agp --alpha 0.1 --rmax 1e-7 --L 20 --lr 0.0001 --dropout 0.3 --hidden 2048   --batch 512  --layer 1 --epochs 200 --patience 200 --rep_num 1 |& tee ./logs/pubmed_semi_0_32_.log

#python3 -m memory_profiler multiclass.py --dataset cora_full_semi   --agp_alg appnp_agp --alpha 0.1 --rmax 1e-7 --L 20 --lr 0.0001 --dropout 0.3 --hidden 64   --batch 512  --layer 1 --epochs 200 --patience 200 --rep_num 10 |& tee last_pubmed10.log
#python3 -m memory_profiler multiclass.py --dataset reddit_semi      --agp_alg appnp_agp  --alpha 0.1 --rmax 1e-7 --L 20 --lr 0.0001 --dropout 0.3 --hidden 512 --batch 512 --layer 1 --epochs 200 --patience 200 --rep_num 10 |& tee last_reddit_appnp_10_512_512.log
#python3 -m memory_profiler multiclass.py --dataset amazon           --agp_alg appnp_agp --alpha 0.1 --rmax 1e-7 --L 20 --lr 0.0001 --dropout 0.3 --hidden 64 --batch 256 --layer 1 --epochs 200 --patience 200 --rep_num 10 |& tee last_amazon_appnp_10_64_256.log


# gdc_agp