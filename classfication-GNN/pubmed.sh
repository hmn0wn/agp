python3 -m memory_profiler multiclass.py --dataset cora_full_semi   --agp_alg appnp_agp --alpha 0.1 --rmax 1e-7 --L 20 --lr 0.0001 --dropout 0.3 --hidden 64   --batch 512  --layer 1 --epochs 200 --patience 200 --rep_num 1 |& tee last_pubmed.log

#python3 -m memory_profiler multiclass.py --dataset cora_full_semi   --agp_alg appnp_agp --alpha 0.1 --rmax 1e-7 --L 20 --lr 0.0001 --dropout 0.3 --hidden 64   --batch 512  --layer 1 --epochs 200 --patience 200 --rep_num 10 |& tee last_pubmed10.log
#python3 -m memory_profiler multiclass.py --dataset reddit_semi      --agp_alg gdc_agp   --alpha 0.1 --rmax 1e-7 --L 20 --lr 0.0001 --dropout 0.3 --hidden 128 --batch 512 --layer 1 --epochs 200 --patience 200 --rep_num 10 |& tee last_reddit10_128.log
#python3 -m memory_profiler multiclass.py --dataset amazon           --agp_alg appnp_agp --alpha 0.1 --rmax 1e-7 --L 20 --lr 0.0001 --dropout 0.3 --hidden 512 --batch 512 --layer 1 --epochs 200 --patience 200 --rep_num 10 |& tee last_amazon10_512.log
