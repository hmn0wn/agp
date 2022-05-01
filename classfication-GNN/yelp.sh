python3 multilabel.py --dataset yelp          --agp_alg appnp_agp --alpha 0.9 --L 20 --rmax 1e-8 --lr 0.01   --dropout 0.1 --hidden 2048   --batch 512            --epochs 200 --patience 200 --rep_num 5 |& tee ./logs/yelp_2048_512_200_200_5.log
#python3 multiclass.py --dataset pubmed_semi   --agp_alg appnp_agp --alpha 0.1 --rmax 1e-7 --L 20 --lr 0.0001 --dropout 0.3 --hidden 2048   --batch 512  --layer 1 --epochs 200 --patience 200 --rep_num 5 |& tee ./logs/pubmed_semi_2048_512_200_200_5.log

#python -u multilabel.py --dataset yelp --agp_alg gdc_agp --ti 4 --L 20 --rmax 5e-7 --lr 0.01 --dropout 0.1 --hidden 2048 --batch 30000
#python -u multilabel.py --dataset yelp --agp_alg sgc_agp --L 10 --rmax 1e-8 --lr 0.01 --dropout 0.1 --hidden 2048 --batch 30000