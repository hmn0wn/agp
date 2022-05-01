 
current_time=$(date "+%y%m%d%H%M%S")
echo "Current Time : $current_time"
 
repeats=1
func=appnp_agp #gdc_agp
dim=32
batch=512
epochs=10

python3 -m memory_profiler multiclass.py --dataset pubmed_semi   --agp_alg $func --alpha 0.1 --rmax 1e-7 --L 20 --lr 0.0001 --dropout 0.3 --hidden $dim  --batch $batch  --layer 1 --epochs $epochs --patience $epochs --rep_num 1 |& tee "./logs/pubmed_${repeats}_${func}_${dim}_${batch}_${epochs}_${current_time}.log"

#python3 -m memory_profiler multiclass.py --dataset cora_full_semi   --agp_alg $func --alpha 0.1 --rmax 1e-7 --L 20 --lr 0.0001 --dropout 0.3 --hidden $dim  --batch $batch  --layer 1 --epochs $epochs --patience $epochs --rep_num 1 |& tee "./logs/cora_full_${repeats}_${func}_${dim}_${batch}_${epochs}_${current_time}.log"

#python3 -m memory_profiler multiclass.py --dataset reddit_semi   --agp_alg $func --alpha 0.1 --rmax 1e-7 --L 20 --lr 0.0001 --dropout 0.3 --hidden $dim  --batch $batch  --layer 1 --epochs $epochs --patience $epochs --rep_num 1 |& tee ./logs/reddit_$repeats_$func_$dim_$batch_$epochs_$current_time.log

#python3 -m memory_profiler multiclass.py --dataset amazon   --agp_alg $func --alpha 0.1 --rmax 1e-7 --L 20 --lr 0.0001 --dropout 0.3 --hidden $dim  --batch $batch  --layer 1 --epochs $epochs --patience $epochs --rep_num 1 |& tee ./logs/amazon_$repeats_$func_$dim_$batch_$epochs_$current_time.log
