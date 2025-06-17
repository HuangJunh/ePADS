dirdate=$(date +%Y-%m-%d-%H-%M-%S)
sigma=1.0

# Random Search
#nohup python main.py --algo 'rs' --runs 500 --max_evaluations 1000 --sl 6 --ptype 'nasbench201' --atom 5  --GPU '0' --sigma $sigma >> ./log_nb201-$dirdate.log 2>&1 &

# Correlation
#nohup python main_score.py --ptype 'nds_pnas' --sigma $sigma --GPU '0' >> ./running_pnas-$dirdate.log 2>&1 &
#nohup python main_score.py --ptype 'nds_enas' --sigma $sigma --GPU '0' >> ./running_enas-$dirdate.log 2>&1 &
#nohup python main_score.py --ptype 'nds_darts' --sigma $sigma --GPU '0' >> ./running_darts-$dirdate.log 2>&1 &
#nohup python main_score.py --ptype 'nds_nasnet' --sigma $sigma --GPU '0' >> ./running_nasnet-$dirdate.log 2>&1 &
#nohup python main_score.py --ptype 'nds_amoeba' --sigma $sigma --GPU '0' >> ./running_amoeba-$dirdate.log 2>&1 &
nohup python main_score.py --ptype 'nasbench201' --sigma $sigma --GPU '0' --dataset cifar10 >> ./running_nb201c10-$dirdate.log 2>&1 &
#nohup python main_score.py --ptype 'nasbench201' --sigma $sigma --GPU '0' --dataset cifar100 --data_loc ./datasets/CIFAR100_data/ >> ./running_nb201c100-$dirdate.log 2>&1 &
#nohup python main_score.py --ptype 'nasbench201' --sigma $sigma --GPU '0' --dataset ImageNet16-120 --data_loc ./datasets/ImageNet16/ >> ./running_nb201IMGNet-$dirdate.log 2>&1 &

