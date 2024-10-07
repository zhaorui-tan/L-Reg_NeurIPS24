DATASET="PACS"
DATADIR="/your/dataset/path"
# reproduce baselines, including ERM, CE, MIRO, GMDG
CUDA_VISIBLE_DEVICES=0 python train_all.py exp_name --data_dir ${DATADIR}  --dataset ${DATASET}  --algorithm MIRO --ld 0.0   --model swag_regnety_16gf --batch_size 22 --swad True --shift 0.   --d_shift 0.   --mask_range 0.5 --confidence 0.  --lr_mult 0.5   --low_degree 0.    --setting_name "CE"
CUDA_VISIBLE_DEVICES=0 python train_all.py exp_name --data_dir ${DATADIR}  --dataset ${DATASET}  --algorithm MIRO --ld 0.0   --model swag_regnety_16gf --batch_size 22 --swad True --shift 0.   --d_shift 0.   --mask_range 0.5 --confidence .1  --lr_mult 0.25  --low_degree 0.    --setting_name 'PIM'
CUDA_VISIBLE_DEVICES=0 python train_all.py exp_name --data_dir ${DATADIR}  --dataset ${DATASET}  --algorithm MIRO --ld 0.01  --model swag_regnety_16gf --batch_size 22 --swad True --shift 0.   --d_shift 0.   --mask_range 0.5 --confidence 0.  --lr_mult 0.5   --low_degree 0.    --setting_name "MIRO"
CUDA_VISIBLE_DEVICES=0 python train_all.py exp_name --data_dir ${DATADIR}  --dataset ${DATASET}  --algorithm MIRO --ld 0.01  --model swag_regnety_16gf --batch_size 22 --swad True --shift 0.1  --d_shift 0.1  --mask_range 0.5 --confidence 0.  --lr_mult 0.5   --low_degree 0.    --setting_name "GMDG"

# reproduce baselines with L-Reg
CUDA_VISIBLE_DEVICES=0 python train_all.py exp_name --data_dir ${DATADIR}  --dataset ${DATASET}  --algorithm MIRO --ld 0.0   --model swag_regnety_16gf --batch_size 22 --swad True --shift 0.   --d_shift 0.   --mask_range 0.5 --confidence 0.  --lr_mult 0.5   --low_degree 0.1   --setting_name "CE"
CUDA_VISIBLE_DEVICES=0 python train_all.py exp_name --data_dir ${DATADIR}  --dataset ${DATASET}  --algorithm MIRO --ld 0.0   --model swag_regnety_16gf --batch_size 22 --swad True --shift 0.   --d_shift 0.   --mask_range 0.5 --confidence .1  --lr_mult 0.25  --low_degree 0.01  --setting_name 'PIM'
CUDA_VISIBLE_DEVICES=0 python train_all.py exp_name --data_dir ${DATADIR}  --dataset ${DATASET}  --algorithm MIRO --ld 0.01  --model swag_regnety_16gf --batch_size 22 --swad True --shift 0.   --d_shift 0.   --mask_range 0.5 --confidence 0.  --lr_mult 0.5   --low_degree 0.1   --setting_name "MIRO"
CUDA_VISIBLE_DEVICES=0 python train_all.py exp_name --data_dir ${DATADIR}  --dataset ${DATASET}  --algorithm MIRO --ld 0.01  --model swag_regnety_16gf --batch_size 22 --swad True --shift 0.1  --d_shift 0.1  --mask_range 0.5 --confidence 0.  --lr_mult 0.5   --low_degree 0.1   --setting_name "GMDG"
