DATASET="TerraIncognita"
DATADIR="/your/dataset/path"
# reproduce baselines, including ERM, CE, MIRO, GMDG
CUDA_VISIBLE_DEVICES=1 python train_all.py exp_name --data_dir ${DATADIR}  --dataset ${DATASET}  --algorithm MIRO  --model swag_regnety_16gf --batch_size 22 --swad True  --ld 0.0   --shift 0.   --d_shift 0.    --mask_range 0.5 --confidence 0.    --lr_mult 10.  --low_degree 0.
CUDA_VISIBLE_DEVICES=4 python train_all.py exp_name --data_dir ${DATADIR}  --dataset ${DATASET}  --algorithm MIRO  --model swag_regnety_16gf --batch_size 22 --swad True  --ld 0.0   --shift 0.   --d_shift 0.    --mask_range 0.5 --confidence 0.1   --lr_mult 10.  --low_degree 0.
CUDA_VISIBLE_DEVICES=1 python train_all.py exp_name --data_dir ${DATADIR}  --dataset ${DATASET}  --algorithm MIRO  --model swag_regnety_16gf --batch_size 22 --swad True  --ld 0.01  --shift 0.   --d_shift 0.    --mask_range 0.5 --confidence 0.    --lr_mult 10.  --low_degree 0.
CUDA_VISIBLE_DEVICES=1 python train_all.py exp_name --data_dir ${DATADIR}  --dataset ${DATASET}  --algorithm MIRO  --model swag_regnety_16gf --batch_size 22 --swad True  --ld 0.01  --shift 0.01 --d_shift 0.01  --mask_range 0.5 --confidence 0.    --lr_mult 5.  --low_degree 0.

# reproduce baselines with L-Reg
CUDA_VISIBLE_DEVICES=4 python train_all.py exp_name --data_dir ${DATADIR}  --dataset ${DATASET}  --algorithm MIRO  --model swag_regnety_16gf --batch_size 22 --swad True  --ld 0.0   --shift 0.   --d_shift 0.    --mask_range 0.5 --confidence 0.    --lr_mult 10.  --low_degree .1
CUDA_VISIBLE_DEVICES=4 python train_all.py exp_name --data_dir ${DATADIR}  --dataset ${DATASET}  --algorithm MIRO  --model swag_regnety_16gf --batch_size 22 --swad True  --ld 0.0   --shift 0.   --d_shift 0.    --mask_range 0.5 --confidence 0.1   --lr_mult 10.  --low_degree .1
CUDA_VISIBLE_DEVICES=4 python train_all.py exp_name --data_dir ${DATADIR}  --dataset ${DATASET}  --algorithm MIRO  --model swag_regnety_16gf --batch_size 22 --swad True  --ld 0.01  --shift 0.   --d_shift 0.    --mask_range 0.5 --confidence 0.    --lr_mult 10.  --low_degree .1
CUDA_VISIBLE_DEVICES=4 python train_all.py exp_name --data_dir ${DATADIR}  --dataset ${DATASET}  --algorithm MIRO  --model swag_regnety_16gf --batch_size 22 --swad True  --ld 0.01  --shift 0.01 --d_shift 0.01  --mask_range 0.5 --confidence 0.    --lr_mult 5.  --low_degree .1
