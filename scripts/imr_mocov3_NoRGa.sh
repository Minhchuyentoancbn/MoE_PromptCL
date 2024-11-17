for seed in 42 40 44
do
python main.py imr_norgaprompt \
--original_model vit_base_patch16_224_mocov3 \
--model vit_base_patch16_224_mocov3 \
--batch-size 128 \
--data-path ./local_datasets/ \
--output_dir ./output/norga_imr_mocov3_covariance_mlp_2_seed$seed \
--seed $seed \
--train_inference_task_only \
--ca_storage_efficient_method covariance \
--lr 0.0005 --ca_lr 0.005 \
--crct_epochs 30 --epochs 20 \
--shuffle True
done

for seed in 42 40 44
do
python main.py imr_norgaprompt \
--model vit_base_patch16_224_mocov3 \
--original_model vit_base_patch16_224_mocov3 \
--batch-size 128 \
--data-path ./local_datasets/ --seed $seed \
--ca_lr 0.005 \
--crct_epochs 30 --epochs 50 \
--prompt_momentum 0.01 --length 20 \
--reg 0.5 \
--sched constant --larger_prompt_lr \
--ca_storage_efficient_method covariance \
--trained_original_model ./output/norga_imr_mocov3_covariance_mlp_2_seed$seed \
--output_dir ./output/norga_imr_mocov3_pe_seed$seed \
--shuffle True --reset --gate_act gelu
done