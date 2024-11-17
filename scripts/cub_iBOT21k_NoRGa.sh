for seed in 42 40 44
do
python main.py cub_norgaprompt \
--original_model vit_base_patch16_224_21k_ibot \
--model vit_base_patch16_224_21k_ibot \
--batch-size 128 \
--epochs 20 \
--data-path ./local_datasets/ \
--lr 0.0005 \
--output_dir ./output/norga_cub_ibot21k_covariance_mlp_2_seed$seed \
--seed $seed \
--train_inference_task_only \
--ca_storage_efficient_method covariance \
--crct_epochs 30 --shuffle True
done


for seed in 42 40 44
do
python main.py cub_norgaprompt \
--model vit_base_patch16_224_21k_ibot \
--original_model vit_base_patch16_224_21k_ibot \
--batch-size 128 \
--epochs 50 \
--data-path ./local_datasets/ \
--seed $seed \
--prompt_momentum 0.01 \
--reg 0.01 \
--length 20 \
--gate_act tanh \
--ca_storage_efficient_method covariance \
--trained_original_model ./output/norga_cub_ibot21k_covariance_mlp_2_seed$seed \
--output_dir ./output/norga_cub_ibot21k_pe_seed$seed --shuffle True --reset
done