for seed in 42 40 44
do
python main.py cifar100_hideprompt_5e \
--original_model vit_base_patch16_224_21k_ibot \
--model vit_base_patch16_224_21k_ibot \
--batch-size 128 \
--data-path ./local_datasets/ \
--output_dir ./output/HiDe_cifar100_ibot21k_covariance_mlp_2_seed$seed \
--seed $seed \
--train_inference_task_only \
--ca_storage_efficient_method covariance \
--lr 0.0005 --ca_lr 0.005 \
--crct_epochs 30 --epochs 20 --shuffle True
done

for seed in 42 40 44
do
python main.py cifar100_hideprompt_5e \
--model vit_base_patch16_224_21k_ibot \
--original_model vit_base_patch16_224_21k_ibot \
--batch-size 128 \
--epochs 20 \
--data-path ./local_datasets/ \
--ca_lr 0.005 \
--crct_epochs 30 \
--seed $seed \
--prompt_momentum 0.1 \
--reg 0.1 \
--length 5 \
--larger_prompt_lr \
--ca_storage_efficient_method covariance \
--trained_original_model ./output/HiDe_cifar100_ibot21k_covariance_mlp_2_seed$seed \
--output_dir ./output/HiDe_cifar100_ibot21k_pe_seed$seed --shuffle True --reset
done