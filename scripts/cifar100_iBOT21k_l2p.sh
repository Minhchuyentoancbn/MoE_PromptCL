for seed in 42 40 44
do
python main.py cifar100_l2p \
--original_model vit_base_patch16_224_21k_ibot \
--model vit_base_patch16_224_21k_ibot \
--batch-size 128 \
--data-path ./local_datasets/ \
--output_dir ./output/l2p_cifar100_ibot21k_pe_seed$seed \
--seed $seed \
--shuffle True
done