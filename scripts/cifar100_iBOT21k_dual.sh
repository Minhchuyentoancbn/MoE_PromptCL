for seed in 42 40 44
do
python main.py cifar100_dualprompt \
--original_model vit_base_patch16_224_21k_ibot \
--model vit_base_patch16_224_21k_ibot \
--batch-size 128 \
--data-path ./local_datasets/ \
--output_dir ./output/dual_cifar100_ibot21_pe_seed$seed \
--seed $seed \
--shuffle True
done