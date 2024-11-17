for seed in 42 40 44
do
python main.py cifar100_sprompt_5e \
--original_model vit_base_patch16_224 \
--model vit_base_patch16_224 \
--batch-size 128 \
--data-path ./local_datasets/ \
--output_dir ./output/sprompt_cifar100_vit_pe_seed$seed \
--seed $seed \
--shuffle True
done