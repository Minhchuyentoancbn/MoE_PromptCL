for seed in 42 40 44
do
python main.py imr_l2p \
--original_model vit_base_patch16_224 \
--model vit_base_patch16_224 \
--batch-size 128 \
--data-path ./local_datasets/ \
--output_dir ./output/l2p_imr_vit_pe_seed$seed \
--seed $seed \
--shuffle True
done