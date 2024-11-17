for seed in 42
do
python main.py five_datasets_l2p \
--original_model vit_base_patch16_224_21k_ibot \
--model vit_base_patch16_224_21k_ibot \
--batch-size 128 \
--data-path ./local_datasets/ \
--output_dir ./output/l2p_5datasets_ibot21k_pe_seed$seed \
--seed $seed \
--shuffle True
done