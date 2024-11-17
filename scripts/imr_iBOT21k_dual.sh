for seed in 42 40 44
do
python main.py imr_dualprompt \
--original_model vit_base_patch16_224_21k_ibot \
--model vit_base_patch16_224_21k_ibot \
--batch-size 128 \
--data-path ./local_datasets/ \
--output_dir ./output/dual_imr_ibot21k_pe_seed$seed \
--seed $seed \
--shuffle True
done