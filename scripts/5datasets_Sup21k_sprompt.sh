for seed in 42
do
python main.py five_datasets_sprompt_5e \
--original_model vit_base_patch16_224 \
--model vit_base_patch16_224 \
--batch-size 128 \
--data-path ./local_datasets/ \
--output_dir ./output/sprompt_5datasets_vit_pe_seed$seed \
--seed $seed \
--shuffle True --epochs 40
done