for seed in 42
do
python main.py five_datasets_norgaprompt \
--original_model vit_base_patch16_224 \
--model vit_base_patch16_224 \
--batch-size 128 \
--data-path ./local_datasets/ \
--output_dir ./output/NoRGa_5datasets_vit_covariance_mlp_2_seed$seed \
--sched constant \
--seed $seed \
--train_inference_task_only \
--ca_storage_efficient_method covariance \
--lr 0.001 --ca_lr 0.005 \
--crct_epochs 30 --epochs 20 --shuffle True
done


for seed in 42
do
python main.py five_datasets_norgaprompt \
--original_model vit_base_patch16_224 \
--model vit_base_patch16_224 \
--batch-size 128 \
--data-path ./local_datasets/ \
--output_dir ./output/NoRGa_5datasets_vit_pe_seed$seed \
--epochs 20 \
--sched constant \
--lr 0.03 \
--ca_lr 0.005 \
--clip-grad 2 \
--reg 0.1 \
--crct_epochs 30 \
--prompt_momentum 0.01 \
--seed $seed \
--larger_prompt_lr \
--ca_storage_efficient_method covariance \
--trained_original_model ./output/NoRGa_5datasets_vit_covariance_mlp_2_seed$seed \
--shuffle True --reset --gate_act tanh
done