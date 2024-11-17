import datetime
import os
import numpy as np
import time
import torch
import warnings
import utils
import vits.norga_prompt_vision_transformer as norga_prompt_vision_transformer

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from datasets import build_continual_dataloader
from engines.norga_tii_engine import *

warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')

def train(args):
    device = torch.device(args.device)
    # Build data loader
    data_loader, data_loader_per_cls, class_mask, target_task_map = build_continual_dataloader(args)

    # Build ViT
    print(f"Creating original model: {args.original_model}")
    model = create_model(
        args.original_model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        mlp_structure=args.original_model_mlp_structure,
    )
    model.to(device)

    # Freeze ViT except MLP head
    if args.freeze:
        for n, p in model.named_parameters():
            if n.startswith(tuple(args.freeze)):
                p.requires_grad = False
    print(args)

    if args.eval:  # Evaluate only, no training
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

        for task_id in range(args.num_tasks):
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
            if os.path.exists(checkpoint_path):
                print('Loading checkpoint from:', checkpoint_path)
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model'])
            else:
                print('No checkpoint found at:', checkpoint_path)
                return
            _ = evaluate_till_now(model, data_loader, device,
                                  task_id, class_mask, target_task_map, acc_matrix, args, )
            

        # checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(args.num_tasks))
        checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task2_checkpoint.pth')
        if os.path.exists(checkpoint_path):
            print('Loading checkpoint from:', checkpoint_path)
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model'])
        else:
            print('No checkpoint found at:', checkpoint_path)
            return

        return
    

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # Set up learning rate
    if args.unscale_lr:
        global_batch_size = args.batch_size
    else:
        global_batch_size = args.batch_size * args.world_size
    args.lr = args.lr * global_batch_size / 256.0
    args.ca_lr = args.ca_lr * global_batch_size / 256.0

    # Build optimizer
    optimizer = create_optimizer(args, model_without_ddp)

    # Build scheduler
    if args.sched != 'constant':
        lr_scheduler, _ = create_scheduler(args, optimizer)
    elif args.sched == 'constant':
        lr_scheduler = None

    criterion = torch.nn.CrossEntropyLoss().to(device)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    train_and_evaluate(model, model_without_ddp,
                       criterion, data_loader, data_loader_per_cls, optimizer, lr_scheduler,
                       device, class_mask, target_task_map, args)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")

