# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import math
import time
from collections import defaultdict, deque
import datetime
import argparse
import numpy as np
from timm.utils import get_state_dict
            
import torch
import math
import copy

from pathlib import Path

import torch
import torch.distributed as dist
# from torch._six import inf
from torch import inf


from tensorboardX import SummaryWriter

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class TensorboardLogger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(logdir=log_dir)
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head='scalar', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(head + "/" + k, v, self.step if step is None else step)

    def flush(self):
        self.writer.flush()


class WandbLogger(object):
    def __init__(self, args):
        self.args = args

        try:
            import wandb
            self._wandb = wandb
        except ImportError:
            raise ImportError(
                "To use the Weights and Biases Logger please install wandb."
                "Run `pip install wandb` to install it."
            )

        # Initialize a W&B run 
        if self._wandb.run is None:
            self._wandb.init(
                project=args.project,
                config=args
            )

    def log_epoch_metrics(self, metrics, commit=True):
        """
        Log train/test metrics onto W&B.
        """
        # Log number of model parameters as W&B summary
        self._wandb.summary['n_parameters'] = metrics.get('n_parameters', None)
        metrics.pop('n_parameters', None)

        # Log current epoch
        self._wandb.log({'epoch': metrics.get('epoch')}, commit=False)
        metrics.pop('epoch')

        for k, v in metrics.items():
            if 'train' in k:
                self._wandb.log({f'Global Train/{k}': v}, commit=False)
            elif 'test' in k:
                self._wandb.log({f'Global Test/{k}': v}, commit=False)

        self._wandb.log({})

    def log_checkpoints(self):
        output_dir = self.args.output_dir
        model_artifact = self._wandb.Artifact(
            self._wandb.run.id + "_model", type="model"
        )

        model_artifact.add_dir(output_dir)
        self._wandb.log_artifact(model_artifact, aliases=["latest", "best"])

    def set_steps(self):
        # Set global training step
        self._wandb.define_metric('Rank-0 Batch Wise/*', step_metric='Rank-0 Batch Wise/global_train_step')
        # Set epoch-wise step
        self._wandb.define_metric('Global Train/*', step_metric='epoch')
        self._wandb.define_metric('Global Test/*', step_metric='epoch')


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    print(f"the elem is {os.environ}")
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()

        os.environ['RANK'] = str(args.rank)
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['WORLD_SIZE'] = str(args.world_size)
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

#lr_schedule_values = utils.cosine_scheduler(args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,)
def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule

def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
    for checkpoint_path in checkpoint_paths:
        to_save = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'scaler': loss_scaler.state_dict(),
            'args': args,
        }

        if model_ema is not None:
            to_save['model_ema'] = get_state_dict(model_ema)

        save_on_master(to_save, checkpoint_path)
    
    if is_main_process() and isinstance(epoch, int):
        to_del = epoch - args.save_ckpt_num * args.save_ckpt_freq
        old_ckpt = output_dir / ('checkpoint-%s.pth' % to_del)
        if os.path.exists(old_ckpt):
            os.remove(old_ckpt)




import os
import torch
import numpy as np
import argparse
from pathlib import Path

def update_ckpt(args, checkpoint, model_without_ddp):
    # 这里假设 update_ckpt 函数已有具体实现
    return checkpoint

def filter_mismatched_params(model_state_dict, checkpoint):
    """
    过滤掉形状不匹配的参数
    """
    filtered_checkpoint = {k: v for k, v in checkpoint.items() if
                           k in model_state_dict and v.shape == model_state_dict[k].shape}
    return filtered_checkpoint

def auto_load_model(args, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    output_dir = Path(args.output_dir)
    if args.auto_resume and len(args.resume) == 0:
        checkpoints = os.listdir(output_dir)
        checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth') and 'best' not in ckpt]
        print(f"All checkpoints founded in {output_dir}: {checkpoints}")
        if len(checkpoints) > 0:
            latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
            print(f"The latest checkpoint founded: {latest_checkpoint}")
            args.resume = latest_checkpoint

            try:
                # 获取要允许的全局对象
                scalar = np.core.multiarray.scalar
                dtype = np.dtype
                Float64DType = np.dtypes.Float64DType
                Namespace = argparse.Namespace  # 获取 argparse.Namespace 对象
                # 使用 safe_globals 上下文管理器允许特定的全局对象
                with torch.serialization.safe_globals([scalar, dtype, Float64DType, Namespace]):
                    torch.load(args.resume, map_location='cpu')
            except:
                latest_checkpoint = min([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
                print(f"Save unsuccessfully! The oldest checkpoint founded: {latest_checkpoint}")
                args.resume = latest_checkpoint

        print("Auto resume checkpoint: %s" % args.resume)

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            # 获取要允许的全局对象
            scalar = np.core.multiarray.scalar
            dtype = np.dtype
            Float64DType = np.dtypes.Float64DType
            Namespace = argparse.Namespace  # 获取 argparse.Namespace 对象
            # 使用 safe_globals 上下文管理器允许特定的全局对象
            with torch.serialization.safe_globals([scalar, dtype, Float64DType, Namespace]):
                checkpoint = torch.load(args.resume, map_location='cpu')

        model_state_dict = model_without_ddp.state_dict()
        try:
            if 'model' in checkpoint:
                filtered_checkpoint = filter_mismatched_params(model_state_dict, checkpoint['model'])
                model_state_dict.update(filtered_checkpoint)
                model_without_ddp.load_state_dict(model_state_dict, strict=False)
            else:
                filtered_checkpoint = filter_mismatched_params(model_state_dict, checkpoint)
                model_state_dict.update(filtered_checkpoint)
                model_without_ddp.load_state_dict(model_state_dict, strict=False)
            print("Loaded model state dict with filtered parameters.")
        except Exception as e:
            print(f"Failed to load model state dict: {e}")
            raise

        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except ValueError as e:
                print(f"Error loading optimizer state: {e}. Trying to partially load...")
                # 手动调整优化器状态
                saved_optimizer_state_dict = checkpoint['optimizer']
                current_optimizer_state_dict = optimizer.state_dict()

                # 只加载匹配的参数组
                for group in saved_optimizer_state_dict['param_groups']:
                    current_group = None
                    for c_group in current_optimizer_state_dict['param_groups']:
                        if len(c_group['params']) == len(group['params']):
                            current_group = c_group
                            break
                    if current_group is not None:
                        current_group['lr'] = group['lr']
                        current_group['weight_decay'] = group['weight_decay']
                        # 可以根据需要添加更多参数的复制

                # 加载状态
                optimizer.load_state_dict(current_optimizer_state_dict)

            if not isinstance(checkpoint['epoch'], str):  # does not support resuming with 'best', 'best-ema'
                try:
                    print(
                        "checkpoint['args'].replay_times:",
                        checkpoint['args'].replay_times
                    )
                except:
                    checkpoint['args'].replay_times = 1

                print('previous_start_epoch:', checkpoint['epoch'] + 1)
                if args.replay_times != checkpoint['args'].replay_times:
                    args.start_epoch = round(
                        (checkpoint['epoch'] + 1) * (checkpoint['args'].replay_times / args.replay_times))
                else:
                    args.start_epoch = checkpoint['epoch'] + 1

                args.start_epoch = round(
                    args.start_epoch * (checkpoint['args'].input_size ** 2) / (args.input_size ** 2))
                print('new_start_epoch:', args.start_epoch)
            else:
                assert args.eval, 'Does not support resuming with checkpoint-best'
            if hasattr(args, 'model_ema') and args.model_ema:
                if 'model_ema' in checkpoint.keys():
                    model_ema.ema.load_state_dict(checkpoint['model_ema'], strict=False)
                else:
                    model_ema.ema.load_state_dict(checkpoint['model'], strict=False)
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")
    return model



# import torch
# import os
# from pathlib import Path

# def update_ckpt(args, checkpoint, model):
#     # 此函数暂时不做修改，根据实际情况调整
#     return checkpoint

# def auto_load_model(args, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
#     output_dir = Path(args.output_dir)
#     if args.auto_resume and len(args.resume) == 0:
#         checkpoints = os.listdir(output_dir)
#         checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth') and 'best' not in ckpt]
#         print(f"All checkpoints founded in {output_dir}: {checkpoints}")
#         if len(checkpoints) > 0:
#             latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
#             print(f"The latest checkpoint founded: {latest_checkpoint}")
#             args.resume = latest_checkpoint

#             try:
#                 torch.load(args.resume, map_location='cpu')
#             except:
#                 latest_checkpoint = min([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
#                 print(f"Save unsuccessfully! The oldest checkpoint founded: {latest_checkpoint}")
#                 args.resume = latest_checkpoint

#         print("Auto resume checkpoint: %s" % args.resume)

#     if args.resume:
#         if args.resume.startswith('https'):
#             checkpoint = torch.hub.load_state_dict_from_url(
#                 args.resume, map_location='cpu', check_hash=True)
#         else:
#             checkpoint = torch.load(args.resume, map_location='cpu')

#         # 处理键不匹配问题
#         new_checkpoint = {}
#         for key in checkpoint.keys():
#             # 这里可以根据具体的键不匹配情况进行调整
#             new_key = key
#             if 'stage1.' in key:
#                 new_key = key.replace('stage1.', 'stage1.0.')
#             if 'stage4.' in key:
#                 parts = key.split('.')
#                 if parts[1].isdigit():
#                     new_key = f"stage4.{parts[1]}.{'.'.join(parts[2:])}"
#             if 'stage6.' in key:
#                 parts = key.split('.')
#                 if parts[1].isdigit():
#                     new_key = f"stage6.{parts[1]}.{'.'.join(parts[2:])}"
#             new_checkpoint[new_key] = checkpoint[key]

#         # 处理形状不匹配问题
#         model_state_dict = model_without_ddp.state_dict()
#         for key in new_checkpoint.keys():
#             if key in model_state_dict and new_checkpoint[key].shape != model_state_dict[key].shape:
#                 print(f"Shape mismatch for {key}: {new_checkpoint[key].shape} vs {model_state_dict[key].shape}")
#                 if key == 'conv2.weight':
#                     new_checkpoint[key] = new_checkpoint[key][:model_state_dict[key].size(0), :, :, :]
#                 elif key == 'conv2.bias':
#                     new_checkpoint[key] = new_checkpoint[key][:model_state_dict[key].size(0)]

#         try:
#             model_without_ddp.load_state_dict(new_checkpoint)
#             print(f"Loaded model state dict from checkpoint.")
#         except Exception as load_error:
#             print(f"Model name: {args.model}")
#             if 'deit' in args.model or 'swin' in args.model:
#                 new_checkpoint = update_ckpt(args, new_checkpoint, model_without_ddp)
#                 model_without_ddp.load_state_dict(new_checkpoint, strict=False)
#             else:
#                 print(f"Failed to load model state dict: {load_error}")
#                 try:
#                     model_without_ddp.load_state_dict(new_checkpoint, strict=False)
#                     print(f"Loaded model state dict from checkpoint with strict=False.")
#                 except Exception as new_error:
#                     print(f"Still failed to load model state dict: {new_error}")
#                     raise NotImplementedError

#         print("Resume checkpoint %s" % args.resume)
#         if 'optimizer' in new_checkpoint and 'epoch' in new_checkpoint:
#             optimizer.load_state_dict(new_checkpoint['optimizer'])
#             if not isinstance(new_checkpoint['epoch'], str):
#                 try:
#                     print(
#                         "new_checkpoint['args'].replay_times:",
#                         new_checkpoint['args'].replay_times
#                     )
#                 except:
#                     new_checkpoint['args'].replay_times = 1

#                 print('previous_start_epoch:', new_checkpoint['epoch'] + 1)
#                 if args.replay_times != new_checkpoint['args'].replay_times:
#                     args.start_epoch = round(
#                         (new_checkpoint['epoch'] + 1) * (new_checkpoint['args'].replay_times / args.replay_times))
#                 else:
#                     args.start_epoch = new_checkpoint['epoch'] + 1

#                 args.start_epoch = round(
#                     args.start_epoch * (new_checkpoint['args'].input_size ** 2) / (args.input_size ** 2))
#                 print('new_start_epoch:', args.start_epoch)
#             else:
#                 assert args.eval, 'Does not support resuming with checkpoint-best'
#             if hasattr(args, 'model_ema') and args.model_ema:
#                 if 'model_ema' in new_checkpoint.keys():
#                     model_ema.ema.load_state_dict(new_checkpoint['model_ema'], strict=False)
#                 else:
#                     model_ema.ema.load_state_dict(new_checkpoint, strict=False)
#             if 'scaler' in new_checkpoint:
#                 loss_scaler.load_state_dict(new_checkpoint['scaler'])
#             print("With optim & sched!")

    # return model
            
            
            


# def update_ckpt(args, checkpoint, model):
#     if 'deit' in args.model:
#         if 'model_ema' in checkpoint.keys():
#             to_update_list = ['model_ema']
#         else:
#             to_update_list = []
#         # 直接处理检查点中的模型参数
#         model_ckp_new = {}
#         for k in checkpoint.keys():
#             if 'pos_embed' not in k:
#                 model_ckp_new[k] = checkpoint[k]
#             if 'pos_embed' in k:
#                 print('bilinear interpolate & load:', k)
#                 temp_pb = checkpoint[k]
#                 pre_H_or_W = int(math.sqrt(temp_pb.size(1) - 1))
#                 num_embed_dim = temp_pb.size(2)
#                 new_H_or_W = int(args.input_size / 16)

#                 pos_embed_temp = temp_pb[:, 1:, :].transpose(1, 2).view(1, -1, pre_H_or_W, pre_H_or_W)
#                 pos_embed_temp = torch.nn.functional.interpolate(
#                     pos_embed_temp, size=(new_H_or_W, new_H_or_W), mode='bilinear', align_corners=False)
#                 pos_embed_temp = pos_embed_temp.flatten(2).transpose(1, 2)
#                 pos_embed_temp = torch.cat((temp_pb[:, 0, :].unsqueeze(1), pos_embed_temp), dim=1)

#                 model_ckp_new[k] = pos_embed_temp

#         # 更新检查点中的模型参数
#         for k in model_ckp_new.keys():
#             checkpoint[k] = model_ckp_new[k]

#         # 处理 optimizer 部分
#         opt_state_dict = checkpoint['optimizer']
#         new_opt_state_dict = copy.deepcopy(opt_state_dict)
#         for key in opt_state_dict['state']:
#             for kk in ['exp_avg', 'exp_avg_sq']:
#                 if len(opt_state_dict['state'][key][kk].shape) == 3:
#                     if opt_state_dict['state'][key][kk].size(0) == 1 and \
#                             opt_state_dict['state'][key][kk].size(1) == pre_H_or_W * pre_H_or_W + 1 and \
#                             opt_state_dict['state'][key][kk].size(2) == num_embed_dim:
#                         temp_pb_exp_avg = opt_state_dict['state'][key][kk]
#                         pre_H_or_W = int(math.sqrt(temp_pb_exp_avg.size(1) - 1))
#                         num_embed_dim = temp_pb_exp_avg.size(2)
#                         new_H_or_W = int(args.input_size / 16)

#                         pos_embed_temp = temp_pb_exp_avg[:, 1:, :].transpose(1, 2).view(1, -1, pre_H_or_W, pre_H_or_W)
#                         pos_embed_temp = torch.nn.functional.interpolate(
#                             pos_embed_temp, size=(new_H_or_W, new_H_or_W), mode='bilinear', align_corners=False)
#                         pos_embed_temp = pos_embed_temp.flatten(2).transpose(1, 2)
#                         pos_embed_temp = torch.cat((temp_pb_exp_avg[:, 0, :].unsqueeze(1), pos_embed_temp), dim=1)

#                         new_opt_state_dict['state'][key][kk] = pos_embed_temp

#         checkpoint['optimizer'] = new_opt_state_dict

#         # 处理 model_ema
#         for to_update in to_update_list:
#             model_ckp = checkpoint[to_update]
#             model_ckp_new = {}
#             for k in model_ckp.keys():
#                 if 'pos_embed' not in k:
#                     model_ckp_new[k] = model_ckp[k]
#                 if 'pos_embed' in k:
#                     print('bilinear interpolate & load:', k)
#                     temp_pb = model_ckp[k]
#                     pre_H_or_W = int(math.sqrt(temp_pb.size(1) - 1))
#                     num_embed_dim = temp_pb.size(2)
#                     new_H_or_W = int(args.input_size / 16)

#                     pos_embed_temp = temp_pb[:, 1:, :].transpose(1, 2).view(1, -1, pre_H_or_W, pre_H_or_W)
#                     pos_embed_temp = torch.nn.functional.interpolate(
#                         pos_embed_temp, size=(new_H_or_W, new_H_or_W), mode='bilinear', align_corners=False)
#                     pos_embed_temp = pos_embed_temp.flatten(2).transpose(1, 2)
#                     pos_embed_temp = torch.cat((temp_pb[:, 0, :].unsqueeze(1), pos_embed_temp), dim=1)

#                     model_ckp_new[k] = pos_embed_temp

#             checkpoint[to_update] = model_ckp_new

#     elif 'swin' in args.model:
#         if 'model_ema' in checkpoint.keys():
#             to_update_list = ['model_ema']
#         else:
#             to_update_list = []
#         # 直接处理检查点中的模型参数
#         model_ckp_new = {}
#         for k in checkpoint.keys():
#             if 'attn_mask' not in k:
#                 if 'relative_position_bias_table' not in k:
#                     if 'relative_position_index' not in k:
#                         model_ckp_new[k] = checkpoint[k]
#             if 'relative_position_bias_table' in k:
#                 print('bilinear interpolate & load:', k)
#                 temp_rl_pb = checkpoint[k]
#                 pre_2H_1 = int(math.sqrt(temp_rl_pb.size(0)))
#                 num_head = temp_rl_pb.size(1)
#                 new_2H_1 = 2 * model.window_size - 1
#                 temp_rl_pb = temp_rl_pb.view(pre_2H_1, pre_2H_1, -1).permute(2, 0, 1).unsqueeze(0)
#                 temp_rl_pb = torch.nn.functional.interpolate(
#                     temp_rl_pb, size=(new_2H_1, new_2H_1), mode='bilinear', align_corners=False
#                 ).squeeze().permute(1, 2, 0).view(new_2H_1 * new_2H_1, num_head)
#                 model_ckp_new[k] = temp_rl_pb

#         # 更新检查点中的模型参数
#         for k in model_ckp_new.keys():
#             checkpoint[k] = model_ckp_new[k]

#         # 处理 optimizer 部分
#         opt_state_dict = checkpoint['optimizer']
#         new_opt_state_dict = copy.deepcopy(opt_state_dict)
#         for key in opt_state_dict['state']:
#             for kk in ['exp_avg', 'exp_avg_sq']:
#                 if len(opt_state_dict['state'][key][kk].shape) == 2:
#                     if opt_state_dict['state'][key][kk].size(0) == pre_2H_1 * pre_2H_1 and \
#                             opt_state_dict['state'][key][kk].size(1) in model.num_heads_list:
#                         temp_rl_pb_exp_avg = opt_state_dict['state'][key][kk]
#                         pre_2H_1 = int(math.sqrt(temp_rl_pb_exp_avg.size(0)))
#                         num_head = temp_rl_pb_exp_avg.size(1)
#                         new_2H_1 = 2 * model.window_size - 1
#                         temp_rl_pb_exp_avg = temp_rl_pb_exp_avg.view(pre_2H_1, pre_2H_1, -1).permute(2, 0, 1).unsqueeze(0)
#                         temp_rl_pb_exp_avg = torch.nn.functional.interpolate(
#                             temp_rl_pb_exp_avg, size=(new_2H_1, new_2H_1), mode='bilinear', align_corners=False
#                         ).squeeze().permute(1, 2, 0).view(new_2H_1 * new_2H_1, num_head)
#                         new_opt_state_dict['state'][key][kk] = temp_rl_pb_exp_avg

#         checkpoint['optimizer'] = new_opt_state_dict

#         # 处理 model_ema
#         for to_update in to_update_list:
#             model_ckp = checkpoint[to_update]
#             model_ckp_new = {}
#             for k in model_ckp.keys():
#                 if 'attn_mask' not in k:
#                     if 'relative_position_bias_table' not in k:
#                         if 'relative_position_index' not in k:
#                             model_ckp_new[k] = model_ckp[k]
#                 if 'relative_position_bias_table' in k:
#                     print('bilinear interpolate & load:', k)
#                     temp_rl_pb = model_ckp[k]
#                     pre_2H_1 = int(math.sqrt(temp_rl_pb.size(0)))
#                     num_head = temp_rl_pb.size(1)
#                     new_2H_1 = 2 * model.window_size - 1
#                     temp_rl_pb = temp_rl_pb.view(pre_2H_1, pre_2H_1, -1).permute(2, 0, 1).unsqueeze(0)
#                     temp_rl_pb = torch.nn.functional.interpolate(
#                         temp_rl_pb, size=(new_2H_1, new_2H_1), mode='bilinear', align_corners=False
#                     ).squeeze().permute(1, 2, 0).view(new_2H_1 * new_2H_1, num_head)
#                     model_ckp_new[k] = temp_rl_pb

#             checkpoint[to_update] = model_ckp_new

#     return checkpoint

def update_ckpt(args, checkpoint, model):

    if 'deit' in args.model:
        import math

        if 'model_ema' in checkpoint.keys():
            to_update_list = ['model', 'model_ema']
        else:
            to_update_list = ['model']

        for to_update in to_update_list:
            model_ckp = checkpoint[to_update]
            model_ckp_new = {}

            for k in model_ckp.keys():
                if 'pos_embed' not in k:
                    model_ckp_new[k] = model_ckp[k]
                if 'pos_embed' in k:
                    print('bilinear interpolate & load:', k)
                    
                    temp_pb = model_ckp[k]
                    pre_H_or_W = int(math.sqrt(temp_pb.size(1) - 1))
                    num_embed_dim = temp_pb.size(2)
                    new_H_or_W = int(args.input_size / 16)
                    
                    pos_embed_temp = temp_pb[:, 1:, :].transpose(1, 2).view(1, -1, pre_H_or_W, pre_H_or_W)
                    pos_embed_temp = torch.nn.functional.interpolate(
                        pos_embed_temp, size=(new_H_or_W, new_H_or_W), mode='bilinear', align_corners=False)
                    pos_embed_temp = pos_embed_temp.flatten(2).transpose(1, 2)
                    pos_embed_temp = torch.cat((temp_pb[:, 0, :].unsqueeze(1), pos_embed_temp), dim=1)
                    
                    model_ckp_new[k] = pos_embed_temp

            checkpoint[to_update] = model_ckp_new

        import copy
        opt_state_dict = checkpoint['optimizer']
        new_opt_state_dict = copy.deepcopy(opt_state_dict)
        for key in opt_state_dict['state']:
            for kk in ['exp_avg', 'exp_avg_sq']:
                if len(opt_state_dict['state'][key][kk].shape) == 3:
                    
                    if opt_state_dict['state'][key][kk].size(0) == 1 and \
                        opt_state_dict['state'][key][kk].size(1) == pre_H_or_W * pre_H_or_W + 1 and \
                        opt_state_dict['state'][key][kk].size(2) == num_embed_dim:
                        
                        temp_pb_exp_avg = opt_state_dict['state'][key][kk]
                        pre_H_or_W = int(math.sqrt(temp_pb_exp_avg.size(1) - 1))
                        num_embed_dim = temp_pb_exp_avg.size(2)
                        new_H_or_W = int(args.input_size / 16)

                        pos_embed_temp = temp_pb_exp_avg[:, 1:, :].transpose(1, 2).view(1, -1, pre_H_or_W, pre_H_or_W)
                        pos_embed_temp = torch.nn.functional.interpolate(
                            pos_embed_temp, size=(new_H_or_W, new_H_or_W), mode='bilinear', align_corners=False)
                        pos_embed_temp = pos_embed_temp.flatten(2).transpose(1, 2)
                        pos_embed_temp = torch.cat((temp_pb_exp_avg[:, 0, :].unsqueeze(1), pos_embed_temp), dim=1)
                        
                        new_opt_state_dict['state'][key][kk] = pos_embed_temp
                        
        checkpoint['optimizer'] = new_opt_state_dict

    elif 'swin' in args.model:
        import math

        if 'model_ema' in checkpoint.keys():
            to_update_list = ['model', 'model_ema']
        else:
            to_update_list = ['model']

        for to_update in to_update_list:
            model_ckp = checkpoint[to_update]
            model_ckp_new = {}

            for k in model_ckp.keys():
                if 'attn_mask' not in k:
                    if 'relative_position_bias_table' not in k:
                        if 'relative_position_index' not in k:
                            model_ckp_new[k] = model_ckp[k]

                if 'relative_position_bias_table' in k:
                    print('bilinear interpolate & load:', k)
                    temp_rl_pb = model_ckp[k]
                    pre_2H_1 = int(math.sqrt(temp_rl_pb.size(0)))
                    num_head = temp_rl_pb.size(1)
                    new_2H_1 = 2 * model.window_size - 1
                    temp_rl_pb = temp_rl_pb.view(pre_2H_1, pre_2H_1, -1).permute(2, 0, 1).unsqueeze(0)
                    temp_rl_pb = torch.nn.functional.interpolate(
                        temp_rl_pb, size=(new_2H_1, new_2H_1), mode='bilinear', align_corners=False
                    ).squeeze().permute(1, 2, 0).view(new_2H_1 * new_2H_1, num_head)
                    model_ckp_new[k] = temp_rl_pb

            checkpoint[to_update] = model_ckp_new

        import copy
        opt_state_dict = checkpoint['optimizer']
        new_opt_state_dict = copy.deepcopy(opt_state_dict)
        for key in opt_state_dict['state']:
            for kk in ['exp_avg', 'exp_avg_sq']:
                if len(opt_state_dict['state'][key][kk].shape) == 2:
                    # 重新计算 pre_2H_1，避免未定义错误
                    temp_rl_pb_exp_avg = opt_state_dict['state'][key][kk]
                    pre_2H_1 = int(math.sqrt(temp_rl_pb_exp_avg.size(0)))
                    num_head = temp_rl_pb_exp_avg.size(1)
                    new_2H_1 = 2 * model.window_size - 1
                    if temp_rl_pb_exp_avg.size(0) == pre_2H_1 * pre_2H_1 and \
                                    temp_rl_pb_exp_avg.size(1) in model.num_heads_list:
                        temp_rl_pb_exp_avg = temp_rl_pb_exp_avg.view(pre_2H_1, pre_2H_1, -1).permute(2, 0, 1).unsqueeze(0)
                        temp_rl_pb_exp_avg = torch.nn.functional.interpolate(
                            temp_rl_pb_exp_avg, size=(new_2H_1, new_2H_1), mode='bilinear', align_corners=False
                        ).squeeze().permute(1, 2, 0).view(new_2H_1 * new_2H_1, num_head)
                        new_opt_state_dict['state'][key][kk] = temp_rl_pb_exp_avg
                        
        checkpoint['optimizer'] = new_opt_state_dict

    return checkpoint





checkpoint_path = '/home/bygpu/med/newmodel_weights.pth'  # 替换为实际的检查点文件路径
checkpoint = torch.load(checkpoint_path, map_location='cpu')
print(checkpoint.keys())