# @Time    : 2024/1/20 03:13
# @Author  : zhangchenming
import os
import time
import glob
import torch
import torch.nn as nn
import torch.distributed as dist
from PIL import Image

from functools import partial
from stereo.datasets import build_dataloader
from stereo.utils import common_utils
from stereo.utils.common_utils import color_map_tensorboard, write_tensorboard
from stereo.utils.warmup import LinearWarmup
from stereo.utils.clip_grad import ClipGrad
from stereo.utils.lamb import Lamb
from stereo.evaluation.metric_per_image import epe_metric, d1_metric, threshold_metric


class TrainerTemplate:
    def __init__(self, args, cfgs, local_rank, global_rank, logger, tb_writer, model):
        self.args = args
        self.cfgs = cfgs
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.logger = logger
        self.tb_writer = tb_writer
        self.enable_profiler = True

        self.logger.info('Building model...')
        self.model = self.build_model(model)
        self.logger.info('Model built successfully.')
        
        self.best_epe = 1e6
        self.eval_epes = {}

        self.train_losses = {}
        self.train_lrs = {}

        self.last_train_epoch_idx = None
        self.last_train_loss = None

        error_map_list = self.cfgs.TRAINER.get('SAVE_ERROR_MAP_LIST', None)
        if error_map_list is not None:
            print(f"Save error maps for: {error_map_list}")
        else:
            print(f"Did not find SAVE_ERROR_MAP_LIST in {self.cfgs}")  

        # --- Add this block t
        if self.global_rank == 0:
            num_params = sum(p.numel() for p in self.model.parameters())
            num_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.logger.info(f"Model total parameters: {num_params:,}")
            self.logger.info(f"Model trainable parameters: {num_trainable:,}")
            if self.tb_writer is not None:
                self.tb_writer.add_text("Model/Stats", 
                    f"Total parameters: {num_params:,}<br>Trainable parameters: {num_trainable:,}", 
                    global_step=0)
        # --- End block ---

        if self.args.run_mode in ['train', 'eval']:
            self.eval_set, self.eval_loader, self.eval_sampler = self.build_eval_loader()

        if hasattr(cfgs,'TESTING'):
            self.test_set, self.test_loader, self.test_sampler = self.build_test_loader()

        if self.args.run_mode == 'train':
            self.train_set, self.train_loader, self.train_sampler = self.build_train_loader()

            if not hasattr(self, 'total_epochs'):
                self.total_epochs = cfgs.OPTIMIZATION.NUM_EPOCHS # set in child class

            if not hasattr(self, 'last_epoch'): # set in build_model if pretrained model is used
                self.last_epoch = -1

            self.optimizer, self.scheduler = self.build_optimizer_and_scheduler()
            # self.scaler = torch.cuda.amp.GradScaler(enabled=cfgs.OPTIMIZATION.AMP)
            self.scaler = torch.amp.GradScaler('cuda:%d' % self.local_rank, enabled=cfgs.OPTIMIZATION.AMP or cfgs.OPTIMIZATION.get('AMP_ENABLED_AFTER_EPOCH', None) is not None)

            if self.cfgs.MODEL.CKPT > -1:
                self.resume_ckpt()

            self.warmup_scheduler = self.build_warmup()
            self.clip_gard = self.build_clip_grad()

    def custom_after_train_callback(self):
        if self.last_train_loss is not None:
            print(f"After Epoch {self.last_train_epoch_idx}: Last Train Loss = {self.last_train_loss}")
            if torch.isnan(torch.tensor(self.last_train_loss)):
                raise ValueError("Last training loss is NaN after epoch callback.")
            elif torch.isinf(torch.tensor(self.last_train_loss)):
                raise ValueError("Last training loss is Inf after epoch callback.")
            elif self.last_train_loss > 1e3:
                raise ValueError("Last training loss is too large after epoch callback.")

           

    def build_train_loader(self):
        train_set, train_loader, train_sampler = build_dataloader(
            data_cfg=self.cfgs.DATA_CONFIG,
            batch_size=self.cfgs.OPTIMIZATION.BATCH_SIZE_PER_GPU,
            is_dist=self.args.dist_mode,
            workers=self.args.workers,
            pin_memory=self.args.pin_memory,
            mode='training')
        self.logger.info('Total samples for train dataset: %d' % (len(train_set)))
        return train_set, train_loader, train_sampler

    def build_eval_loader(self):
        eval_set, eval_loader, eval_sampler = build_dataloader(
            data_cfg=self.cfgs.DATA_CONFIG,
            batch_size=self.cfgs.EVALUATOR.BATCH_SIZE_PER_GPU,
            is_dist=self.args.dist_mode,
            workers=self.args.workers,
            pin_memory=self.args.pin_memory,
            mode='evaluating')
        self.logger.info('Total samples for eval dataset: %d' % (len(eval_set)))
        return eval_set, eval_loader, eval_sampler

    def build_test_loader(self):
        test_set, test_loader, test_sampler = build_dataloader(
            data_cfg=self.cfgs.DATA_CONFIG,
            batch_size=self.cfgs.TESTING.BATCH_SIZE_PER_GPU,
            is_dist=self.args.dist_mode,
            workers=self.args.workers,
            pin_memory=self.args.pin_memory,
            mode='testing')
        self.logger.info('Total samples for test dataset: %d' % (len(test_set)))
        return test_set, test_loader, test_sampler

    def build_model(self, model):
        if self.cfgs.OPTIMIZATION.get('FREEZE_BN', False):
            model = common_utils.freeze_bn(model)
            self.logger.info('Freeze the batch normalization layers')

        if self.cfgs.OPTIMIZATION.SYNC_BN and self.args.dist_mode:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            self.logger.info('Convert batch norm to sync batch norm')
        model = model.to(self.local_rank)

        if self.args.dist_mode:
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[self.local_rank], output_device=self.local_rank,
                find_unused_parameters=self.cfgs.MODEL.FIND_UNUSED_PARAMETERS)

        # load pretrained model
        if self.cfgs.MODEL.PRETRAINED_MODEL:
            self.logger.info('Loading parameters from checkpoint %s' % self.cfgs.MODEL.PRETRAINED_MODEL)

            # import re
            # numbers = re.findall(r'\d+', self.cfgs.MODEL.PRETRAINED_MODEL)
            # if numbers:
            #     epoch_num = numbers[-1]
            #     self.logger.info(f"Loading pretrained model from epoch {epoch_num}")

            # self.last_epoch = int(epoch_num) if numbers else -1

            # try:
            #     self.resume_ckpt_from_filename(self.cfgs.MODEL.PRETRAINED_MODEL)
            #     self.adjust_learning_rate_from_pretrained()
            # except Exception as e:
            # self.logger.info(f"Supplied pretrained model {self.cfgs.MODEL.PRETRAINED_MODEL} not containing all training states, will try to load model weights only. Got exception: {e}\n Continuing to load model weights only...")

            if not os.path.isfile(self.cfgs.MODEL.PRETRAINED_MODEL):
                raise FileNotFoundError
            common_utils.load_params_from_file(
                model, self.cfgs.MODEL.PRETRAINED_MODEL, device='cuda:%d' % self.local_rank,
                dist_mode=self.args.dist_mode, logger=self.logger, strict=False)
        return model

    def adjust_learning_rate_from_pretrained(self):
        """
        Adjust learning rate when loading from a pretrained model (not a full resume).
        This ensures the scheduler is in sync with the pretrained weight's epoch.
        """
        if self.last_epoch > -1:
            self.logger.info(f"Adjusting scheduler for pretrained model: last_epoch={self.last_epoch}")
            
            # If scheduler steps on epoch (like MultiStepLR in your config)
            if self.cfgs.OPTIMIZATION.SCHEDULER.ON_EPOCH:
                 # Scheduler starts at -1. We need to step it (last_epoch + 1) times to reach state AFTER last_epoch.
                 # e.g., if last_epoch=14 (finished ep 14), we want state ready for start of ep 15.
                steps_to_take = self.last_epoch + 1
                self.logger.info(f"Fast-forwarding scheduler {steps_to_take} steps (Epochs).")
                for _ in range(steps_to_take):
                    self.scheduler.step()
            else:
                # If scheduler steps on iteration
                if hasattr(self, 'train_loader'):
                    steps_to_take = (self.last_epoch + 1) * len(self.train_loader)
                    self.logger.info(f"Fast-forwarding scheduler {steps_to_take} steps (Iterations).")
                    for _ in range(steps_to_take):
                        self.scheduler.step()
                else:
                    raise ValueError("Cannot adjust learning rate from pretrained model because train_loader is not defined.")
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f"Scheduler state adjusted. Current Learning Rate: {current_lr}")


    def resume_ckpt_from_filename(self, ckpt_path):
        self.logger.info('Resume from ckpt: %s' % ckpt_path)
        print('Resume from ckpt: %s' % ckpt_path)

        checkpoint = torch.load(ckpt_path, map_location='cuda:%d' % self.local_rank)
        self.last_epoch = checkpoint['epoch']
        self.logger.info(f"Loaded checkpoint epoch: {self.last_epoch}")
        print(f"Loaded checkpoint epoch: {self.last_epoch}")
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scaler.load_state_dict(checkpoint['scaler_state'])
        if self.args.dist_mode:
            self.model.module.load_state_dict(checkpoint['model_state'])
        else:
            self.model.load_state_dict(checkpoint['model_state'])

    def build_optimizer_and_scheduler(self):
        if self.cfgs.OPTIMIZATION.OPTIMIZER.NAME == 'Lamb':
            optimizer_cls = Lamb
        else:
            optimizer_cls = getattr(torch.optim, self.cfgs.OPTIMIZATION.OPTIMIZER.NAME)
        valid_arg = common_utils.get_valid_args(optimizer_cls, self.cfgs.OPTIMIZATION.OPTIMIZER, ['name'])
        optimizer = optimizer_cls(params=[p for p in self.model.parameters() if p.requires_grad], **valid_arg)

        self.cfgs.OPTIMIZATION.SCHEDULER.TOTAL_STEPS = self.total_epochs * len(self.train_loader)
        scheduler_cls = getattr(torch.optim.lr_scheduler, self.cfgs.OPTIMIZATION.SCHEDULER.NAME)
        valid_arg = common_utils.get_valid_args(scheduler_cls, self.cfgs.OPTIMIZATION.SCHEDULER, ['name', 'on_epoch'])
        scheduler = scheduler_cls(optimizer, **valid_arg)

        return optimizer, scheduler

    def resume_ckpt(self):
        if self.cfgs.MODEL.get('CKPT_DIR', None) is not None:
            dir = self.cfgs.MODEL.CKPT_DIR
        else:
            dir = self.args.ckpt_dir

        ckpt_path = str(os.path.join(dir, 'checkpoint_epoch_%d.pth' % self.cfgs.MODEL.CKPT))
        self.logger.info('Resume from ckpt file: %s' % ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location='cuda:%d' % self.local_rank)
        self.last_epoch = checkpoint['epoch']
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scaler.load_state_dict(checkpoint['scaler_state'])
        if self.args.dist_mode:
            self.model.module.load_state_dict(checkpoint['model_state'])
        else:
            self.model.load_state_dict(checkpoint['model_state'])

        if self.cfgs.OPTIMIZATION.get('OPTI_REINIT', False):
            self.logger.info("Reinitializing optimizer and scheduler after loading pretrained model.")
            self.reinitialize_optimizer_and_scheduler(self.last_epoch)

    def reinitialize_optimizer_and_scheduler(self, last_epoch):
        self.optimizer, self.scheduler = self.build_optimizer_and_scheduler()
        self.warmup_scheduler = self.build_warmup()

        # fast forward scheduler to current epoch if resuming from pretrained model
        if self.cfgs.OPTIMIZATION.SCHEDULER.ON_EPOCH:
            steps_to_take = last_epoch + 1
            self.logger.info(f"Fast-forwarding scheduler {steps_to_take} steps (Epochs).")
            for _ in range(steps_to_take):
                self.scheduler.step()
        else:
            steps_to_take = (last_epoch + 1) * len(self.train_loader)
            self.logger.info(f"Fast-forwarding scheduler {steps_to_take} steps (Iterations).")
            for _ in range(steps_to_take):
                self.scheduler.step()

    def build_warmup(self):
        last_step = (self.last_epoch + 1) * len(self.train_loader) - 1
        if 'WARMUP' in self.cfgs.OPTIMIZATION.SCHEDULER:
            warmup_steps = self.cfgs.OPTIMIZATION.SCHEDULER.WARMUP.get('WARM_STEPS', 1)
            warmup_scheduler = LinearWarmup(
                self.optimizer,
                warmup_period=warmup_steps,
                last_step=last_step)
        else:
            warmup_scheduler = LinearWarmup(
                self.optimizer,
                warmup_period=1,
                last_step=last_step)

        return warmup_scheduler

    def build_clip_grad(self):
        clip_gard = None
        if 'CLIP_GRAD' in self.cfgs.OPTIMIZATION:
            clip_type = self.cfgs.OPTIMIZATION.CLIP_GRAD.get('TYPE', None)
            clip_value = self.cfgs.OPTIMIZATION.CLIP_GRAD.get('CLIP_VALUE', 0.1)
            max_norm = self.cfgs.OPTIMIZATION.CLIP_GRAD.get('MAX_NORM', 35)
            norm_type = self.cfgs.OPTIMIZATION.CLIP_GRAD.get('NORM_TYPE', 2)
            clip_gard = ClipGrad(clip_type, clip_value, max_norm, norm_type)
        return clip_gard

    def train(self, current_epoch, tbar):
        self.model.train()
        if self.cfgs.OPTIMIZATION.get('FREEZE_BN', False):
            self.model = common_utils.freeze_bn(self.model)
        if self.args.dist_mode:
            self.train_sampler.set_epoch(current_epoch)
            print(f"Rank {self.local_rank} set epoch to {current_epoch} for distributed training.")

        self.train_one_epoch(current_epoch=current_epoch, tbar=tbar)

        if self.cfgs.OPTIMIZATION.AMP == False:
            if self.cfgs.OPTIMIZATION.get('AMP_ENABLED_AFTER_EPOCH', None) is not None:
                if current_epoch+1 >= self.cfgs.OPTIMIZATION.AMP_ENABLED_AFTER_EPOCH:
                    self.logger.info(f"Enabling AMP after epoch {current_epoch}.")
                    self.cfgs.OPTIMIZATION.AMP = True
                    # self.scaler = torch.amp.GradScaler('cuda:%d' % self.local_rank, enabled=self.cfgs.OPTIMIZATION.AMP)
            # else:
                # self.logger.info("AMP is disabled for the entire training.")

        if self.args.dist_mode:
            dist.barrier()
        if self.cfgs.OPTIMIZATION.SCHEDULER.ON_EPOCH:
            self.scheduler.step()
            self.warmup_scheduler.lrs = [group['lr'] for group in self.optimizer.param_groups]

    def evaluate(self, current_epoch):
        self.model.eval()
        self.eval_one_epoch(current_epoch=current_epoch)
        if self.args.dist_mode:
            dist.barrier()

    def test(self, current_epoch):
        self.model.eval()
        results = self.test_one_epoch(current_epoch=current_epoch)
        if self.args.dist_mode:
            dist.barrier()
        return results

    def save_ckpt(self, current_epoch):
        if (current_epoch % self.cfgs.TRAINER.CKPT_SAVE_INTERVAL == 0 or current_epoch == self.total_epochs - 1) and self.global_rank == 0:
            ckpt_list = glob.glob(os.path.join(self.args.ckpt_dir, 'checkpoint_epoch_*.pth'))
            ckpt_list.sort(key=os.path.getmtime)
            if len(ckpt_list) >= self.cfgs.TRAINER.MAX_CKPT_SAVE_NUM:
                for cur_file_idx in range(0, len(ckpt_list) - self.cfgs.TRAINER.MAX_CKPT_SAVE_NUM + 1):
                    os.remove(ckpt_list[cur_file_idx])
            ckpt_name = os.path.join(self.args.ckpt_dir, 'checkpoint_epoch_%d.pth' % current_epoch)
            common_utils.save_checkpoint(self.model, self.optimizer, self.scheduler, self.scaler,
                                         self.args.dist_mode, current_epoch, filename=ckpt_name)
        if self.args.dist_mode:
            dist.barrier()

    def save_best_pth(self, current_epoch):
        if self.global_rank == 0:
            best_pth_name = os.path.join(self.args.ckpt_dir, 'best_model.pth')
            common_utils.save_checkpoint(self.model, self.optimizer, self.scheduler, self.scaler,
                                         self.args.dist_mode, current_epoch, filename=best_pth_name)

    def export_full_pytorch_model(model, location, name, example_input):
        """
        Exports the full PyTorch model as a TorchScript file.
        Args:
            model: The PyTorch model to export.
            location: Directory to save the model.
            name: File name (without extension).
            example_input: A sample input tensor or tuple for tracing.
        """
        os.makedirs(location, exist_ok=True)
        model.eval()
        # Use torch.jit.trace to export the model
        traced_model = torch.jit.trace(model, example_input)
        export_path = os.path.join(location, f"{name}.pt")
        traced_model.save(export_path)
        print(f"Model exported to {export_path}")

    def train_one_epoch(self, current_epoch, tbar):
        start_epoch = self.last_epoch + 1
        logger_iter_interval = self.cfgs.TRAINER.LOGGER_ITER_INTERVAL
        total_loss = 0.0
        loss_func = self.model.module.get_loss if self.args.dist_mode else self.model.get_loss

        print()
        print(f"DefaultTrainer: Training epoch: {current_epoch}")
        print()                  

        # Set debug printer epoch and total samples
        try:
            from tools.debug_utils import debug_printer
            debug_printer.set_type('train')
            debug_printer.set_epoch(current_epoch)
            debug_printer.set_total_samples(len(self.train_loader))
        except ImportError:
            pass  # Debug utils not available

        # profiler
        prof = None
        if self.enable_profiler and self.local_rank == 0 and current_epoch == 0:
            from torch.profiler import profile, schedule, tensorboard_trace_handler, ProfilerActivity

            prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule(wait=1, warmup=11, active=3, repeat=0),
                on_trace_ready=tensorboard_trace_handler(os.path.join(self.args.output_dir, "profiler"), worker_name=f"{self.args.experiment_id}_{self.args.slurm_job_id}_train"),
                record_shapes=False,
                profile_memory=False,
                # record_shapes=True,
                # profile_memory=True,
                with_stack=True
            )
            prof.__enter__()
        # profiler


        train_loader_iter = iter(self.train_loader)
        for i in range(0, len(self.train_loader)):
            # Update debug printer sample index
            try:
                from tools.debug_utils import debug_printer
                debug_printer.set_sample(i) # this is actually batch number
            except ImportError:
                pass  # Debug utils not available
                
            self.optimizer.zero_grad()
            lr = self.optimizer.param_groups[0]['lr']

            start_timer = time.time()
            data = next(train_loader_iter)
            for k, v in data.items():
                data[k] = v.to(self.local_rank) if torch.is_tensor(v) else v
            data_timer = time.time()

            # with torch.cuda.amp.autocast(enabled=self.cfgs.OPTIMIZATION.AMP):
            with torch.amp.autocast('cuda:%d' % self.local_rank, enabled=self.cfgs.OPTIMIZATION.AMP):
                data['iteration'] = i
                model_pred = self.model(data)
                infer_timer = time.time()
                loss, tb_info = loss_func(model_pred, data)

            # 不要在autocast下调用, calls backward() on scaled loss to create scaled gradients.
            self.scaler.scale(loss).backward()
            # In your training loop, after loss.backward()
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm()
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        print(f"NaN/Inf gradient in {name}")
                        print(f"Param stats: min={param.min():.3f}, max={param.max():.3f}")
                        
            # Check BatchNorm running stats
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):
                    if hasattr(module, 'running_mean'):
                        if torch.isnan(module.running_mean).any():
                            print(f"NaN in BatchNorm running_mean: {name}")
                        if torch.isinf(module.running_var).any():
                            print(f"Inf in BatchNorm running_var: {name}")
            # 做梯度剪裁的时候需要先unscale, unscales the gradients of optimizer's assigned params in-place
            self.scaler.unscale_(self.optimizer)
            # 梯度剪裁
            if self.clip_gard is not None:
                self.clip_gard(self.model)
            # optimizer's gradients are already unscaled, so scaler.step does not unscale them
            self.scaler.step(self.optimizer)
            # Updates the scale for next iteration.
            self.scaler.update()
            # torch.cuda.empty_cache()

            # warmup_scheduler period>1 和 batch_scheduler 不要同时使用
            with self.warmup_scheduler.dampening():
                if not self.cfgs.OPTIMIZATION.SCHEDULER.ON_EPOCH:
                    self.scheduler.step()

            total_loss += loss.item()
            total_iter = current_epoch * len(self.train_loader) + i
            trained_time_past_all = tbar.format_dict['elapsed']
            single_iter_second = trained_time_past_all / (total_iter + 1 - start_epoch * len(self.train_loader))
            remaining_second_all = single_iter_second * (self.total_epochs * len(self.train_loader) - total_iter - 1)
            if total_iter % logger_iter_interval == 0:
                message = ('Training Epoch:{:>2d}/{} Iter:{:>4d}/{} '
                           'Loss:{:#.6g}({:#.6g}) LR:{:.4e} '
                           'DataTime:{:.2f} InferTime:{:.2f}ms '
                           'Time cost: {}/{}'
                           ).format(current_epoch, self.total_epochs, i, len(self.train_loader),
                                    loss.item(), total_loss / (i + 1), lr,
                                    data_timer - start_timer, (infer_timer - data_timer) * 1000,
                                    tbar.format_interval(trained_time_past_all),
                                    tbar.format_interval(remaining_second_all))
                self.logger.info(message)
                # if total_loss / (i + 1) == float('inf') or 
                if torch.isnan(torch.tensor(total_loss / (i + 1))):
                    self.logger.error("Loss is NaN. Stopping training.")
                    raise ValueError("Loss is inf or NaN.")

            if self.cfgs.TRAINER.TRAIN_VISUALIZATION:
                tb_info['image/train/image'] = torch.cat([data['left'][0], data['right'][0]], dim=1) / 256
                tb_info['image/train/disp'] = color_map_tensorboard(data['disp'][0], model_pred['disp_pred'].squeeze(1)[0])

            tb_info.update({'scalar/train/lr': lr})
            if total_iter % logger_iter_interval == 0 and self.local_rank == 0 and self.tb_writer is not None:
                write_tensorboard(self.tb_writer, tb_info, total_iter)
            if prof:
                prof.step()
                message = ('Profiling Epoch:{:>2d} to file {:s} Iter:{:>4d}').format(
                    current_epoch, os.path.join(self.args.output_dir, "profiler"), i)
                self.logger.info(message)

           
                if i >= 15:  # Stop profiling after a few steps
                    self.logger.info("Profiling finished.")
                    message = ('{:s}').format(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
                    self.logger.info(message)
                    prof.__exit__(None, None, None)
                    prof = None

        self.train_losses[current_epoch] = total_loss / len(self.train_loader)
        self.train_lrs[current_epoch] = self.optimizer.param_groups[0]['lr']

        self.last_train_epoch_idx = current_epoch
        self.last_train_loss = total_loss / len(self.train_loader)

        if hasattr(self, 'custom_after_train_callback'):
            self.custom_after_train_callback()

    @torch.no_grad()
    def eval_one_epoch(self, current_epoch):

        metric_func_dict = {
            'epe': epe_metric,
            'd1_all': d1_metric,
            'thres_1': partial(threshold_metric, threshold=1),
            'thres_2': partial(threshold_metric, threshold=2),
            'thres_3': partial(threshold_metric, threshold=3),
        }

        evaluator_cfgs = self.cfgs.EVALUATOR
        local_rank = self.local_rank

        epoch_metrics = {}
        for k in evaluator_cfgs.METRIC:
            epoch_metrics[k] = {'indexes': [], 'values': []}

        # Set debug printer epoch and total samples
        try:
            from tools.debug_utils import debug_printer
            debug_printer.set_type('eval')
            debug_printer.set_epoch(current_epoch)
            debug_printer.set_total_samples(len(self.eval_loader))
        except ImportError:
            pass  # Debug utils not available

        # profiler
        prof = None
        if (self.enable_profiler and self.local_rank == 0 and current_epoch == 0) or current_epoch == -1:
            from torch.profiler import profile, schedule, tensorboard_trace_handler, ProfilerActivity
            # socket.gethostname()}_{os.getpid().pt.torch.json
            print("Initializing profiler for evaluation...")
            if current_epoch == -1:
                filename = f"{self.args.experiment_id}_{self.args.slurm_job_id}_eval_before_train"
            else:
                filename = f"{self.args.experiment_id}_{self.args.slurm_job_id}_eval"
            prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule(wait=1, warmup=11, active=3, repeat=0),
                on_trace_ready=tensorboard_trace_handler(os.path.join(self.args.output_dir, "profiler"), worker_name=filename),
                record_shapes=False,
                profile_memory=False,
                with_stack=True
            )
            prof.__enter__()

        for i, data in enumerate(self.eval_loader):
            # Update debug printer sample index
            try:
                from tools.debug_utils import debug_printer
                debug_printer.set_sample(i) # this is actually batch number
            except ImportError:
                pass  # Debug utils not available

            for k, v in data.items():
                data[k] = v.to(local_rank) if torch.is_tensor(v) else v

            with torch.amp.autocast('cuda:%d' % local_rank, enabled=self.cfgs.OPTIMIZATION.AMP):
                infer_start = time.time()
                model_pred = self.model(data)
                infer_time = time.time() - infer_start

            disp_pred = model_pred['disp_pred']
            disp_gt = data["disp"]
            mask = (disp_gt < evaluator_cfgs.MAX_DISP) & (disp_gt > 0)
            if 'occ_mask' in data and evaluator_cfgs.get('APPLY_OCC_MASK', False):
                mask = mask & ~data['occ_mask'].to(torch.bool)

            for m in evaluator_cfgs.METRIC:
                if m not in metric_func_dict:
                    raise ValueError("Unknown metric: {}".format(m))
                metric_func = metric_func_dict[m]
                res = metric_func(disp_pred.squeeze(1), disp_gt, mask)
                epoch_metrics[m]['indexes'].extend(data['index'].tolist())
                epoch_metrics[m]['values'].extend(res.tolist())

            if i % self.cfgs.TRAINER.LOGGER_ITER_INTERVAL == 0:
                message = ('Evaluating Epoch:{:>2d} Iter:{:>4d}/{} InferTime: {:.2f}ms'
                           ).format(current_epoch, i, len(self.eval_loader), infer_time * 1000)
                self.logger.info(message)

                if self.cfgs.TRAINER.EVAL_VISUALIZATION and self.tb_writer is not None:
                    tb_info = {
                        'image/eval/image': torch.cat([data['left'][0], data['right'][0]], dim=1) / 256,
                        'image/eval/disp': color_map_tensorboard(data['disp'][0], model_pred['disp_pred'].squeeze(1)[0])
                    }
                    if current_epoch != -1:
                        write_tensorboard(self.tb_writer, tb_info, current_epoch * len(self.eval_loader) + i)

            if prof:
                prof.step()
                message = ('Evaluation Profiling Epoch:{:>2d} to file {:s} Iter:{:>4d}').format(
                    current_epoch, os.path.join(self.args.output_dir, "profiler"), i)
                self.logger.info(message)
                if i >= 15:  # Stop profiling after a few steps
                    self.logger.info("Evaluation Profiling finished.")
                    message = ('{:s}').format(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
                    self.logger.info(message)
                    prof.__exit__(None, None, None)
                    prof = None

        # gather from all gpus
        if self.args.dist_mode:
            dist.barrier()
            self.logger.info("Start reduce metrics.")
            for k in epoch_metrics.keys():
                indexes = torch.tensor(epoch_metrics[k]["indexes"]).to(local_rank)
                values = torch.tensor(epoch_metrics[k]["values"]).to(local_rank)
                gathered_indexes = [torch.zeros_like(indexes) for _ in range(dist.get_world_size())]
                gathered_values = [torch.zeros_like(values) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_indexes, indexes)
                dist.all_gather(gathered_values, values)
                unique_dict = {}
                for key, value in zip(torch.cat(gathered_indexes, dim=0).tolist(),
                                      torch.cat(gathered_values, dim=0).tolist()):
                    if key not in unique_dict:
                        unique_dict[key] = value
                epoch_metrics[k]["indexes"] = list(unique_dict.keys())
                epoch_metrics[k]["values"] = list(unique_dict.values())

        results = {}
        for k in epoch_metrics.keys():
            results[k] = torch.tensor(epoch_metrics[k]["values"]).mean()

        if local_rank == 0 and self.tb_writer is not None:
            tb_info = {}
            for k, v in results.items():
                tb_info[f'scalar/val/{k}'] = v.item()

            write_tensorboard(self.tb_writer, tb_info, current_epoch)

        self.logger.info(f"Epoch {current_epoch} metrics: {results}")
        if current_epoch != -1:
            self.eval_epes[current_epoch] = results['epe'].item()
            if results['epe'] < self.best_epe:
                self.best_epe = results['epe']
                self.logger.info(f"New best EPE: {self.best_epe:.4f} at epoch {current_epoch}")
                # self.save_best_pth(current_epoch) # will do this in the main train
        return results



    @torch.no_grad()
    def test_one_epoch(self, current_epoch):
        metric_func_dict = {
            'epe': epe_metric,
            'd1_all': d1_metric,
            'thres_1': partial(threshold_metric, threshold=1),
            'thres_2': partial(threshold_metric, threshold=2),
            'thres_3': partial(threshold_metric, threshold=3),
        }

        testing_cfgs = self.cfgs.TESTING
        local_rank = self.local_rank

        epoch_metrics = {}
        for k in testing_cfgs.METRIC:
            epoch_metrics[k] = {'indexes': [], 'values': []}

        try:
            from tools.debug_utils import debug_printer
            debug_printer.set_type('test')
            debug_printer.set_epoch(current_epoch)
            debug_printer.set_total_samples(len(self.test_loader))
        except ImportError:
            pass  # Debug utils not available

        # profiler
        prof = None
        if self.enable_profiler and self.local_rank == 0 and current_epoch == 0:
            from torch.profiler import profile, schedule, tensorboard_trace_handler, ProfilerActivity

            print("Initializing profiler for testing...")
            
            # socket.gethostname()}_{os.getpid().pt.torch.json
            prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule(wait=1, warmup=11, active=3, repeat=0),
                on_trace_ready=tensorboard_trace_handler(os.path.join(self.args.output_dir, "profiler"), worker_name=f"{self.args.experiment_id}_{self.args.slurm_job_id}_test"),
                record_shapes=False,
                profile_memory=False,
                with_stack=True
            )
            prof.__enter__()

        for i, data in enumerate(self.test_loader):

            try:
                from tools.debug_utils import debug_printer
                debug_printer.set_sample(i) # this is actually batch number
            except ImportError:
                pass  # Debug utils not available
            
            for k, v in data.items():
                data[k] = v.to(local_rank) if torch.is_tensor(v) else v

            with torch.amp.autocast('cuda:%d' % local_rank, enabled=self.cfgs.OPTIMIZATION.AMP):
                infer_start = time.time()
                model_pred = self.model(data)
                infer_time = time.time() - infer_start

            disp_pred = model_pred['disp_pred']
            disp_gt = data["disp"]
            mask = (disp_gt < testing_cfgs.MAX_DISP) & (disp_gt > 0)
            if 'occ_mask' in data and testing_cfgs.get('APPLY_OCC_MASK', False):
                mask = mask & ~data['occ_mask'].to(torch.bool)

            for m in testing_cfgs.METRIC:
                if m not in metric_func_dict:
                    raise ValueError("Unknown metric: {}".format(m))
                metric_func = metric_func_dict[m]
                res = metric_func(disp_pred.squeeze(1), disp_gt, mask)
                epoch_metrics[m]['indexes'].extend(data['index'].tolist())
                epoch_metrics[m]['values'].extend(res.tolist())

            # if i % self.cfgs.TRAINER.LOGGER_ITER_INTERVAL == 0:
            message = ('Testing Epoch:{:>2d} Iter:{:>4d}/{} InferTime: {:.2f}ms'
                        ).format(current_epoch, i, len(self.test_loader), infer_time * 1000)
            self.logger.info(message)

            # check if it has the attribute TEST_VISUALIZATION
            if hasattr(self.cfgs.TRAINER, 'TEST_VISUALIZATION'):
                if self.cfgs.TRAINER.TEST_VISUALIZATION:
                    if self.tb_writer is not None:
                        tb_info = {
                            'image/test/image': torch.cat([data['left'][0], data['right'][0]], dim=1) / 256,
                            'image/test/disp': color_map_tensorboard(data['disp'][0], model_pred['disp_pred'].squeeze(1)[0])
                        }
                        write_tensorboard(self.tb_writer, tb_info, current_epoch * len(self.test_loader) + i)

                # print(f"{self.cfgs.SAVE_ERROR_MAP_FOR})
                # if data['name'] in self.cfgs.TRAINER.SAVE_ERROR_MAP_LIST:
 
                for idx, data_name in enumerate(data['name']):
                    if data_name in self.cfgs.TRAINER.SAVE_ERROR_MAP_LIST:
                        print(f"Saving error map for: {data_name}")
                        error_map = color_map_tensorboard(data['disp'][idx], model_pred['disp_pred'].squeeze(1)[idx])
                        save_directory = os.path.join(self.args.output_dir, 'test_error_maps')
                        
                        file_name = common_utils.get_filename_from_path(data_name)
                        file_name = f"{self.args.experiment_id}-{file_name}"
                        saving_loc = os.path.join(save_directory, file_name)
                        saving_loc = f"{saving_loc}.png"
                        os.makedirs(os.path.dirname(saving_loc), exist_ok=True)
                        print(f"saving error map: {saving_loc}")

                        im = Image.fromarray(error_map.mul(255).byte().cpu().numpy().transpose(1,2,0))
                        im.save(saving_loc)
                    # else:
                    #     print(f"Saving error map for {data_name} not requested.")
                        # save_image(error_map, saving_loc)


            if prof:
                prof.step()
                message = ('Profiling Testing Epoch:{:>2d} to file {:s} Iter:{:>4d}').format(
                    current_epoch, os.path.join(self.args.output_dir, "profiler"), i)
                self.logger.info(message)
                if i >= 15:  # Stop profiling after a few steps
                    self.logger.info("Evaluation Profiling finished.")
                    message = ('{:s}').format(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
                    self.logger.info(message)
                    prof.__exit__(None, None, None)
                    prof = None

        # gather from all gpus
        if self.args.dist_mode:
            dist.barrier()
            self.logger.info("Start reduce metrics.")
            for k in epoch_metrics.keys():
                indexes = torch.tensor(epoch_metrics[k]["indexes"]).to(local_rank)
                values = torch.tensor(epoch_metrics[k]["values"]).to(local_rank)
                gathered_indexes = [torch.zeros_like(indexes) for _ in range(dist.get_world_size())]
                gathered_values = [torch.zeros_like(values) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_indexes, indexes)
                dist.all_gather(gathered_values, values)
                unique_dict = {}
                for key, value in zip(torch.cat(gathered_indexes, dim=0).tolist(),
                                      torch.cat(gathered_values, dim=0).tolist()):
                    if key not in unique_dict:
                        unique_dict[key] = value
                epoch_metrics[k]["indexes"] = list(unique_dict.keys())
                epoch_metrics[k]["values"] = list(unique_dict.values())

        results = {}
        for k in epoch_metrics.keys():
            results[k] = torch.tensor(epoch_metrics[k]["values"]).mean()

        if local_rank == 0 and self.tb_writer is not None:
            tb_info = {}
            for k, v in results.items():
                tb_info[f'scalar/test/{k}'] = v.item()

            write_tensorboard(self.tb_writer, tb_info, current_epoch)

        self.logger.info(f"Testing: Epoch {current_epoch} metrics: {results}")

        return results

