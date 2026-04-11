# @Time    : 2024/7/22 15:09
# @Author  : hao.hong

from stereo.modeling.trainer_template import TrainerTemplate
from .iinet import IINet

import time
from functools import partial

import os

import numpy as np
import torch
import torch.distributed as dist

from stereo.evaluation.metric_per_image import epe_metric, d1_metric, threshold_metric
from stereo.utils.common_utils import color_map_tensorboard, write_tensorboard

__all__ = {
    'IINet': IINet,
}

class Trainer(TrainerTemplate):
    def __init__(self, args, cfgs, local_rank, global_rank, logger, tb_writer):
        model = __all__[cfgs.MODEL.NAME](cfgs.MODEL)

        if args.run_mode == 'train':
            self.total_epochs = cfgs.OPTIMIZATION.NUM_EPOCHS
            if hasattr(cfgs.OPTIMIZATION, "NUM_EPOCHS_IINET"):
                print(f"Using specific IINet training epochs: {cfgs.OPTIMIZATION.NUM_EPOCHS_IINET}")
                logger.info(f"Using specific IINet training epochs: {cfgs.OPTIMIZATION.NUM_EPOCHS_IINET}")
                self.total_epochs = cfgs.OPTIMIZATION.NUM_EPOCHS_IINET

        super().__init__(args, cfgs, local_rank, global_rank, logger, tb_writer, model)

    def _get_pos_fullres(self, fx, w, h):
        x_range = (np.linspace(0, w - 1, w) + 0.5 - w // 2) / fx
        y_range = (np.linspace(0, h - 1, h) + 0.5 - h // 2) / fx
        x, y = np.meshgrid(x_range, y_range)
        z = np.ones_like(x)
        pos_grid = np.stack([x, y, z], axis=0).astype(np.float32)
        return pos_grid
    
    def train_one_epoch(self, current_epoch, tbar):
        start_epoch = self.last_epoch + 1
        logger_iter_interval = self.cfgs.TRAINER.LOGGER_ITER_INTERVAL
        total_loss = 0.0
        loss_func = self.model.module.get_loss if self.args.dist_mode else self.model.get_loss

        only_uncer = current_epoch <= self.cfgs.TRAINER.UNCER_ONLY_EPOCHS

        # Set debug printer epoch and total samples
        try:
            from tools.debug_utils import debug_printer
            debug_printer.set_type('train')
            debug_printer.set_epoch(current_epoch)
            debug_printer.set_total_samples(len(self.train_loader))
        except ImportError:
            pass  # Debug utils not available

        print(f"IINet Trainer: Epoch {current_epoch}: only_uncer = {only_uncer}")

        train_loader_iter = iter(self.train_loader)
        for i in range(0, len(self.train_loader)):
            self.optimizer.zero_grad()
            lr = self.optimizer.param_groups[0]['lr']

            start_timer = time.time()
            data = next(train_loader_iter)

            # Update debug printer sample index
            try:
                from tools.debug_utils import debug_printer
                debug_printer.set_sample(i) # this is actually batch number
                samples = ','.join(data['name']) if 'name' in data else 'unknown samples'
                debug_printer.set_sample_name(samples)
            except ImportError:
                pass  # Debug utils not available
                
            # IINet 预处理
            data_timer = time.time()    
            data['disp_pyr'] = data['disp'].unsqueeze(1)

            for k, v in data.items():
                data[k] = v.to(self.local_rank) if torch.is_tensor(v) else v
            data_timer = time.time()

            with torch.amp.autocast('cuda', enabled=self.cfgs.OPTIMIZATION.AMP):
                model_pred = self.model(input = data, only_uncer = only_uncer)
                infer_timer = time.time()
                loss, tb_info = loss_func(self.cfgs, data, model_pred, only_uncer = only_uncer)

            # 不要在autocast下调用, calls backward() on scaled loss to create scaled gradients.
            self.scaler.scale(loss).backward()
            # 做梯度剪裁的时候需要先unscale, unscales the gradients of optimizer's assigned params in-place
            self.scaler.unscale_(self.optimizer)

            # --- CUSTOM SAFETY CHECK FOR ABNORMAL GRADIENTS ---
            grad_is_valid = True
            grad_limit = 10.0 # Set threshold for "explosion"
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    # 1. Check for Inf/NaN (Scaler does this too, but this lets us log it)
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        debug_printer.print_of_function_force_print(lambda : f"NaN/Inf gradient detected in {name}")
                        # debug_printer.print_of_function_force_print(lambda : f"Skipping batch: NaN/Inf gradient detected in {name}")
                        # grad_is_valid = False
                        break
                    
                    # 2. Check for Finite Explosion
                    grad_max = param.grad.abs().max()
                    if grad_max > grad_limit:
                        # debug_printer.print_of_function_force_print(lambda : f"Skipping batch: Gradient explosion ({grad_max:.2f}) detected in {name}")
                        debug_printer.print_of_function_force_print(lambda : f"Gradient explosion ({grad_max:.2f}) detected in {name}")
                        # grad_is_valid = False
                        break
            
            # Only update weights if gradients are valid
            if grad_is_valid:
                # Gradient Clipping
                if self.clip_gard is not None:
                    self.clip_gard(self.model)
                
                # Update weights
                self.scaler.step(self.optimizer)
                self.scaler.update() 
            else:
                # If invalid, we throw away these gradients and do nothing to weights
                self.optimizer.zero_grad()

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

            if self.cfgs.TRAINER.TRAIN_VISUALIZATION:
                std = torch.tensor([ 0.229, 0.224, 0.225 ]).reshape(3,1,1).to(data['left'].device)
                mean = torch.tensor([ 0.485, 0.456, 0.406 ] ).reshape(3,1,1).to(data['left'].device)
                left = data['left'][0]*std + mean
                right = data['right'][0]*std + mean
                tb_info['image/train/image'] = torch.cat([left, right], dim=1)
                tb_info['image/train/disp'] = color_map_tensorboard(data['disp'][0], model_pred['disp_pred'].squeeze(1)[0]*16)

            tb_info.update({'scalar/train/lr': lr})
            if total_iter % logger_iter_interval == 0 and self.local_rank == 0 and self.tb_writer is not None:
                write_tensorboard(self.tb_writer, tb_info, total_iter)

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

        # Set debug printer epoch and total samples
        try:
            from tools.debug_utils import debug_printer
            debug_printer.set_type('eval')
            debug_printer.set_epoch(current_epoch)
            debug_printer.set_total_samples(len(self.eval_loader))
            debug_printer.set_print_eval_too(True)
        except ImportError:
            pass  # Debug utils not available

        evaluator_cfgs = self.cfgs.EVALUATOR
        local_rank = self.local_rank

        epoch_metrics = {}
        for k in evaluator_cfgs.METRIC:
            epoch_metrics[k] = {'indexes': [], 'values': []}

        # profiler
        prof = None
        if (self.enable_profiler and self.local_rank == 0 and current_epoch == 0) or current_epoch == -1:
            from torch.profiler import profile, schedule, tensorboard_trace_handler, ProfilerActivity
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
            with torch.amp.autocast('cuda', enabled=self.cfgs.OPTIMIZATION.AMP):
                infer_start = time.time()
                model_pred = self.model(data)
                infer_time = time.time() - infer_start

            disp_pred = model_pred['disp_pred']
            disp_gt = data["disp"]
            mask = (disp_gt < evaluator_cfgs.MAX_DISP) & (disp_gt > 0)
            if 'occ_mask' in data and evaluator_cfgs.get('APPLY_OCC_MASK', False):
                mask = mask & (data['occ_mask'] == 255.0)

            for m in evaluator_cfgs.METRIC:
                if m not in metric_func_dict:
                    raise ValueError("Unknown metric: {}".format(m))
                metric_func = metric_func_dict[m]
                res = metric_func(disp_pred.squeeze(1), disp_gt, mask)
                epoch_metrics[m]['indexes'].extend(data['index'].tolist())
                epoch_metrics[m]['values'].extend(res.tolist())


            results = {}
            for k in epoch_metrics.keys():
                results[k] = torch.tensor(epoch_metrics[k]["values"]).mean()

            if local_rank == 0 and self.tb_writer is not None:
                tb_info = {}
                for k, v in results.items():
                    tb_info[f'scalar/val/{k}'] = v.item()

                write_tensorboard(self.tb_writer, tb_info, current_epoch)

            self.eval_epes[current_epoch] = results['epe'].item()

            if i % self.cfgs.TRAINER.LOGGER_ITER_INTERVAL == 0:
                message = ('Evaluating Epoch:{:>2d} Iter:{:>4d}/{} InferTime: {:.2f}ms'
                           ).format(current_epoch, i, len(self.eval_loader), infer_time * 1000)
                self.logger.info(message)

                if self.cfgs.TRAINER.EVAL_VISUALIZATION and self.tb_writer is not None:
                    std = torch.tensor([ 0.229, 0.224, 0.225 ]).reshape(3,1,1).to(data['left'].device)
                    mean = torch.tensor([ 0.485, 0.456, 0.406 ] ).reshape(3,1,1).to(data['left'].device)
                    left = data['left'][0]*std + mean
                    right = data['right'][0]*std + mean
                    tb_info = {
                        'image/eval/image': torch.cat([left, right], dim=1),
                        'image/eval/disp': color_map_tensorboard(data['disp'][0], model_pred['disp_pred'].squeeze(1)[0])
                    }
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