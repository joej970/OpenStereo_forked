# @Time    : 2024/10/17 11:39
# @Author  : zanr
from stereo.modeling.trainer_template import TrainerTemplate
from functools import partial
from stereo.utils.common_utils import color_map_tensorboard, write_tensorboard
from stereo.evaluation.metric_per_image import epe_metric, d1_metric, threshold_metric
import torch.distributed as dist

# from .stereobase_gru import StereoBase as StereoBaseGRU
from .LeanStereo import LeanStereoNet

import os
import time
import torch

__all__ = {
    'LeanStereo': LeanStereoNet,
}


class Trainer(TrainerTemplate):
    def __init__(self, args, cfgs, local_rank, global_rank, logger, tb_writer):
        cfgs.MODEL.aux_mode = args.run_mode  # 'train' or 'eval' or 'test'
        print(f"Constructing LeanStereo model with aux_mode: {cfgs.MODEL.aux_mode}")
        model = __all__[cfgs.MODEL.NAME](cfgs.MODEL)
        super().__init__(args, cfgs, local_rank, global_rank, logger, tb_writer, model)

    def train_one_epoch(self, current_epoch, tbar):
        start_epoch = self.last_epoch + 1
        logger_iter_interval = self.cfgs.TRAINER.LOGGER_ITER_INTERVAL
        total_loss = 0.0
        loss_func = self.model.module.get_loss if self.args.dist_mode else self.model.get_loss

        print()
        print(f"LeanStereoTrainer: Training epoch: {current_epoch}")
        print()

        # Set debug printer epoch and total samples
        try:
            from stereo.modeling.models.oneStereo.debug_utils import debug_printer
            debug_printer.set_type('train')
            debug_printer.set_epoch(current_epoch)
            debug_printer.set_total_samples(len(self.train_loader))
        except ImportError:
            pass  # Debug utils not available

        # profiler
        prof = None
        if self.enable_profiler and self.local_rank == 0 and current_epoch == 0:
            from torch.profiler import profile, schedule, tensorboard_trace_handler, ProfilerActivity
            # socket.gethostname()}_{os.getpid().pt.torch.json
            message = ('\n\nInitializing profiler for training...\n\n')
            print(message)
            self.logger.info(message)
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
        else:
            message = f"Profiler not enabled or not local_rank 0 or not epoch 0: enable_profiler={self.enable_profiler}, local_rank={self.local_rank}, current_epoch={current_epoch}"
               
            print(message)
            self.logger.info(message)
        # profiler


        train_loader_iter = iter(self.train_loader)
        for i in range(0, len(self.train_loader)):
            # Update debug printer sample index
            try:
                from stereo.modeling.models.oneStereo.debug_utils import debug_printer
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
                loss, tb_info = loss_func(model_pred["disparities"], data)

            # 不要在autocast下调用, calls backward() on scaled loss to create scaled gradients.
            self.scaler.scale(loss).backward()
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

            if self.cfgs.TRAINER.TRAIN_VISUALIZATION:
                tb_info['image/train/image'] = torch.cat([data['left'][0], data['right'][0]], dim=1) / 256
                tb_info['image/train/disp'] = color_map_tensorboard(data['disp'][0], model_pred["disp_pred"].squeeze(1)[0])

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
            from stereo.modeling.models.oneStereo.debug_utils import debug_printer
            debug_printer.set_type('eval')
            debug_printer.set_epoch(current_epoch)
            debug_printer.set_total_samples(len(self.eval_loader))
        except ImportError:
            pass  # Debug utils not available

        # profiler
        prof = None
        if self.enable_profiler and self.local_rank == 0 and current_epoch == 0:
            from torch.profiler import profile, schedule, tensorboard_trace_handler, ProfilerActivity
            # socket.gethostname()}_{os.getpid().pt.torch.json
            print("Initializing profiler for evaluation...")
            prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule(wait=1, warmup=11, active=3, repeat=0),
                on_trace_ready=tensorboard_trace_handler(os.path.join(self.args.output_dir, "profiler"), worker_name=f"{self.args.experiment_id}_{self.args.slurm_job_id}_eval"),
                record_shapes=False,
                profile_memory=False,
                with_stack=True
            )
            prof.__enter__()
        # profiler

        for i, data in enumerate(self.eval_loader):
            # Update debug printer sample index
            try:
                from stereo.modeling.models.oneStereo.debug_utils import debug_printer
                debug_printer.set_sample(i) # this is actually batch number
            except ImportError:
                pass  # Debug utils not available

            for k, v in data.items():
                data[k] = v.to(local_rank) if torch.is_tensor(v) else v

            with torch.amp.autocast('cuda:%d' % local_rank, enabled=self.cfgs.OPTIMIZATION.AMP):
                infer_start = time.time()
                model_pred = self.model(data)
                infer_time = time.time() - infer_start

            disp_pred = model_pred["disp_pred"]
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
                        'image/eval/disp': color_map_tensorboard(data['disp'][0], model_pred["disp_pred"].squeeze(1)[0])
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

                # Set debug printer epoch and total samples
        try:
            from stereo.modeling.models.oneStereo.debug_utils import debug_printer
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
                with_stack=False
            )
            prof.__enter__()

        for i, data in enumerate(self.test_loader):
            for k, v in data.items():
                data[k] = v.to(local_rank) if torch.is_tensor(v) else v

            with torch.amp.autocast('cuda:%d' % local_rank, enabled=self.cfgs.OPTIMIZATION.AMP):
                infer_start = time.time()
                model_pred = self.model(data)
                infer_time = time.time() - infer_start

            disp_pred = model_pred["disp_pred"]
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
                if self.cfgs.TRAINER.TEST_VISUALIZATION and self.tb_writer is not None:
                    tb_info = {
                        'image/test/image': torch.cat([data['left'][0], data['right'][0]], dim=1) / 256,
                        'image/test/disp': color_map_tensorboard(data['disp'][0], model_pred["disp_pred"].squeeze(1)[0])
                    }
                    write_tensorboard(self.tb_writer, tb_info, current_epoch * len(self.test_loader) + i)

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
