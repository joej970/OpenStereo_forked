# @Time    : 2024/2/9 11:39
# @Author  : zhangchenming
from stereo.modeling.trainer_template import TrainerTemplate
# from .stereobase_gru import StereoBase as StereoBaseGRU
from .oneStereo import oneStereo

__all__ = {
    'oneStereo': oneStereo,
}


class Trainer(TrainerTemplate):
    def __init__(self, args, cfgs, local_rank, global_rank, logger, tb_writer, enable_profiler=False):
        model = __all__[cfgs.MODEL.NAME](cfgs.MODEL)
        super().__init__(args, cfgs, local_rank, global_rank, logger, tb_writer, model, enable_profiler=enable_profiler)
