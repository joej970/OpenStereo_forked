# @Time    : 2024/2/9 11:39
# @Author  : zhangchenming
from stereo.modeling.trainer_template import TrainerTemplate
# from .stereobase_gru import StereoBase as StereoBaseGRU
from .oneStereo import oneStereo

__all__ = {
    'oneStereo': oneStereo,
}


class Trainer(TrainerTemplate):
    def __init__(self, args, cfgs, local_rank, global_rank, logger, tb_writer):
        model = __all__[cfgs.MODEL.NAME](cfgs.MODEL)

        if args.run_mode == 'train':
            self.total_epochs = cfgs.OPTIMIZATION.NUM_EPOCHS
            if hasattr(cfgs.OPTIMIZATION, "NUM_EPOCHS_LIGHTSTEREO"):
                print(f"Using specific LightStereo training epochs: {cfgs.OPTIMIZATION.NUM_EPOCHS_LIGHTSTEREO}")
                logger.info(f"Using specific LightStereo training epochs: {cfgs.OPTIMIZATION.NUM_EPOCHS_LIGHTSTEREO}")
                self.total_epochs = cfgs.OPTIMIZATION.NUM_EPOCHS_LIGHTSTEREO

        super().__init__(args, cfgs, local_rank, global_rank, logger, tb_writer, model)

