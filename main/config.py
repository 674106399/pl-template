import os
import os.path as osp
import sys

class Config:
    
    ## dataset
    dataset = 'rej_100w_20201209' # InterHand2.6M, RHD, STB

    ## input, output
    input_img_shape = (256, 256)

    ## model
    resnet_type = 50 # 18, 34, 50, 101, 152

    ## training config
    lr_dec_epoch = [15, 17] if dataset == 'rej_100w_20201209' else [45,47]
    end_epoch = 20 if dataset == 'rej_100w_20201209' else 50
    lr = 3e-4
    lr_dec_factor = 10
    train_batch_size = 12

    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    # data_dir = osp.join(root_dir, 'data')
    output_dir = osp.join(root_dir, 'output')
    model_dir = osp.join(output_dir, 'model_dump')
    vis_dir = osp.join(output_dir, 'vis')
    log_dir = osp.join(output_dir, 'log')
    result_dir = osp.join(output_dir, 'result')

    ## others
    num_thread = 8
    # gpu_ids = '0'
    # num_gpus = 1
    # continue_train = False

    # def set_args(self, gpu_ids, continue_train=False):
    #     self.gpu_ids = gpu_ids
    #     self.num_gpus = len(self.gpu_ids.split(','))
    #     self.continue_train = continue_train
    #     os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
    #     print('>>> Using GPU: {}'.format(self.gpu_ids))

cfg = Config()

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
from utils.filesystem import add_pypath, make_folder
# add_pypath(osp.join(cfg.data_dir))
# add_pypath(osp.join(cfg.data_dir, cfg.dataset))
make_folder(cfg.model_dir)
make_folder(cfg.vis_dir)
make_folder(cfg.log_dir)
make_folder(cfg.result_dir)