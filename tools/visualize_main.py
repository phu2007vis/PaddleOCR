# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

import yaml
import paddle
import paddle.distributed as dist

from ppocr.data import build_dataloader, set_signal_handlers
from ppocr.utils.utility import set_seed
import tools.program as program
from visualize import visualize
dist.get_world_size()


def main(config, device, logger, vdl_writer, seed):
    # init dist environment
    if config["Global"]["distributed"]:
        dist.init_parallel_env()

    global_config = config["Global"]

    # build dataloader
    set_signal_handlers()
    train_dataloader = build_dataloader(config, "Train", device, logger, seed)
    
    for aug in config['Train']['dataset']['transforms']:
        name_aug =  list(aug)[0]
        if name_aug == 'NormalizeImage':
            mean = paddle.to_tensor(aug['NormalizeImage']['mean'])
            std = paddle.to_tensor(aug['NormalizeImage']['std'])
            print(f'NormalizeImage mean: {mean}, std: {std}')

    visualize(train_dataloader,'train',save_folder='/work/21013187/phuoc/paddle_detect/data/visualize',mean = mean,std = std)


 

if __name__ == "__main__":
    config, device, logger, vdl_writer = program.preprocess(is_train=True)
    seed = config["Global"]["seed"] if "seed" in config["Global"] else 1024
    set_seed(seed)
    main(config, device, logger, vdl_writer, seed)
    # test_reader(config, device, logger)
