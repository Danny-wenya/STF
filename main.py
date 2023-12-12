import os
import json
import torch
import numpy as np
from data.dataset.stfgnn_dataset import STFGNNDataset
from model.STFGNN import STFGNN
from executor.multi_step_executor import MultiStepExecutor as STFGNNExecutor


def main(config):
    dataset = STFGNNDataset(config)
    train_data, valid_data, test_data = dataset.get_data()
    data_feature = dataset.get_data_feature()
    logging.basicConfig(filename="./log/dataset_info.log",level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    model = STFGNN(config, data_feature)
    executor = STFGNNExecutor(config, model)
    train = True #标识是否需要重新训练

    if train or not os.path.exists(config["model_cache_file"]):
        executor.load_model(config["model_cache_file"]) # [250,260,270,280,290,300,310,320,330,350 ]
        executor.train(train_data, valid_data)
        executor.save_model(config["model_cache_file"])
    else:
        executor.load_model(config["model_cache_file"])
    # 评估，评估结果将会放在 cache/evaluate_cache 下
    executor.evaluate(test_data)


import glob
import re
import logging

if __name__=="__main__":
    config = {}
    for filename in ["config/CITYFLOW.json", "config/STFGNN_CITYFLOW.json"]:
        with open(filename, "r") as f:
            _config = json.load(f)
            for key in _config:
                if key not in config:
                    config[key] = _config[key]

    main(config)


