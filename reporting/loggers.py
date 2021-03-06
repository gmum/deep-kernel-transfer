import neptune.new as neptune
import numpy as np
from copy import deepcopy

PROJECT_NAME = "" #add here your project name 
from neptune.new.types import File


class ResultsLogger:
    def __init__(self, params):
        self.params = params
        if self.params.neptune:
            self.neptune_logger = NeptuneLogger(params)
        self.local_logger = LocalLogger()

    def log(self, key, value):
        self.local_logger.log(key, value)
        if self.params.neptune:
            self.neptune_logger.log(key, value)    
        

    def log_dict(self, dict):
        self.local_logger.log_dict(dict)
        if self.params.neptune:
            self.neptune_logger.log_dict(dict)
        

    def get_array(self, key):
        return self.local_logger.get_array(key)

    def save(self):
        suffix = "test_results.npy" if self.params.test else "train_results.npy"
        if self.params.out_of_range:
            suffix="outr_{}".format(suffix)
        save_file = '%s/checkpoints/%s/%s_%s_%s' % (
            self.params.save_dir, self.params.dataset, self.params.model, self.params.method, suffix)
        np.save(save_file, self.local_logger.results_dict)


class NeptuneLogger:
    def __init__(self, params):
        try:
           self.run = neptune.init(PROJECT_NAME)
        except neptune.new.exceptions.NeptuneConnectionLostException:
           self.run = neptune.init(PROJECT_NAME, mode="offline")
        self.run["parameters"] = vars(params)

    def log_dict(self, dict):
        for key, value in dict.items():
            self.log(key, value)

    def log(self, key, v):
        value = deepcopy(v)
        if isinstance(value, np.ndarray) and len(value.shape) == 0:
            value = float(value)
        elif isinstance(value, np.ndarray) and len(value.shape) == 1:
            value = File.as_image(np.array([value]))
        elif isinstance(value, np.ndarray) and len(value.shape) <= 3:
            value = File.as_image(value)
        elif isinstance(value, np.ndarray):
            return
        self.run[key].log(value)


class LocalLogger:
    def __init__(self):
        self.results_dict = {}

    def log_dict(self, dict):
        for key, value in dict.items():
            self.log(key, value)

    def log(self, key, value):
        if key not in self.results_dict:
            self.results_dict[key] = []
        self.results_dict[key].append(value)
        

    def get_array(self, key):
        if key not in self.results_dict:
            return []
        return self.results_dict[key]
