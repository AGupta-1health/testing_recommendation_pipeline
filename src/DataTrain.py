import pandas as pd


class DataTrain:

    def __init__(self, baseline_dict, recent_dict):
        self.baseline_dict = baseline_dict
        self.recent_dict = recent_dict
        self.baseline_training_data = self.features_to_training_data(self.baseline_dict)
        self.recent_training_data = self.features_to_training_data(self.recent_dict)

    def features_to_training_data(self, data_dict):

        training_data = pd.DataFrame()
        for key,val in data_dict.items():
            training_data = pd.concat([training_data, val[0]], axis=1)

        return training_data








