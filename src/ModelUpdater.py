
from collections import defaultdict
from src.FeatureEngineering import FeatureEngineering
from src.DataTrain import DataTrain
from src.LabelGeneration import LabelGeneration
from src.Modeling.VitaminD import VitaminD
from src.Modeling.HbA1c import HbA1c
from src.Modeling.Metabolism import Metabolism
from src.Modeling.Inflammation import Inflammation
from src.Modeling.Thyroid import Thyroid
from src.Modeling.Food_Sentivity import Food_Senstivity
from src.Modeling.Heart_Health import Heart_Health

from src.Modeling.train import Train


class modelUpdater:

    def __init__(self,data, questionnaire_fts):
        self.data = data
        self.questionnaire_fts = questionnaire_fts
        self.pipeline()

    def feature_engineering(self):
        feature_eng = FeatureEngineering(self.data)
        return feature_eng

    def label_generation(self):
        labels = LabelGeneration(self.data)
        return labels

    def training_data(self, feature_eng):
        training_data = DataTrain(feature_eng.data_dict_baseline, feature_eng.data_dict_recent)
        return training_data

    def modeling(self, train_data, labels):
        all_cols = list(train_data.recent_training_data.columns)
        train1 = Train()

        metabolism = Metabolism()  ## Correct 42016
        final_data_metabolism = metabolism.combine_data_labels(train_data, labels, self.questionnaire_fts, all_cols)
        train1.training(data=final_data_metabolism, label="metabolism_l",
                        parameters={"class_weight": 53.81, "objective_choice": "binary:logistic",
                                    "eval_metric_choice": 'auc'},
                        model_name="XG", experiment_name="Metabolism")  # 70

        vitaminD = VitaminD()
        final_data_vitamind = vitaminD.combine_data_labels(train_data, labels, self.questionnaire_fts, all_cols)
        train1.training(data=final_data_vitamind, label="vitamind_l",
                        parameters={"n_estimators": 500, "class_weight": "balanced_subsample", "max_samples": 0.25,
                                    "random_state": 0, "oob_score": True, "min_samples_leaf": 5, "max_depth": 10},
                        model_name="RF",experiment_name="VitaminD")

        hbA1c = HbA1c() ## Correct Labels
        final_data_a1c = hbA1c.combine_data_labels(train_data, labels, self.questionnaire_fts, all_cols)
        train1.training(data=final_data_a1c, label="hba1c_l",
                        parameters={"class_weight" : 4.12, "objective_choice" : "binary:logistic", "eval_metric_choice" : 'auc'},
                        model_name="XG",experiment_name="HbA1c") #6


        inflammation = Inflammation()
        final_data_inflammation = inflammation.combine_data_labels(train_data, labels, self.questionnaire_fts, all_cols)
        train1.training(data=final_data_inflammation, label="inflammation_l",
                        parameters={"class_weight": 23.11, "objective_choice": "binary:logistic",
                                    "eval_metric_choice": 'auc'},
                        model_name="XG", experiment_name="inflammation") #8.44

        thyroid = Thyroid()
        final_data_thyroid = thyroid.combine_data_labels(train_data, labels, self.questionnaire_fts, all_cols)
        train1.training(data=final_data_thyroid, label="thyroid_l",
                        parameters={"class_weight": 14.24, "objective_choice": "binary:logistic",
                                    "eval_metric_choice": 'auc'},
                        model_name="XG", experiment_name="thyroid") #13.7

        food_senstivity = Food_Senstivity()
        final_data_food = food_senstivity.combine_data_labels(train_data, labels, self.questionnaire_fts, all_cols)
        train1.training(data=final_data_food, label="food_sens_l",
                        parameters={"class_weight": 31.51, "objective_choice": "binary:logistic",
                                    "eval_metric_choice": 'auc'},
                        model_name="XG", experiment_name="food_senstivity") #8.688

        heart_health = Heart_Health()
        final_data_tri, final_data_hdl, final_data_ldl, final_data_chol, final_data_crp = heart_health.combine_data_labels(train_data, labels, self.questionnaire_fts, all_cols)
        train1.training(data=final_data_tri, label="tri_l",
                        parameters={"class_weight": 1.47, "objective_choice": "binary:logistic",
                                    "eval_metric_choice": 'auc'},
                        model_name="XG", experiment_name="triglycerides")  # 0.60

        train1.training(data=final_data_ldl, label="ldl_l",
                        parameters={"class_weight": 0.14, "objective_choice": "binary:logistic",
                                    "eval_metric_choice": 'auc'},
                        model_name="XG", experiment_name="ldl")  # 0.14

        train1.training(data=final_data_hdl, label="hdl_l",
                        parameters={"class_weight": 3.61, "objective_choice": "binary:logistic",
                                    "eval_metric_choice": 'auc'},
                        model_name="XG", experiment_name="hdl")  # 0.72

        train1.training(data=final_data_chol, label="chol_l",
                        parameters={"class_weight": 1.05, "objective_choice": "binary:logistic",
                                    "eval_metric_choice": 'auc'},
                        model_name="XG", experiment_name="chol")  # 0.56

        train1.training(data=final_data_crp, label="creac_l",
                        parameters={"class_weight": 3.38, "objective_choice": "binary:logistic",
                                    "eval_metric_choice": 'auc'},
                        model_name="XG", experiment_name="creac")  # 0.77


    def pipeline(self):
        feature_eng = self.feature_engineering()
        labels = self.label_generation()
        train_data = self.training_data(feature_eng)
        self.modeling(train_data,labels)








