from collections import defaultdict
import src.utility as utility
import pandas as pd
import numpy as np
from src.Imputation import ImputeMissingValues

class FeatureEngineering:

    def __init__(self, data):
        self.data_dict_baseline = defaultdict(list)
        self.data_dict_recent = defaultdict(list)
        self.cols = list(data.columns)
        self.imp = ImputeMissingValues()
        self._preprocess(data)

    def eid(self, data):
        select_cols = filter(lambda x: x.startswith('eid'), self.cols)
        df_filtered = data[select_cols]
        utility.add_baseline(self.data_dict_baseline, df_filtered, "eid", "eid", "index")
        utility.add_recent(self.data_dict_recent, df_filtered, "eid", "eid", "index")


    def long_standing_illness(self, data):
        select_cols = filter(lambda x: x.startswith('2188'), self.cols)
        df_filtered = data[select_cols]
        recent, baseline = utility.multiple_instance_one_array(df_filtered,4,1,"2188")
        imputed_val = -1
        recent, baseline = self.imp.fill_na_with_number(pd.Series(recent), baseline, imputed_val)
        utility.add_baseline(self.data_dict_baseline, baseline, "long_standing_illness", "2188", "categorical")
        utility.add_recent(self.data_dict_recent, recent, "long_standing_illness", "2188", "categorical")

    def other_prescriptions(self, data):
        select_cols = filter(lambda x: x.startswith('2492'), self.cols)
        df_filtered = data[select_cols]
        recent, baseline = utility.multiple_instance_one_array(df_filtered, 4, 1, "2492")
        imputed_val = -1
        recent, baseline = self.imp.fill_na_with_number(pd.Series(recent), baseline, imputed_val)
        utility.add_baseline(self.data_dict_baseline, baseline, "other_prescriptions", "2492", "categorical")
        utility.add_recent(self.data_dict_recent, recent, "other_prescriptions", "2492", "categorical")

    def overall_health_rating(self, data):
        select_cols = filter(lambda x: x.startswith('2178'), self.cols)
        df_filtered = data[select_cols]
        recent, baseline = utility.multiple_instance_one_array(df_filtered, 4, 1, "2178")
        imputed_val = -1
        recent, baseline = self.imp.fill_na_with_number(pd.Series(recent), baseline, imputed_val)
        utility.add_baseline(self.data_dict_baseline, baseline, "overall_health_rating", "2178", "categorical")
        utility.add_recent(self.data_dict_recent, recent, "overall_health_rating", "2178", "categorical")

    def number_of_treatments_taken(self, data):
        select_cols = filter(lambda x: x.startswith('137'), self.cols)
        df_filtered = data[select_cols]
        recent, baseline = utility.multiple_instance_one_array(df_filtered, 4, 1, "137")
        imputed_val = 0
        recent, baseline = self.imp.fill_na_with_number(pd.Series(recent), baseline, imputed_val)
        utility.add_baseline(self.data_dict_baseline, baseline, "number_of_treatments_taken", "137", "numerical")
        utility.add_recent(self.data_dict_recent, recent, "number_of_treatments_taken", "137", "numerical")




    def age(self, data):
        select_cols = filter(lambda x: x.startswith('21003'), self.cols)
        df_filtered = data[select_cols]
        recent, baseline = utility.multiple_instance_one_array(df_filtered, 4, 1, "21003")
        recent, baseline = self.imp.fill_na_with_average(pd.Series(recent), baseline)
        utility.add_baseline(self.data_dict_baseline, baseline, "age", "21003", "numerical")
        utility.add_recent(self.data_dict_recent, recent, "age", "21003", "numerical")

    def bmi(self, data):
        select_cols = filter(lambda x: x.startswith('21001'), self.cols)
        df_filtered = data[select_cols]
        recent, baseline = utility.multiple_instance_one_array(df_filtered, 4, 1, "21001")
        recent, baseline = self.imp.fill_na_with_average(pd.Series(recent), baseline)
        utility.add_baseline(self.data_dict_baseline, baseline, "bmi", "21001", "numerical")
        utility.add_recent(self.data_dict_recent, recent, "bmi", "21001", "numerical")

    def ethnicity(self, data):
        select_cols = filter(lambda x: x.startswith('21000'), self.cols)
        df_filtered = data[select_cols]
        recent, baseline = utility.multiple_instance_one_array(df_filtered, 3, 1, "21000")
        imputed_val = -1
        recent, baseline = self.imp.fill_na_with_number(pd.Series(recent), baseline, imputed_val)
        utility.add_baseline(self.data_dict_baseline, baseline, "ethnicity", "21000", "categorical")
        utility.add_recent(self.data_dict_recent, recent, "ethnicity", "21000", "categorical")

    def glucose(self, data):
        select_cols = filter(lambda x: x.startswith('30740'), self.cols)
        df_filtered = data[select_cols]
        recent, baseline = utility.multiple_instance_one_array(df_filtered, 2, 1, "30740")
        recent, baseline = self.imp.fill_na_with_average(pd.Series(recent), baseline)
        utility.add_baseline(self.data_dict_baseline, baseline, "glucose", "30740", "numerical")
        utility.add_recent(self.data_dict_recent, recent, "glucose", "30740", "numerical")

    def smoking(self, data):
        select_cols = filter(lambda x: x.startswith('20116'), self.cols)
        df_filtered = data[select_cols]
        recent, baseline = utility.multiple_instance_one_array(df_filtered, 4, 1, "20116")
        imputed_val = -3
        recent, baseline = self.imp.fill_na_with_number(pd.Series(recent), baseline, imputed_val)
        utility.add_baseline(self.data_dict_baseline, baseline, "smoking", "20116", "categorical")
        utility.add_recent(self.data_dict_recent, recent, "smoking", "20116", "categorical")

    def family_history_ibs(self, data):
        select_cols = filter(lambda x: x.startswith('21065'), self.cols)
        df_filtered = data[select_cols]
        recent, baseline = utility.multiple_instance_one_array(df_filtered, 1, 1, "21065")
        imputed_val = -121
        recent, baseline = self.imp.fill_na_with_number(pd.Series(recent), baseline, imputed_val)
        utility.add_baseline(self.data_dict_baseline, baseline, "family_history_ibs", "21065", "categorical")
        utility.add_recent(self.data_dict_recent, recent, "family_history_ibs", "21065", "categorical")

    def major_diet_change_last_5_years(self, data):
        select_cols = filter(lambda x: x.startswith('1538'), self.cols)
        df_filtered = data[select_cols]
        recent, baseline = utility.multiple_instance_one_array(df_filtered, 4, 1, "1538")
        imputed_val = -3
        recent, baseline = self.imp.fill_na_with_number(pd.Series(recent), baseline, imputed_val)
        utility.add_baseline(self.data_dict_baseline, baseline, "major_diet_change_last_5_years", "1538", "categorical")
        utility.add_recent(self.data_dict_recent, recent, "major_diet_change_last_5_years", "1538", "categorical")

    def gender(self, data):
        select_cols = filter(lambda x: x.startswith('31'), self.cols)
        df_filtered = data[select_cols]
        recent, baseline = utility.multiple_instance_one_array(df_filtered, 1, 1, "31")
        imputed_val = -1
        recent, baseline = self.imp.fill_na_with_number(pd.Series(recent), baseline, imputed_val)
        utility.add_baseline(self.data_dict_baseline, baseline, "gender", "31", "categorical")
        utility.add_recent(self.data_dict_recent, recent, "gender", "31", "categorical")

    def other_major_operations(self, data):
        select_cols = filter(lambda x: x.startswith('2844'), self.cols)
        df_filtered = data[select_cols]
        recent, baseline = utility.multiple_instance_one_array(df_filtered, 4, 1, "2844")
        imputed_val = -1
        recent, baseline = self.imp.fill_na_with_number(pd.Series(recent), baseline, imputed_val)
        utility.add_baseline(self.data_dict_baseline, baseline, "other_major_operations", "2844", "categorical")
        utility.add_recent(self.data_dict_recent, recent, "other_major_operations", "2844", "categorical")

    def weight(self, data):
        select_cols = filter(lambda x: x.startswith('21002'), self.cols)
        df_filtered = data[select_cols]
        recent, baseline = utility.multiple_instance_one_array(df_filtered, 4, 1, "21002")
        recent, baseline = self.imp.fill_na_with_average(pd.Series(recent), baseline)
        utility.add_baseline(self.data_dict_baseline, baseline, "weight", "21002", "numerical")
        utility.add_recent(self.data_dict_recent, recent, "weight", "21002", "numerical")

    def whole_body_fat_mass(self, data):
        select_cols = filter(lambda x: x.startswith('23100'), self.cols)
        df_filtered = data[select_cols]
        recent, baseline = utility.multiple_instance_one_array(df_filtered, 4, 1, "23100")
        recent, baseline = self.imp.fill_na_with_average(pd.Series(recent), baseline)
        utility.add_baseline(self.data_dict_baseline, baseline, "whole_body_fat_mass", "23100", "numerical")
        utility.add_recent(self.data_dict_recent, recent, "whole_body_fat_mass", "23100", "numerical")

    def body_fat_per(self, data):
        select_cols = filter(lambda x: x.startswith('23099'), self.cols)
        df_filtered = data[select_cols]
        recent, baseline = utility.multiple_instance_one_array(df_filtered, 4, 1, "23099")
        recent, baseline = self.imp.fill_na_with_average(pd.Series(recent), baseline)
        utility.add_baseline(self.data_dict_baseline, baseline, "body_fat_per", "23099", "numerical")
        utility.add_recent(self.data_dict_recent, recent, "body_fat_per", "23099", "numerical")

    def hip_circum(self, data):
        select_cols = filter(lambda x: x.startswith('49'), self.cols)
        df_filtered = data[select_cols]
        recent, baseline = utility.multiple_instance_one_array(df_filtered, 4, 1, "49")
        recent, baseline = self.imp.fill_na_with_average(pd.Series(recent), baseline)
        utility.add_baseline(self.data_dict_baseline, baseline, "hip_circum", "49", "numerical")
        utility.add_recent(self.data_dict_recent, recent, "hip_circum", "49", "numerical")

    def waist_circum(self, data):
        select_cols = filter(lambda x: x.startswith('48'), self.cols)
        df_filtered = data[select_cols]
        recent, baseline = utility.multiple_instance_one_array(df_filtered, 4, 1, "48")
        recent, baseline = self.imp.fill_na_with_average(pd.Series(recent), baseline)
        utility.add_baseline(self.data_dict_baseline, baseline, "waist_circum", "48", "numerical")
        utility.add_recent(self.data_dict_recent, recent, "waist_circum", "48", "numerical")

    def pregnant(self, data):
        select_cols = filter(lambda x: x.startswith('3140'), self.cols)
        df_filtered = data[select_cols]
        recent, baseline = utility.multiple_instance_one_array(df_filtered, 4, 1, "3140")
        imputed_val = 2
        recent, baseline = self.imp.fill_na_with_number(pd.Series(recent), baseline, imputed_val)
        utility.add_baseline(self.data_dict_baseline, baseline, "pregnant", "3140", "categorical")
        utility.add_recent(self.data_dict_recent, recent, "pregnant", "3140", "categorical")

    def breast_cancer_screening(self, data):
        select_cols = filter(lambda x: x.startswith('2674'), self.cols)
        df_filtered = data[select_cols]
        recent, baseline = utility.multiple_instance_one_array(df_filtered, 4, 1, "2674")
        imputed_val = -1
        recent, baseline = self.imp.fill_na_with_number(pd.Series(recent), baseline, imputed_val)
        utility.add_baseline(self.data_dict_baseline, baseline, "breast_cancer_screening", "2674", "categorical")
        utility.add_recent(self.data_dict_recent, recent, "breast_cancer_screening", "2674", "categorical")

    def diagnose_by_doc(self, data):
        select_cols = filter(lambda x: x.startswith('2473'), self.cols)
        df_filtered = data[select_cols]
        recent, baseline = utility.multiple_instance_one_array(df_filtered, 4, 1, "2473")
        imputed_val = -1
        recent, baseline = self.imp.fill_na_with_number(pd.Series(recent), baseline, imputed_val)
        utility.add_baseline(self.data_dict_baseline, baseline, "diagnose_by_doc", "2473", "categorical")
        utility.add_recent(self.data_dict_recent, recent, "diagnose_by_doc", "2473", "categorical")

    def pulse(self, data):
        select_cols = filter(lambda x: x.startswith('4194'), self.cols)
        df_filtered = data[select_cols]
        recent, baseline = utility.multiple_instance_one_array(df_filtered, 4, 1, "4194")
        recent, baseline = self.imp.fill_na_with_average(pd.Series(recent), baseline)
        utility.add_baseline(self.data_dict_baseline, baseline, "pulse", "4194", "numerical")
        utility.add_recent(self.data_dict_recent, recent, "pulse", "4194", "numerical")

    def standing_height(self, data):
        select_cols = filter(lambda x: x.startswith('50'), self.cols)
        df_filtered = data[select_cols]
        recent, baseline = utility.multiple_instance_one_array(df_filtered, 4, 1, "50")
        recent, baseline = self.imp.fill_na_with_average(pd.Series(recent), baseline)
        utility.add_baseline(self.data_dict_baseline, baseline, "standing_height", "50", "numerical")
        utility.add_recent(self.data_dict_recent, recent, "standing_height", "50", "numerical")

    def frequency_of_excercise(self, data):
        select_cols = filter(lambda x: x.startswith('3637'), self.cols)
        df_filtered = data[select_cols]
        recent, baseline = utility.multiple_instance_one_array(df_filtered, 4, 1, "3637")
        imputed_val = -1
        recent, baseline = self.imp.fill_na_with_number(pd.Series(recent), baseline, imputed_val)
        utility.add_baseline(self.data_dict_baseline, baseline, "frequency_of_excercise", "3637", "categorical")
        utility.add_recent(self.data_dict_recent, recent, "frequency_of_excercise", "3637", "categorical")

    def fractured_bone_last_5_years(self, data):
        select_cols = filter(lambda x: x.startswith('2463'), self.cols)
        df_filtered = data[select_cols]
        recent, baseline = utility.multiple_instance_one_array(df_filtered, 4, 1, "2463")
        imputed_val = -1
        recent, baseline = self.imp.fill_na_with_number(pd.Series(recent), baseline, imputed_val)
        utility.add_baseline(self.data_dict_baseline, baseline, "fractured_bone_last_5_years", "2463", "categorical")
        utility.add_recent(self.data_dict_recent, recent, "fractured_bone_last_5_years", "2463", "categorical")

    def job_involves_walking_standing(self, data):
        select_cols = filter(lambda x: x.startswith('806'), self.cols)
        df_filtered = data[select_cols]
        recent, baseline = utility.multiple_instance_one_array(df_filtered, 4, 1, "806")
        imputed_val = -1
        recent, baseline = self.imp.fill_na_with_number(pd.Series(recent), baseline, imputed_val)
        utility.add_baseline(self.data_dict_baseline, baseline, "job_involves_walking_standing", "806", "categorical")
        utility.add_recent(self.data_dict_recent, recent, "job_involves_walking_standing", "806", "categorical")

    def diabetes_diagonosed_by_doc(self, data):
        select_cols = filter(lambda x: x.startswith('2443'), self.cols)
        df_filtered = data[select_cols]
        recent, baseline = utility.multiple_instance_one_array(df_filtered, 4, 1, "2443")
        imputed_val = -1
        recent, baseline = self.imp.fill_na_with_number(pd.Series(recent), baseline, imputed_val)
        utility.add_baseline(self.data_dict_baseline, baseline, "diabetes_diagonosed_by_doc", "2443", "categorical")
        utility.add_recent(self.data_dict_recent, recent, "diabetes_diagonosed_by_doc", "2443", "categorical")

    def amount_of_alcohol(self, data):
        select_cols = filter(lambda x: x.startswith('20403'), self.cols)
        df_filtered = data[select_cols]
        recent, baseline = utility.multiple_instance_one_array(df_filtered, 1, 1, "20403")
        imputed_val = -818
        recent, baseline = self.imp.fill_na_with_number(pd.Series(recent), baseline, imputed_val)
        utility.add_baseline(self.data_dict_baseline, baseline, "amout_of_alcohol", "20403", "categorical")
        utility.add_recent(self.data_dict_recent, recent, "amout_of_alcohol", "20403", "categorical")

    def occurrence_of_cancer(self, data):
        select_cols = filter(lambda x: x.startswith('40009'), self.cols)
        df_filtered = data[select_cols]
        recent, baseline = utility.multiple_instance_one_array(df_filtered, 1, 1, "40009")
        imputed_val = -1
        recent, baseline = self.imp.fill_na_with_number(pd.Series(recent), baseline, imputed_val)
        utility.add_baseline(self.data_dict_baseline, baseline, "occurrence_of_cancer", "40009", "numerical")
        utility.add_recent(self.data_dict_recent, recent, "occurrence_of_cancer", "40009", "numerical")

    def number_of_self_reporter_non_cancer_codes(self, data):
        select_cols = filter(lambda x: x.startswith('135'), self.cols)
        df_filtered = data[select_cols]
        recent, baseline = utility.multiple_instance_one_array(df_filtered, 4, 1, "135")
        imputed_val = -1
        recent, baseline = self.imp.fill_na_with_number(pd.Series(recent), baseline, imputed_val)
        utility.add_baseline(self.data_dict_baseline, baseline, "number_of_self_reporter_non_cancer_codes", "135", "numerical")
        utility.add_recent(self.data_dict_recent, recent, "number_of_self_reporter_non_cancer_codes", "135", "numerical")

    def job_involves_heavy_physical_work(self, data):
        select_cols = filter(lambda x: x.startswith('816'), self.cols)
        df_filtered = data[select_cols]
        recent, baseline = utility.multiple_instance_one_array(df_filtered, 4, 1, "816")
        imputed_val = -1
        recent, baseline = self.imp.fill_na_with_number(pd.Series(recent), baseline, imputed_val)
        utility.add_baseline(self.data_dict_baseline, baseline, "job_involves_heavy_physical_work", "816", "numerical")
        utility.add_recent(self.data_dict_recent, recent, "job_involves_heavy_physical_work", "816", "numerical")

    def age_cancer_diag(self, data):
        select_cols = filter(lambda x: x.startswith('40008'), self.cols)
        df_filtered = data[select_cols]
        recent, baseline = utility.columns_with_lowest_val(df_filtered, '40008')
        imputed_val = -1
        recent, baseline = self.imp.fill_na_with_number(pd.Series(recent), baseline, imputed_val)
        utility.add_baseline(self.data_dict_baseline, baseline, "age_cancer_diag", "40008", "numerical")
        utility.add_recent(self.data_dict_recent, recent, "age_cancer_diag", "40008",
                           "numerical")

    def age_diabetes_diag(self, data):
        select_cols = filter(lambda x: x.startswith('2976'), self.cols)
        df_filtered = data[select_cols]
        recent, baseline = utility.columns_with_lowest_val(df_filtered, '2976')
        imputed_val = -1
        recent, baseline = self.imp.fill_na_with_number(pd.Series(recent), baseline, imputed_val)
        utility.add_baseline(self.data_dict_baseline, baseline, "age_diabetes_diag", "2976", "numerical")
        utility.add_recent(self.data_dict_recent, recent, "age_diabetes_diag", "2976",
                           "numerical")

    def age_at_chronic_pulmonary_disease(self, data):

        ## Convert to age
        select_cols = filter(lambda x: x.startswith('42016'), self.cols)
        df_filtered = data[select_cols]
        recent, baseline = utility.columns_with_lowest_val(df_filtered, '42016')

        # select_cols = filter(lambda x: x.startswith('33'), self.cols)
        # df_filtered = data[select_cols]
        # recent_dob, baseline_dob = utility.multiple_instance_one_array(df_filtered, 1, 1, "33")

        imputed_val = -1
        recent, baseline = self.imp.fill_na_with_number(pd.Series(recent), baseline, imputed_val)
        # recent_dob, baseline_dob = self.imp.fill_na_with_number(pd.Series(recent_dob), baseline_dob, imputed_val)
        #
        # recent = (recent - recent_dob) / np.timedelta64(1, 'Y')
        # baseline = (baseline - baseline_dob) / np.timedelta64(1, 'Y')

        utility.add_baseline(self.data_dict_baseline, baseline, "age_at_chronic_pulmonary_disease", "42016", "numerical")
        utility.add_recent(self.data_dict_recent, recent, "age_at_chronic_pulmonary_disease", "42016",
                           "numerical")

    def heart_prob_diag_by_doc(self, data):

        coding_df = pd.read_csv("UK_biobank_codings/single_coding/6150.tsv", sep="\t")
        select_cols = filter(lambda x: x.startswith('6150'), self.cols)
        df_filtered = data[select_cols]
        imputed_val = -1
        df_filtered = self.imp.fill_df(df_filtered, imputed_val)
        df_recent, df_baseline = utility.category_to_multiple_features(df_filtered, coding_df, 4, "6150")

        utility.add_baseline(self.data_dict_baseline, df_baseline, "heart_prob_diag_by_doc", "6150",
                             "dataframe")
        utility.add_recent(self.data_dict_recent, df_recent, "heart_prob_diag_by_doc", "6150",
                             "dataframe")

    def cholesterol_lowering_medication_females(self, data):
        coding_df = pd.read_csv("UK_biobank_codings/single_coding/6153.tsv", sep="\t")
        select_cols = filter(lambda x: x.startswith('6153'), self.cols)
        df_filtered = data[select_cols]
        imputed_val = -1
        df_filtered = self.imp.fill_df(df_filtered, imputed_val)
        df_recent, df_baseline = utility.category_to_multiple_features(df_filtered, coding_df, 4, "6153")
        utility.add_baseline(self.data_dict_baseline, df_baseline, "cholesterol_lowering_medication_females", "6153",
                             "dataframe")
        utility.add_recent(self.data_dict_recent, df_recent, "cholesterol_lowering_medication_females", "6153",
                             "dataframe")

    def cholesterol_lowering_medication_males(self, data):
        coding_df = pd.read_csv("UK_biobank_codings/single_coding/6177.tsv", sep="\t")
        select_cols = filter(lambda x: x.startswith('6177'), self.cols)
        df_filtered = data[select_cols]
        imputed_val = -1
        df_filtered = self.imp.fill_df(df_filtered, imputed_val)
        df_recent, df_baseline = utility.category_to_multiple_features(df_filtered, coding_df, 4, "6177")
        utility.add_baseline(self.data_dict_baseline, df_baseline, "cholesterol_lowering_medication_males", "6177",
                             "dataframe")
        utility.add_recent(self.data_dict_recent, df_recent, "cholesterol_lowering_medication_males", "6177",
                             "dataframe")

    def vitamin_mineral_supp(self, data):
        coding_df = pd.read_csv("UK_biobank_codings/single_coding/6155.tsv", sep="\t")
        select_cols = filter(lambda x: x.startswith('6155'), self.cols)
        df_filtered = data[select_cols]
        imputed_val = -7
        df_filtered = self.imp.fill_df(df_filtered, imputed_val)
        df_recent, df_baseline = utility.category_to_multiple_features(df_filtered, coding_df, 4, "6155")
        utility.add_baseline(self.data_dict_baseline, df_baseline, "vitamin_mineral_supp", "6155",
                             "dataframe")
        utility.add_recent(self.data_dict_recent, df_recent, "vitamin_mineral_supp", "6155",
                             "dataframe")

    def mineral_supp(self, data):
        coding_df = pd.read_csv("UK_biobank_codings/single_coding/6179.tsv", sep="\t")
        select_cols = filter(lambda x: x.startswith('6179'), self.cols)
        df_filtered = data[select_cols]
        imputed_val = -7
        df_filtered = self.imp.fill_df(df_filtered, imputed_val)
        df_recent, df_baseline = utility.category_to_multiple_features(df_filtered, coding_df, 4, "6179")
        utility.add_baseline(self.data_dict_baseline, df_baseline, "mineral_supp", "6179",
                             "dataframe")
        utility.add_recent(self.data_dict_recent, df_recent, "mineral_supp", "6179",
                             "dataframe")

    def hospital_episode_type(self, data):
        coding_df = pd.read_csv("UK_biobank_codings/single_coding/41231.tsv", sep="\t")
        select_cols = filter(lambda x: x.startswith('41231'), self.cols)
        df_filtered = data[select_cols]
        imputed_val = -1
        df_filtered = self.imp.fill_df(df_filtered, imputed_val)
        df_recent, df_baseline = utility.category_to_multiple_features(df_filtered, coding_df, 3, "41231")
        utility.add_baseline(self.data_dict_baseline, df_baseline, "hospital_episode_type", "41231",
                             "dataframe")
        utility.add_recent(self.data_dict_recent, df_recent, "hospital_episode_type", "41231",
                             "dataframe")

    def non_cancer_illness_code(self, data):
        coding_df = pd.read_csv("UK_biobank_codings/node_parent/20002.tsv", sep="\t")
        select_cols = filter(lambda x: x.startswith('20002'), self.cols)
        df_filtered = data[select_cols]
        imputed_val = -1
        df_filtered = self.imp.fill_df(df_filtered, imputed_val)
        df_recent, df_baseline = utility.category_to_multiple_features_parent(df_filtered, coding_df, 34, "20002")
        utility.add_baseline(self.data_dict_baseline, df_baseline, "non_cancer_illness_code", "20002",
                             "dataframe")
        utility.add_recent(self.data_dict_recent, df_recent, "non_cancer_illness_code", "20002",
                             "dataframe")

    def operation_code(self, data):
        coding_df = pd.read_csv("UK_biobank_codings/node_parent/20004.tsv", sep="\t")
        select_cols = filter(lambda x: x.startswith('20004'), self.cols)
        df_filtered = data[select_cols]
        imputed_val = -1
        df_filtered = self.imp.fill_df(df_filtered, imputed_val)
        df_recent, df_baseline = utility.category_to_multiple_features_parent(df_filtered, coding_df, 32, "20004")
        utility.add_baseline(self.data_dict_baseline, df_baseline, "operation_code", "20004",
                             "dataframe")
        utility.add_recent(self.data_dict_recent, df_recent, "operation_code", "20004",
                             "dataframe")

    def forced_vital_capacity(self, data):
        select_cols = filter(lambda x: x.startswith('3062'), self.cols)
        df_filtered = data[select_cols]
        recent, baseline = utility.columns_with_lowest_val(df_filtered, '3062')
        recent, baseline = self.imp.fill_na_with_average(pd.Series(recent), baseline)
        utility.add_baseline(self.data_dict_baseline, baseline, "forced_vital_capacity", "3062",
                             "numerical")
        utility.add_recent(self.data_dict_recent, recent, "forced_vital_capacity", "3062",
                           "numerical")

    def systolic_blood_pressure(self, data):
        select_cols = filter(lambda x: x.startswith('4080'), self.cols)
        df_filtered = data[select_cols]
        recent, baseline = utility.columns_with_highest_val(df_filtered, '4080')
        recent, baseline = self.imp.fill_na_with_average(pd.Series(recent), baseline)
        utility.add_baseline(self.data_dict_baseline, baseline, "systolic_blood_pressure", "4080",
                             "numerical")
        utility.add_recent(self.data_dict_recent, recent, "systolic_blood_pressure", "4080",
                           "numerical")

    def _preprocess(self, data):

        utility.call_all_public_function_within_obj(self, data)






