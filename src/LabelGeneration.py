import src.utility as utility
import pandas as pd
import numpy as np


class LabelGeneration:

    def __init__(self, data):

        self.cols = list(data.columns)
        self.labels_dict = {}
        self.generate_labels(data)

    def vitamind(self, data):
        select_cols = filter(lambda x: x.startswith('30890'), self.cols)
        df_filtered = data[select_cols]
        recent, baseline = utility.multiple_instance_one_array(df_filtered,2,1,"30890")
        recent = recent/2.496
        baseline = baseline/2.496
        final_labels = data[["eid"]]
        final_labels["vitamind"] = recent
        ## remove nulls
        final_labels.dropna(subset=["vitamind"], inplace=True)
        final_labels["vitamind_l"] = 1
        final_labels["vitamind_l"] = np.where(final_labels["vitamind"] > 29, 0, final_labels["vitamind_l"])
        final_labels.drop(columns=["vitamind"], inplace= True)
        self.labels_dict["vitaminD"] = final_labels

        print("Vitamin D", final_labels["vitamind_l"].value_counts())

    def hba1c(self, data):
        select_cols = filter(lambda x: x.startswith('30750'), self.cols)
        df_filtered = data[select_cols]
        recent, baseline = utility.multiple_instance_one_array(df_filtered, 2, 1, "30750")
        recent = (recent * 0.0915) + 2.15
        baseline = (baseline * 0.0915) + 2.15
        final_labels = data[["eid"]]
        final_labels["hba1c"] = recent
        final_labels.dropna(subset=["hba1c"], inplace=True)
        final_labels["hba1c_l"] = 0
        final_labels["hba1c_l"] = np.where(final_labels["hba1c"] > 5.7, 1, final_labels["hba1c_l"])
        final_labels.drop(columns=["hba1c"], inplace= True)
        self.labels_dict["hbA1c"] = final_labels

        print("Hba1c", final_labels["hba1c_l"].value_counts())

    def metabolism(self, data):
        select_cols = filter(lambda x: x.startswith('41270'), self.cols)
        df_filtered = data[select_cols]
        codings = pd.read_excel("UK_biobank_codings/label_coding/Metabolism_codings.xlsx")
        coding_lst = list(codings.coding.values)
        t = df_filtered
        t['combined'] = t.values.tolist()
        for i in range(213):
            t["label_" + str(i)] = np.NAN
            t["label_" + str(i)] = np.where(t['41270-0.' + str(i)].isin(coding_lst), 1, t["label_" + str(i)])
        t.drop(columns=select_cols, inplace=True)
        t.drop(columns=["combined"], inplace=True)
        t["max"] = t.max(axis=1)
        t["max"] = t["max"].fillna(0)
        t["max"].value_counts()
        final_labels = data[["eid"]]
        final_labels["metabolism_l"] = t["max"]
        final_labels.dropna(subset=["metabolism_l"], inplace=True)
        self.labels_dict["metabolism"] = final_labels

        print("Metabolism", final_labels["metabolism_l"].value_counts())

    def triglycerides(self, data):
        select_cols = filter(lambda x: x.startswith('30870'), self.cols)
        df_filtered = data[select_cols]
        recent, baseline = utility.multiple_instance_one_array(df_filtered, 2, 1, "30870")
        recent = recent / 0.01129
        baseline = baseline / 0.01129
        return recent

    def hdl(self, data):
        select_cols = filter(lambda x: x.startswith('30760'), self.cols)
        df_filtered = data[select_cols]
        recent, baseline = utility.multiple_instance_one_array(df_filtered, 2, 1, "30760")
        recent = recent / 0.02586
        baseline = baseline / 0.02586
        return recent

    def ldl(self, data):
        select_cols = filter(lambda x: x.startswith('30780'), self.cols)
        df_filtered = data[select_cols]
        recent, baseline = utility.multiple_instance_one_array(df_filtered, 2, 1, "30780")
        recent = recent / 0.02586
        baseline = baseline / 0.02586
        return recent

    def cholesterol(self, data):
        select_cols = filter(lambda x: x.startswith('30690'), self.cols)
        df_filtered = data[select_cols]
        recent, baseline = utility.multiple_instance_one_array(df_filtered, 2, 1, "30690")
        recent = recent / 0.02586
        baseline = baseline / 0.02586
        return recent

    def cReactive(self, data):
        select_cols = filter(lambda x: x.startswith('30710'), self.cols)
        df_filtered = data[select_cols]
        recent, baseline = utility.multiple_instance_one_array(df_filtered, 2, 1, "30710")
        return recent

    def thyroid(self, data):
        select_cols = filter(lambda x: x.startswith('41270'), self.cols)
        df_filtered = data[select_cols]
        codings = pd.read_excel("UK_biobank_codings/label_coding/Thyroid_codings.xlsx")
        coding_lst = list(codings.coding.values)
        t = df_filtered
        t['combined'] = t.values.tolist()
        for i in range(213):
            t["label_" + str(i)] = np.NAN
            t["label_" + str(i)] = np.where(t['41270-0.' + str(i)].isin(coding_lst), 1, t["label_" + str(i)])
        t.drop(columns=select_cols, inplace=True)
        t.drop(columns=["combined"], inplace=True)
        t["max"] = t.max(axis=1)
        t["max"] = t["max"].fillna(0)
        t["max"].value_counts()
        final_labels = data[["eid"]]
        final_labels["thyroid_l"] = t["max"]
        final_labels.dropna(subset=["thyroid_l"], inplace=True)
        self.labels_dict["thyroid"] = final_labels

        print("Thyroid", final_labels["thyroid_l"].value_counts())

    def food_senstivity(self, data):
        select_cols = filter(lambda x: x.startswith('41270'), self.cols)
        df_filtered = data[select_cols]
        codings = pd.read_excel("UK_biobank_codings/label_coding/Food_Senstivity_codings.xlsx")
        coding_lst = list(codings.coding.values)
        t = df_filtered
        t['combined'] = t.values.tolist()
        for i in range(213):
            t["label_" + str(i)] = np.NAN
            t["label_" + str(i)] = np.where(t['41270-0.' + str(i)].isin(coding_lst), 1, t["label_" + str(i)])
        t.drop(columns=select_cols, inplace=True)
        t.drop(columns=["combined"], inplace=True)
        t["max"] = t.max(axis=1)
        t["max"] = t["max"].fillna(0)
        t["max"].value_counts()
        final_labels = data[["eid"]]
        final_labels["food_sens_l"] = t["max"]
        final_labels.dropna(subset=["food_sens_l"], inplace=True)
        self.labels_dict["food_senstivity"] = final_labels

        print("food senstivity",final_labels["food_sens_l"].value_counts())

    def heart_health(self, data):
        final = data[["eid"]]
        final["tri"] = self.triglycerides(data)
        final["hdl"] = self.hdl(data)
        final["ldl"] = self.ldl(data)
        final["chol"] = self.cholesterol(data)
        final["creac"] = self.cReactive(data)
        final["gender"] = data["31-0.0"]

        final.dropna(subset=["tri", "ldl", "chol", "hdl", "creac"],inplace=True)

        final["tri_l"] = 0
        final["ldl_l"] = 0
        final["chol_ratio"] = 0
        final["chol_l"] = 0
        final["hdl_l"] = 0
        final["creac_l"] = 0

        final["tri_l"] = np.where(final["tri"] > 150, 1, final["tri_l"])
        final["ldl_l"] = np.where(final["ldl"] > 100, 1, final["ldl_l"])
        final["chol_ratio"] = final["chol"] / final["hdl"]
        final["chol_l"] = np.where(final["chol_ratio"] > 4, 1, final["chol_l"])

        final["hdl_l"] = np.where((((final["hdl"] < 40) & (final["gender"] == 1)) | (
                    (final["hdl"] < 50) & (final["gender"] == 0))), 1, final["hdl_l"])

        final["creac_l"] = np.where(final["creac"] > 3, 1, final["creac_l"])

        final_labels = final[["eid","tri_l", "ldl_l", "chol_l", "hdl_l", "creac_l"]]

        self.labels_dict["heart_health"] = final_labels

        print("Tri", final_labels["tri_l"].value_counts())
        print("Ldl", final_labels["ldl_l"].value_counts())
        print("Hdl", final_labels["hdl_l"].value_counts())
        print("Chol", final_labels["chol_l"].value_counts())
        print("CReactive", final_labels["creac_l"].value_counts())

    def inflammation(self, data):
        final_labels = data[["eid"]]
        final_labels["cReactive"] = self.cReactive(data)
        ## remove nulls
        final_labels.dropna(subset=["cReactive"], inplace=True)
        final_labels["inflammation_l"] = 1
        final_labels["inflammation_l"] = np.where(final_labels["cReactive"] > 10, 0, final_labels["inflammation_l"])
        final_labels.drop(columns=["cReactive"], inplace= True)

        self.labels_dict["inflammation"] = final_labels

        print("Inflammation", final_labels["inflammation_l"].value_counts())

    def generate_labels(self, data):
        self.vitamind(data)
        self.hba1c(data)
        self.heart_health(data)
        self.thyroid(data)
        self.inflammation(data)
        self.metabolism(data)
        self.food_senstivity(data)







