import pandas as pd


class DataRefresh:

    def select_features_from_ukbiobank(self,questionnaire_fts):
        selected_fts = ["eid"]
        for i in range(len(questionnaire_fts)):
            feature = questionnaire_fts.iloc[i,0]
            ar = questionnaire_fts.iloc[i,2]
            if ar == 0:
                ar = ar + 1
            ins = questionnaire_fts.iloc[i,3]
            for j in range(ins):
                for k in range(ar):
                    ft_name = str(feature)+"-"+str(j)+"."+str(k)
                    selected_fts.append(ft_name)
        new_data = pd.read_csv("raw_data/ukb45971.csv",usecols=selected_fts)
        return new_data

