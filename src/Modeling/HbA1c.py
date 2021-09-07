class HbA1c:

    def combine_data_labels(self, train_data, labels, questionnaire_fts, all_cols):

        fts = questionnaire_fts[questionnaire_fts["hba1c_test"] == "Hba1c"]
        codes = list(fts.Feature.values)
        cols = [["eid"]]
        for code in codes:
            select_cols = list(filter(lambda x: x.startswith(str(code)), all_cols))
            cols.append(select_cols)

        final_cols = [item for sublist in cols for item in sublist]
        train_data_a1c = train_data.recent_training_data[final_cols]

        labels_a1c = labels.labels_dict["hbA1c"]

        final_data_a1c = labels_a1c.merge(train_data_a1c, how="left", on="eid")
        final_data_a1c.drop(columns=["eid"],inplace=True)
        print(final_data_a1c.shape)

        #final_data_vitamind.to_csv("Vitamin_D_data.csv",index=False)

        return final_data_a1c