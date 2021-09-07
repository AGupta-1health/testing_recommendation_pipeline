class Metabolism:

    def combine_data_labels(self, train_data, labels, questionnaire_fts, all_cols):

        fts = questionnaire_fts[questionnaire_fts["metabolism_test"] == "metabolism"]
        codes = list(fts.Feature.values)
        cols = [["eid"]]
        for code in codes:
            select_cols = list(filter(lambda x: x.startswith(str(code)), all_cols))
            cols.append(select_cols)

        final_cols = [item for sublist in cols for item in sublist]
        train_data_metabolism = train_data.recent_training_data[final_cols]

        labels_metabolism = labels.labels_dict["metabolism"]

        final_data_metabolism = labels_metabolism.merge(train_data_metabolism, how="left", on="eid")
        final_data_metabolism.drop(columns=["eid"],inplace=True)
        print(final_data_metabolism.shape)

        #final_data_vitamind.to_csv("Vitamin_D_data.csv",index=False)

        return final_data_metabolism
