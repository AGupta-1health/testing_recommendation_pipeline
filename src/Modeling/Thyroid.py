class Thyroid:

    def combine_data_labels(self, train_data, labels, questionnaire_fts, all_cols):

        fts = questionnaire_fts[questionnaire_fts["thyroid_test"] == "thyroid"]
        codes = list(fts.Feature.values)
        cols = [["eid"]]
        for code in codes:
            select_cols = list(filter(lambda x: x.startswith(str(code)), all_cols))
            cols.append(select_cols)

        final_cols = [item for sublist in cols for item in sublist]
        train_data_thyroid = train_data.recent_training_data[final_cols]

        labels_thyroid = labels.labels_dict["thyroid"]

        final_data_thyroid = labels_thyroid.merge(train_data_thyroid, how="left", on="eid")
        final_data_thyroid.drop(columns=["eid"],inplace=True)
        print(final_data_thyroid.shape)

        return final_data_thyroid