class Inflammation:

    def combine_data_labels(self, train_data, labels, questionnaire_fts, all_cols):

        fts = questionnaire_fts[questionnaire_fts["Inflammation_test"] == "Inflammation"]
        codes = list(fts.Feature.values)
        cols = [["eid"]]
        for code in codes:
            select_cols = list(filter(lambda x: x.startswith(str(code)), all_cols))
            cols.append(select_cols)

        final_cols = [item for sublist in cols for item in sublist]
        train_data_inflammation = train_data.recent_training_data[final_cols]

        labels_inflammation = labels.labels_dict["inflammation"]

        final_data_inflammation = labels_inflammation.merge(train_data_inflammation, how="left", on="eid")
        final_data_inflammation.drop(columns=["eid"],inplace=True)
        print(final_data_inflammation.shape)

        return final_data_inflammation