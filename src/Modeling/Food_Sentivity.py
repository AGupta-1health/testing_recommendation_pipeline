class Food_Senstivity:

    def combine_data_labels(self, train_data, labels, questionnaire_fts, all_cols):

        fts = questionnaire_fts[questionnaire_fts["food_test"] == "food_senstivity"]
        codes = list(fts.Feature.values)
        cols = [["eid"]]
        for code in codes:
            select_cols = list(filter(lambda x: x.startswith(str(code)), all_cols))
            cols.append(select_cols)

        final_cols = [item for sublist in cols for item in sublist]
        train_data_food = train_data.recent_training_data[final_cols]

        labels_food = labels.labels_dict["food_senstivity"]

        final_data_food = labels_food.merge(train_data_food, how="left", on="eid")
        final_data_food.drop(columns=["eid"],inplace=True)
        print(final_data_food.shape)

        return final_data_food