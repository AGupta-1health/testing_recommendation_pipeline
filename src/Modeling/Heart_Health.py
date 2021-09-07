class Heart_Health:

    def combine_data_labels(self, train_data, labels, questionnaire_fts, all_cols):

        fts = questionnaire_fts[questionnaire_fts["heart_test"] == "Heart_health"]
        codes = list(fts.Feature.values)
        cols = [["eid"]]
        for code in codes:
            select_cols = list(filter(lambda x: x.startswith(str(code)), all_cols))
            cols.append(select_cols)

        final_cols = [item for sublist in cols for item in sublist]
        train_data_vitamind = train_data.recent_training_data[final_cols]

        labels_tri = labels.labels_dict["heart_health"][["eid","tri_l"]]
        labels_hdl = labels.labels_dict["heart_health"][["eid","hdl_l"]]
        labels_ldl = labels.labels_dict["heart_health"][["eid","ldl_l"]]
        labels_chol = labels.labels_dict["heart_health"][["eid","chol_l"]]
        labels_crp = labels.labels_dict["heart_health"][["eid","creac_l"]]

        final_data_tri = labels_tri.merge(train_data_vitamind, how="left", on="eid")
        final_data_hdl = labels_hdl.merge(train_data_vitamind, how="left", on="eid")
        final_data_ldl = labels_ldl.merge(train_data_vitamind, how="left", on="eid")
        final_data_chol = labels_chol.merge(train_data_vitamind, how="left", on="eid")
        final_data_crp = labels_crp.merge(train_data_vitamind, how="left", on="eid")

        final_data_tri.drop(columns=["eid"],inplace=True)
        final_data_hdl.drop(columns=["eid"], inplace=True)
        final_data_ldl.drop(columns=["eid"], inplace=True)
        final_data_chol.drop(columns=["eid"], inplace=True)
        final_data_crp.drop(columns=["eid"], inplace=True)


        #final_data_vitamind.to_csv("Vitamin_D_data.csv",index=False)

        return final_data_tri,final_data_hdl,final_data_ldl,final_data_chol,final_data_crp