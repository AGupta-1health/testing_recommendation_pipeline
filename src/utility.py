from collections import defaultdict

def add_baseline(dict_baseline, sr, name, id, type_of_ft):
    print("baseline_"+name, sr.shape)
    dict_baseline[id] = [sr,name, type_of_ft]


def add_recent(dict_recent, sr, name, id, type_of_ft):
    print("recent_"+name, sr.shape)
    dict_recent[id] = [sr,name, type_of_ft]


def convert_to_baseline_and_recent(recent, df, field_id):
    df[field_id+"_r"] = recent
    baseline = df.iloc[:, 0]
    return df[field_id+"_r"] , baseline


def multiple_instance_one_array(df, number_of_instances, number_of_arrays, field_id):
    get_index_df = (df.iloc[:, 0:]).notnull().astype('int')
    for j in range(1, number_of_instances+1):
        get_index_df.iloc[:, j-1] = get_index_df.iloc[:, j-1] * j
    df["max_idx"] = get_index_df.idxmax(axis=1)
    lst = []
    for k in range(len(df)):
        lst.append(df.loc[k, df.loc[k, "max_idx"]])
    recent, baseline = convert_to_baseline_and_recent(lst, df, field_id)
    return recent, baseline


def columns_with_lowest_val(df, field_id):
    df["min_val"] = df.min(axis=1)
    recent, baseline = convert_to_baseline_and_recent(df["min_val"], df, field_id)
    return recent, baseline


def columns_with_highest_val(df, field_id):
    df["max_val"] = df.max(axis=1)
    recent, baseline = convert_to_baseline_and_recent(df["max_val"], df, field_id)
    return recent, baseline


def category_to_multiple_features(data, coding_df, number_of_arrays, field_id):

    map_cat = {}
    for i in range(len(coding_df)):
        val = coding_df.iloc[i, 0]
        map_cat[val] = [val]

    print(map_cat)
    keys_cat = list(map_cat.keys())
    print(keys_cat)

    new_data_r = data[[str(field_id)+"-0.0"]]
    for i in range(len(keys_cat)):
        new_data_r[field_id+"_"+str(keys_cat[i])+"_r"] = data.iloc[:, 0:len(data.columns)].isin(map_cat[keys_cat[i]]).any(1).astype(int)

    new_data_b = data[[str(field_id)+"-0.0"]]
    for j in range(len(keys_cat)):
        new_data_b[field_id+"_"+str(keys_cat[j])] = data.iloc[:, 0:number_of_arrays+1].isin(map_cat[keys_cat[j]]).any(1).astype(int)
    print("Hello", new_data_b.shape)
    return new_data_r.iloc[:,1:], new_data_b.iloc[:,1:]


def category_to_multiple_features_parent(data, coding_df, number_of_arrays, field_id):
    map_cat = defaultdict(list)
    for i in range(len(coding_df)):
        parent = coding_df.iloc[i, 3]
        if parent in map_cat:
            map_cat[parent].append(coding_df.iloc[i, 0])
        else:
            map_cat[parent] = [coding_df.iloc[i, 0]]

    keys_cat = list(map_cat.keys())

    new_data_r = data[[str(field_id) + "-0.0"]]
    for i in range(len(keys_cat)):
        new_data_r[field_id + "_" + str(keys_cat[i]) + "_r"] = data.iloc[:, 0:len(data.columns)].isin(
            map_cat[keys_cat[i]]).any(1).astype(int)

    new_data_b = data[[str(field_id) + "-0.0"]]
    for j in range(len(keys_cat)):
        new_data_b[field_id + "_" + str(keys_cat[j])] = data.iloc[:, 0:number_of_arrays + 1].isin(
            map_cat[keys_cat[j]]).any(1).astype(int)

    return new_data_r.iloc[:, 1:], new_data_b.iloc[:, 1:]


def call_all_public_function_within_obj(x,data):
    """
    this function will call all the public functions and the function does not start with 'update' for instance x.
    :param x: object, instance
    :param data: DataTrainClass to be modified.
    :return:
    """
    public_method_names = [method for method in dir(x) if callable(getattr(x, method)) if
                           not method.startswith('_')]  # 'private' methods start from _
    for method in public_method_names:
        kwargs = {'data': data}
        getattr(x, method)(**kwargs)