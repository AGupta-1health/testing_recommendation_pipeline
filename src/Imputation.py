from statistics import mean

class ImputeMissingValues:

    def fill_na_with_number(self, recent, baseline, value):
        recent = recent.fillna(value)
        baseline = baseline.fillna(value)
        return recent, baseline

    def fill_na_with_average(self, recent, baseline):
        value_b = baseline.mean()
        value_r = recent.mean()
        recent = recent.fillna(value_r)
        baseline = baseline.fillna(value_b)
        return recent, baseline

    def fill_df(self, df, value):
        df.fillna(value, inplace=True)
        print("Columns", list(df.columns))
        return df