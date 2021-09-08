
#%%
from src.ModelUpdater import modelUpdater
from src.DataRefresh import DataRefresh
import pandas as pd
from datetime import datetime
#%%
questionnaire_fts = pd.read_excel("raw_data/feature_list.xlsx")
##%

#%%

## Data refresh from the main data source -- UKbiobank -- in future from a database
## Select features from ukbiobank data - selected features are for all the tests
## this piece of code is not required to run everytime
## Save the copy of the data with a version in raw data folder
filtered_data = DataRefresh()
latest_data_copy = filtered_data.select_features_from_ukbiobank(questionnaire_fts)
today_date = datetime.today().strftime('%Y-%m-%d')
latest_data_copy.to_csv("raw_data/data_"+today_date+".csv",index=False)
############
#%%
data = pd.read_csv("raw_data/data_2021-08-23.csv")
updater = modelUpdater(data, questionnaire_fts)

## modeling



