import pandas as pd
import sys
import os
from twembeddings import load_dataset

"""
with command python split_by_days.py --file data/event2012.tsv
split the datasets by day in a new folder named dailytweets_event2000
daily files will be tw_{datetime}.tsv
"""



if ("--file" in sys.argv) :
    file_dataset = sys.argv[-1] # index [1] parce que argv[0] == --file
    print("file_dataset",file_dataset)
    name_daily = file_dataset.split(".")[0].split("/")[1]
    print("name daily", name_daily)

else :
    print("need --file argument, end of script")
    quit()

# df = pd.read_csv(file_dataset, sep='\t')
df = load_dataset(file_dataset,"annotated")
def split_by_date(df):
    print("new")
    # dfday = df["created_at"].apply(lambda x: x.split()[2])
    # dfmonth = df["created_at"].apply(lambda x: x.split()[1])
    # dfyear = df["created_at"].apply(lambda x: x.split()[-1])
    # df["datetime"] = dfday + "_" + dfmonth + "_" + dfyear
    dict_dfs = {date: data for date, data in df.groupby(df["date"], dropna = False)}

    return dict_dfs

dict_df_date = split_by_date(df)

if not os.path.exists(f"data/dailytweets_{name_daily}"):
    os.makedirs(f"data/dailytweets_{name_daily}")
for datetime, data in dict_df_date.items() :
    print(datetime)
    data.to_csv(f"data/dailytweets_{name_daily}/tw_{datetime}.tsv", sep='\t')
