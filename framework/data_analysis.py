import pandas as pd
file_name = "C:/Users/张伟/Desktop/causal/rankData/QA.csv"
df = pd.read_csv(file_name)
for x in df.columns.values:
     print(x, ": ", df[x].nunique())
# 找到唯一的题目编号
# for x in df['exercise_code'].unique():
#      print(x, ": ", df['exercise_code'].value_counts()[x])
# for x in df['school_id'].unique():
#      print(x, ": ", df['school_id'].value_counts()[x])
for x in df['user_id'].unique():
     print(x, ": ", df['user_id'].value_counts()[x])
user_id = 40735241896591616
data_sh = df.loc[df['user_id'] == user_id]['exercise_code']
print(data_sh)
# school_id = 18217349272506368
# data_sh = df.loc[df['school_id']==school_id]
# for x in data_sh['exercise_code'].unique():
#      print(x, ": ", data_sh['exercise_code'].value_counts()[x])