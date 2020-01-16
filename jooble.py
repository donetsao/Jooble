# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import re
import tempfile
from datetime import datetime
os.chdir('C:\\Task_Jooble')

# Data import
t = datetime.now()
df_train = pd.read_csv("train.tsv", sep="\t")
df_test = pd.read_csv("test.tsv", sep="\t")

# performance test > 2M
#for i in range(12):
#    df_test = df_test.append(df_test, ignore_index=True)
#len(df_test.index)

# Train Data preparation
feature_cnt = df_train.features[1].count(',') + 1 # feature columns count 
df_train = df_train.features.str.split(',', expand=True).astype(int) # split column
df_train.columns = ['feature_' + str(i) for i in range(feature_cnt)] # feature_names

# Test Data preparation
temp_file =  tempfile.NamedTemporaryFile(delete=False) 
df_test.features.to_csv(temp_file.name, sep="\t", index = False, header = False)
new_df_test = pd.read_csv(temp_file.name, sep=",", header = None) 
temp_file.close()
new_df_test.columns = ['feature_2_' + str(i) for i in range(feature_cnt)] 
df_test = pd.concat([df_test.id_job, new_df_test], axis=1)

# filter feature type = 2 
df_train2 = df_train[df_train['feature_0'] == 2]
df_train2.drop('feature_0', axis=1, inplace=True)
df_test2 = df_test[df_test['feature_2_0'] == 2]
df_test2_id_job = df_test2.id_job
df_test2.drop('feature_2_0', axis=1, inplace=True)
df_test2.drop('id_job', axis=1, inplace=True)

# Z-Score Normalization
train_mean2 = np.array(df_train2.mean())
train_sd2 = np.array(df_train2.std())
df_test2_stand = pd.DataFrame((np.array(df_test2) - train_mean2)/train_sd2)
df_test2_stand.columns = ['feature_2_stand_' + str(i) for i in range(1,feature_cnt)]

# max_feature index and diff
idxmax = df_test2.idxmax(axis = 1).astype(str)
max_feature_2_index = [int(re.findall(r'(\d+)$',i)[0]) for i in idxmax]
df_test2_stand['max_feature_2_index'] = max_feature_2_index
df_test2['max_feature_2_index'] = max_feature_2_index
max_feature_2_abs_mean_diff = \
    [abs(row[row['max_feature_2_index'].astype(int)] \
    - train_mean2[row['max_feature_2_index'].astype(int)]) \
    for lab, row in df_test2.iterrows()]
df_test2_stand['max_feature_2_abs_mean_diff'] = max_feature_2_abs_mean_diff

# write result
pd.concat([df_test2_id_job, df_test2_stand], axis=1).to_csv('test_proc.tsv', sep="\t", index = False)
print(datetime.now() - t)


