# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import re
os.chdir('C:\\Task_Jooble')

df = pd.read_csv("train.tsv", sep="\t")
feature_cnt = df.features[1].count(',') + 1
feature_names = ['feature_' + str(i) for i in range(feature_cnt)]
new_df = df.features.str.split(',', expand=True).astype(int) 
new_df.columns = feature_names
new_df.drop('feature_0', axis=1, inplace=True)

train_mean = [np.mean(new_df[i]) for i in new_df.columns]
train_sd = [np.std(new_df[i]) for i in new_df.columns]
new_df.head()

df_test = pd.read_csv("test.tsv", sep="\t")
feature_names_test = ['feature_2_' + str(i) for i in range(feature_cnt)]
new_df_test = df_test.features.str.split(',', expand=True).astype(int) 
new_df_test.columns = feature_names_test
new_df_test.drop('feature_2_0', axis=1, inplace=True)

df_res = (new_df_test - train_mean)/train_sd
feature_names_res = ['feature_2_stand_' + str(i) for i in range(1,feature_cnt)]
df_res.columns = feature_names_res
idxmax = new_df_test.idxmax(axis = 1).astype(str)

max_feature_2_index = [int(re.findall(r'(\d+)$',i)[0]) for i in idxmax]
df_res['max_feature_2_index'] = max_feature_2_index
new_df_test['max_feature_2_index'] = max_feature_2_index

max_feature_2_abs_mean_diff = \
    [abs(row[row['max_feature_2_index'].astype(int)] \
    - train_mean[row['max_feature_2_index'].astype(int)]) \
    for lab, row in new_df_test.iterrows()]
df_res['max_feature_2_abs_mean_diff'] = max_feature_2_abs_mean_diff
pd.concat([df_test.id_job, df_res], axis=1).to_csv('test_proc.tsv')


