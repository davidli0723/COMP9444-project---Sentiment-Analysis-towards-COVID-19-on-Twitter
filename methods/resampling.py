from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import numpy as np
import os


def over_sampler(df):
    ros = RandomOverSampler()
    train_x, train_y = ros.fit_resample(np.array(df['processed']).reshape(-1, 1), np.array(df['label']).reshape(-1, 1))
    train_os = pd.DataFrame(list(zip([x[0] for x in train_x], train_y)), columns = ['processed', 'label'])
    print(train_os['label'].value_counts())
    print('_'*30)
    return train_os


def under_sampler(df):
    rus = RandomUnderSampler()
    train_x, train_y = rus.fit_resample(np.array(df['processed']).reshape(-1, 1), np.array(df['label']).reshape(-1, 1))
    train_us = pd.DataFrame(list(zip([x[0] for x in train_x], train_y)), columns=['processed', 'label'])
    print(train_us['label'].value_counts())
    print('_'*30)
    return train_us


def summon_samps(source=r'111_newdataset/COVIDSenti-B_cleanest.csv', target_root = 'Data/final_dataset/resampling/'):
    s_name = source.split('/')[-1]
    print('start with', s_name)

    df = pd.read_csv(source)
    print(df['label'].value_counts())
    
    print('_'*30)
    
    a = over_sampler(df)

    name = s_name.replace('.', '_over.')
    print(name)
    target_path = os.path.join(target_root, name)
    a.to_csv(target_path, encoding='utf-8')
    
    b = under_sampler(df)

    name = s_name.replace('.', '_under.')
    print(name)
    target_path = os.path.join(target_root, name)
    b.to_csv(target_path, encoding='utf-8')