import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#data import
X_train = pd.read_csv('./data/processed/X_train.csv')
y_train = pd.read_csv('./data/processed/y_train.csv')
X_valid = pd.read_csv('./data/processed/X_valid.csv')
y_valid = pd.read_csv('./data/processed/y_valid.csv')
X_test = pd.read_csv('./data/processed/X_test.csv')
y_test = pd.read_csv('./data/processed/y_test.csv')

raw_data = pd.read_csv('./data/processed/Train.csv')

def describe_data(df):
    """[summary]

    Args:
        df ([dataframe, series]): data want to analysis

    Returns:
        [tuple]: shape, statistics, data types of data
    """
    shape = df.shape
    described = df.describe()
    data_types = df.dtypes
    return shape, described, data_types


def NA_count_ratio(df):
    """[summary]

    Args:
        df ([dataframe]): data want to analysis about na

    Returns:
        [dataframe]: dataframe of na count and its ratio
    """
    df_na_count = df.isnull().sum().to_frame('nan_count')
    df_ratio = pd.DataFrame(data = df.isnull().sum() / len(df),
        columns=['nan_ratio'])
    na_describe = pd.concat([df_na_count, df_ratio], axis = 1)
    return na_describe


def skew_kurtosis(test_series):
    """[summary]

    Args:
        test_series ([series]): series want to calculate skewnewss, kurtosis

    Returns:
        [int]: value of skewness, kurtosis
        <skewness>
        if negeative : values located at right
        elif positive : values located at left
        <kurtosis>
        if higher than 3 : have many outliers
        else : less extreme values
    """
    skewness = test_series.skew()
    kurtosis = test_series.kurt()

    return skewness, kurtosis


# 범주형 변수일 때 사용
def plot_values(test_series):
    """[summary]

    Args:
        test_series ([type]): [description]
    """
    value_counts = test_series.value_counts().sort_index(ascending=True)
    plt.figure(figsize=(10,5))
    value_counts.plot(kind='bar')
    plt.show()


def correlation(df):
    return df.corr()




print(NA_count_ratio(X_train))
print(correlation(X_train))
# print(plot_values(y_train))

shape, described, data_types = describe_data(raw_data)

print(shape, described, data_types)


skewness, kurtosis = skew_kurtosis(y_train)


print('----------')
print(skewness)
print('-----------')
print(kurtosis)
# print('skewness : %f, kurtosis : %f' % (skewness, kurtosis))








