import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV

RAW_DATA = 'data/interim/draft1_data'
X_train = pd.read_csv(RAW_DATA + '/X_train.csv', index_col=0)
X_test = pd.read_csv(RAW_DATA + '/X_test.csv', index_col=0)
y_train = pd.read_csv(RAW_DATA + '/y_train.csv', index_col=0)
y_test = pd.read_csv(RAW_DATA + '/y_test.csv', index_col=0)

"""
ID: ID Number of Customers.
Warehouse block: The Company have big Warehouse which is divided in to block such as A,B,C,D,E.
Mode of shipment:The Company Ships the products in multiple way such as Ship, Flight and Road.
Customer care calls: The number of calls made from enquiry for enquiry of the shipment. 문의 수인듯
Customer rating: The company has rated from every customer. 1 is the lowest (Worst), 5 is the highest (Best).
Cost of the product: Cost of the Product in US Dollars.
Prior purchases: The Number of Prior Purchase.
Product importance: The company has categorized the product in the various parameter such as low, medium, high.
Gender: Male and Female.
Discount offered: Discount offered on that specific product.
Weight in gms: It is the weight in grams.
Reached on time: It is the target variable, where 1 Indicates that the product has NOT reached on time and 0 indicates it has reached on time.
"""

# EDA
# ID는 제거하자
# prior purchases랑 customer rating.
# 실수형 중에서는, discount offered, weight in gms가 종속변수와 corr이 좀 있어보임
# 명목형은 gender, product importnace, mode of shipment.

# weight in gms + Discount_offered + Customer_care_calls 합체
# prior purchases, customer rating 더해서 하나의  dummy 변수로 만들자
# 명목형들 one-hot encoding 

# prior purchases discount_offered 가 이상치가 많음. 그거 제거하는 코드도 나중에 추가할 것 
def preprocessing(dataframe):

    # create dummy variables
    dataframe['dummy_1'] = dataframe['Weight_in_gms'] + dataframe['Discount_offered'] + dataframe['Customer_care_calls']
    dataframe['dummy_2'] = dataframe['Prior_purchases'] + dataframe['Customer_rating']

    # drop columns
    dataframe.drop(['ID', 'Weight_in_gms','Discount_offered', 'Customer_care_calls','Prior_purchases', 'Customer_rating'], axis = 1, inplace = True)


    # one-hot encoding
    dataframe = pd.get_dummies(dataframe)

    return dataframe

def scaler(train, test):
    #MinMaxScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train)

    X_test = scaler.transform(test)

    return X_train, X_test






def merge_dataframe(df1, df2):
    merged_dataframe = pd.concat([df1, df2], axis = 1)

    return merged_dataframe

def check_ID(dataframe):
    length_id = len(dataframe['ID'].tolist())
    length_set = len(set(dataframe['ID'].tolist()))

    return length_id, length_set

def check_high_correlation(dataframe):
    corr_data = dataframe.corr()
    result = dict()
    try:
        for cols in dataframe.columns:
            high_corr = corr_data[abs(corr_data[cols]) > 0.5]
            result[cols] = high_corr
    except:
        pass
    
    return result
    


# -> 이상치를 제거하고 build한 모델과, 이상치를 제거하지 않고 build한 모델의 (test data 성능 차이를 검증)



if __name__ == '__main__':
    # df = remove_cols(raw_data)
    # print(check_ID(X_train))
    # print(check_ID(X_test))

    # corr_df = check_correlation(merge_dataframe(X_train, y_train))

    # high_corr = corr_df[abs(corr_df['Reached.on.Time_Y.N']) > 0.6]
    # # print(high_corr)
    # print(check_high_correlation(merge_dataframe(X_train, y_train)))

    y_train = y_train['Reached.on.Time_Y.N']
    y_test = y_test['Reached.on.Time_Y.N']

    X_train= preprocessing(X_train)

    X_test = preprocessing(X_test)
    X_train_scaled, X_test_scaled = scaler(X_train, X_test)


    base_model = LogisticRegression(random_state=42)

    penalty = ['none', 'l1', 'l2']
    c = [10, 1.0, 0.1, 0.01]
    random_grid = {
                    'penalty' : penalty,
                    'C' : c
    }
    logistic_random = RandomizedSearchCV(estimator=base_model, param_distributions=random_grid,
    n_iter = 100, cv = 3, verbose = 2, random_state = 42, n_jobs = -1)


    base_model.fit(X_train, y_train)

    logistic_random.fit(X_train, y_train)
    best_random = logistic_random.best_estimator_

    y_pred = base_model.predict(X_test)


    y_pred_2 = best_random.predict(X_test)
    classification = classification_report(y_test, y_pred)

    classification2 = classification_report(y_test, y_pred_2)

    print(classification)

    print(classification2)

