import pandas as pd
from sklearn.model_selection import train_test_split

# make train, validation, test data
# train 50%, valid 20% test 30%

data = pd.read_csv('./data/raw/Train.csv', index_col=0)

train_size = 0.5

X = data.drop(['Reached.on.Time_Y.N'], axis = 1).copy()
y = data['Reached.on.Time_Y.N']

#split train and remain dataset
X_train, X_rem, y_train, y_rem  = train_test_split(X, y, train_size=train_size)

# split valid and test
# test + split 에서 test가 60%가 되어야 전체에서 30%
test_size = 0.6

X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=test_size)

print(X_train.shape), print(y_train.shape)
print(X_valid.shape), print(y_valid.shape)
print(X_test.shape), print(y_test.shape)

X_train.to_csv('./data/processed/X_train.csv')
y_train.to_csv('./data/processed/y_train.csv')
X_valid.to_csv('./data/processed/X_valid.csv')
y_valid.to_csv('./data/processed/y_valid.csv')
X_test.to_csv('./data/processed/X_test.csv')
y_test.to_csv('./data/processed/y_test.csv')

