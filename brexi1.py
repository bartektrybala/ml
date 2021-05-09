from sklearn import svm, datasets
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import pandas as pd

filepath = 'brexit.txt'

df = pd.read_csv(filepath, names=['label', 'sentence'], sep='\t')

first_column = df.pop('sentence')
df.insert(0, 'sentence', first_column)

print(df.head())

X = df['sentence']
y = df['label']

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.80, test_size=0.20, random_state=101)

rbf = svm.SVC(kernel='rbf', gamma=0.5, C=0.1).fit(X_train, y_train)
poly = svm.SVC(kernel='poly', degree=3, C=1).fit(X_train, y_train)
