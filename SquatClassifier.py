import joblib
import pandas as pd
import sys
from sklearn import metrics

pd.set_option('display.max_columns', None)

name = sys.argv[1]
DATA = pd.read_csv('./csvFinal2.csv')
sample = DATA[DATA['Name']== name]

sample_X = sample.drop(['Label', 'Frame' ,'Name'], axis=1)

sample_y = sample['Label']

classifier = joblib.load('./trainedModels/KNeighborsClassifier(n_jobs=-1, n_neighbors=1).joblib')
preds=classifier.predict(sample_X)


df = pd.DataFrame(preds, columns=["Frame predicition"])
print(df.value_counts())
print(df)

acc = metrics.accuracy_score(sample_y, preds)
print("Accuracy: ", acc)
print("real", sample_y.iloc[0])