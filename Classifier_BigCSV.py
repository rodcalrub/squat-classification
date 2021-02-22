import random


#Lectura
import pandas as pd
pd.set_option('display.max_columns', None)

#Preprocesamiento
from sklearn.utils import shuffle
import numpy as np
from sklearn.preprocessing import StandardScaler

#Visualizacion
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
# warnings.simplefilter(action='ignore', category=UserWarning)

#Regresion
from time import time
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, validation_curve, learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor

#Clasificacion
from sklearn import tree
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

#Seleccion y extraccion de atributos
from sklearn.feature_selection import RFE
from sklearn.feature_selection import f_classif, mutual_info_classif, chi2
from sklearn.feature_selection import SelectPercentile, SelectKBest, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

#Clustering
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from scipy.cluster import hierarchy
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, calinski_harabasz_score, silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, DBSCAN, MeanShift, AffinityPropagation

#Ajuste de hiperparametros

from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso, Ridge
# from tpot import TPOTClassifier,TPOTRegressor
random.seed(12345)

import joblib

DATA = pd.read_csv('./csvFinal2.csv')
print('This dataset is size : ', DATA.shape)

#Quitar algun ejemplo para probar luego
examples = pd.DataFrame()
#           good                        good                    bad toe                     bad toe                 bad shallow             bshallow                 b inner b                  b inner                     b head b                bhead                   b back w               b back w                     b back round w b        b back round
samples = ["0918_squat_000001.json", "0918_squat_000004.json", "0918_squat_000040.json","0918_squat_000076.json","0922_squat_000053.json","0922_squat_000197.json", "0922_squat_000180.json","1015_squat_000252.json", "0922_squat_000033.json","1003_squat_000137.json", "0922_squat_000107.json","0918_squat_000052.json",  "0922_squat_000175.json","1015_squat_000233.json" ]
for name in samples:
    examples += DATA[DATA['Name']== name]
DATA = DATA.drop(examples)
#Barajar antes de cortar
DATA = shuffle(DATA)

#separar labels y train test
y = DATA['Label']
X = DATA.drop(['Label', 'Frame' ,'Name'], axis=1)

print(y.value_counts())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)



RESULTADOS_CLAS = pd.DataFrame(columns= ['Accuracy', "Tiempo"])
#multi label clasify
def experimento_clas(clasificador, X_train, y_train, X_test, y_test):
    print('Training... :', clasificador)
    inicio = time()
    print('Fitting')
    clasificador.fit(X_train, y_train)
    print("Predicting")
    y_pred = clasificador.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    joblib.dump(clasificador, './trainedModels/'+str(clasificador)+'.joblib') 
    fin = time()
    print(clasificador, " - acuracy: ", acc, ", tiempo : ", fin-inicio)
    return acc, fin-inicio

clasificadores = [
    GaussianNB(),
    KNeighborsClassifier(n_neighbors=1, n_jobs =-1),
    KNeighborsClassifier(n_neighbors=3, n_jobs =-1),
    KNeighborsClassifier(n_neighbors=5, n_jobs =-1),
    # SVC(kernel="linear",probability=True),
    # SVC(kernel="rbf",probability=True),
    # SVC(kernel="sigmoid",probability=True),
    # SVC(kernel="poly",probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=10, n_jobs =-1),
    RandomForestClassifier(n_estimators=100, n_jobs =-1),
    RandomForestClassifier(n_estimators=1000, n_jobs =-1),
    ExtraTreesClassifier(n_estimators=10, n_jobs =-1),
    ExtraTreesClassifier(n_estimators=100, n_jobs =-1),
    ExtraTreesClassifier(n_estimators=1000, n_jobs =-1)
    ]

#for clas in clasificadores:
#    RESULTADOS_CLAS.loc[str(clas)] = experimento_clas(clas, X_train, y_train, X_test, y_test)

print(RESULTADOS_CLAS)

#CLUSTERING

def muestra_agrupacion(puntos, clusters, name):
    puntos_2D = pd.DataFrame(PCA(n_components=2).fit_transform(puntos), columns=['x', 'y'])
    puntos_2D['grupo'] = clusters
    numero_grupos = len(pd.Series(clusters).unique())
    
    plt.figure(figsize=(5,5))
    paleta = sns.color_palette("bright", numero_grupos)
    sns.scatterplot(x='x', y='y', hue='grupo', data=puntos_2D, palette=paleta)
    plt.xticks([], [])
    plt.xlabel('')
    plt.yticks([], [])
    plt.ylabel('')
    #plt.show()
    plt.savefig('./clustering/'+name+'.png')

muestra_agrupacion(X, y, "No Clustering")

METRICAS = pd.DataFrame(columns=['Resultado'])
for i in range(2,8):
    kmeans=KMeans(n_clusters=i,random_state=0).fit(X)
    #muestra_agrupacion(X, kmeans.labels_)
    print("kmean with k="+str(i))
    METRICAS.loc['K='+str(i)] = metrics.adjusted_rand_score(y,kmeans.labels_, "K"+str(i))
                                  
print(METRICAS)                                


#seleccion 
RESULTADOS_SELEC = pd.DataFrame(columns= ['Accuracy', "Tiempo"])
clasif = KNeighborsClassifier(n_neighbors=1, n_jobs =-1)
selector = SelectKBest(chi2, k=20)

RESULTADOS_SELEC["No selector"] = experimento_clas(clasif, X_train, y_train, X_test, y_test)
selector  = SelectKBest(chi2, k=20)
selector.fit_transform(X, y)
X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(X, y, test_size=0.8)
RESULTADOS_SELEC["chi2"]=experimento_clas(clasif, X_train_sel, y_train_sel, X_test_sel, y_test_sel)
print('Diference'+str(RESULTADOS_SELEC["chi2"]["Accuracy"]-RESULTADOS_SELEC["No selector"]["Accuracy"]))
