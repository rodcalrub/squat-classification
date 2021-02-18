
import os, json
import pandas as pd
from hierarchy import DisplayablePath
# from pathlib import Path
from data import *
import numpy as np
from collections import OrderedDict
from insider import parse
#Path raiz de los archivos
path = ''

#iterar sobre las carpetas y almacenar su nombre para que luego se guarde como label
#iterar sobre las subcarpetas ignorando 1115...
#Iterar sobre los archivos json parseandolos almacenando por cada index que contienen la estructura:
#nombreJson, index, list of 3djoints, nombre de la carpeta(label)


#Pasamos las carpetas con los jsos a un JSON final:
#"D:/squat-clasifier/data/bad_toe": [
        # "0918_squat_000037.json",
        # "0918_squat_000038.json",
        # "0918_squat_000039.json",
        # "0918_squat_000040.json",
        # "0918_squat_000041.json", ....

fileList = {}
def importJSONS(path):
    folders = [pos_json for pos_json in os.listdir(path)]
    for folder in folders:
        if(os.path.isdir(str(path+"/"+folder))):
            sub_folders = os.listdir(path+"/"+folder)
            for json in sub_folders:
                fileList[folder]=[json for json in os.listdir(path+"/"+folder+'/'+json) if json.endswith('.json')]           
    return fileList

#path+"/"+
# JSON con paths de jsons: {<path hasta carpeta(sin carpeta x)>:<lista con jsons dentro>}
GlobalPath= 'D:/squat-clasifier/temp'
jsonFolders = importJSONS(GlobalPath)


i = 0
d = {}
for path_jsons in jsonFolders.keys():
    # parse(json.load(js))
    #print(path_jsons)
    for json_inside in jsonFolders[path_jsons]:
        with open(GlobalPath+"/"+path_jsons+'/1115_3djoints_index/'+json_inside,'r') as f:
            #print(jk+'/'+json_inside)
            listaFrames = parse(json.load(f))
            for key in listaFrames.keys():
                d[i] = [json_inside, key, path_jsons]
                d[i] += [i for i in listaFrames[key]]
                print("Insertando fila: "+str(i)+ "MIERDA PUTA")
                i+=1
print('finish')
column_names = ['Name', 'Frame', 'Label']
column_names += range(1,172)
# print(len(column_names))
df1 = pd.DataFrame(data=d.values(), columns=column_names)
df1.to_csv("D:\\squat-clasifier\\csv_export\\csvFinal.csv")