
import os, json
import pandas as pd
from hierarchy import DisplayablePath
# from pathlib import Path
from data import *
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
                fileList[path+"/"+folder+'/'+json]=[json for json in os.listdir(path+"/"+folder+'/'+json) if json.endswith('.json')]           
    return fileList


jsonF = importJSONS('D:/squat-clasifier/data')
#print(json.dumps(jsonF, indent=4))


for jk in jsonF.keys():
    # parse(json.load(js))
    for json_inside in jsonF[jk]:
         with open(jk+'/'+json_inside,'r') as f:
            print()
            listaFinal = parse(json.load(f))

print(listaFinal)
# df1 = pd.DataFrame.from_dict(data=OrderedDict(d.items()),  orient='index',columns=[])
# df1.to_csv(path+"\\csv_export\\csvFinal.csv")