
import os, json

from hierarchy import DisplayablePath
# from pathlib import Path
from data import *

# from bad_back_round import 
#bad_back_round, bad_back_warp, bad_head, bad_inner.thigh, bad_shallow, bad_toe, good
#nombre de files: 0918_squat_000016.json


#Path raiz de los archivos
path = ''

#iterar sobre las carpetas y almacenar su nombre para que luego se guarde como label
#iterar sobre las subcarpetas ignorando 1115...
#Iterar sobre los archivos json parseandolos almacenando por cada index que contienen la estructura:
#nombreJson, index, list of 3djoints, nombre de la carpeta(label)

# for (root,dirs,files) in walk('./data', topdown=True): 
#         print (root) 
#         print (dirs) 
#         print(files[1])
#         print ('--------------------------------') 


    #print(path.displayable())

import json
from glob import glob

data = []
father = './data' 

#pattern = path.join('/path/to/json/files', '*.json')
for folder in glob(os.path.join(father)):
    print(folder)
    # for file_name in glob(pattern):
    #     with open(file_name) as f:
    #         data.append(json.load(f))


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
                fileList[path+"/"+folder]=[json for json in os.listdir(path+"/"+folder+'/'+json) if json.endswith('.json')]           
    return fileList


jsonF = importJSONS('D:/squat-clasifier/data')
print(json.dumps(jsonF, indent=4))