import pandas as pd
import os
import shutil
        
def read_tsv(path):
    df = pd.read_csv(path, sep="\t")
    df = df.fillna('')
    print(df.keys().values)
    return [df[key].values for key in df.keys()]

def delete_dir(dirpath):
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
