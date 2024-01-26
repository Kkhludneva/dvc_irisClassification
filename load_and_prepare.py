import os
import random
import re
import sys
import xml.etree.ElementTree
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

import yaml

def main():

    params = yaml.safe_load(open("params.yaml"))["load_and_prepare"]

    split = params["split"]
    seed = params["seed"]

    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns = iris.feature_names)

    le = LabelEncoder()
    df['target'] = le.fit_transform(iris.target_names[iris.target])

    train, test = train_test_split(df, test_size=split, random_state=seed)

    output_train = os.path.join("data", "prepared", "train.csv")
    output_test = os.path.join("data", "prepared", "test.csv")

    print(output_train, output_test, sep='\n')

    fd_out_train = open(output_train, "w", encoding="utf-8")
    fd_out_test = open(output_test, "w", encoding="utf-8")

    fd_out_train.write(train.to_csv())
    fd_out_test.write(test.to_csv())

    fd_out_train.close()
    fd_out_test.close()


if __name__ == "__main__":
     main()


#%%
