import os
import random
import re
import sys
import xml.etree.ElementTree
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

import yaml

def main():

    params = yaml.safe_load(open("../params.yaml"))["load_and_prepare"]

    split = params["split"]
    seed = params["seed"]

    iris = datasets.load_iris()
    df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                      columns= iris['feature_names'] + ['target'])
    print(df.shape)
    le = LabelEncoder()
    df['target'] = le.fit_transform(df['target'])

    train, test = train_test_split(df, test_size=split, random_state=seed)
    print(train.shape, test.shape)

    train.to_csv("data/prepared/train.csv")
    test.to_csv("data/prepared/test.csv")


if __name__ == "__main__":
     main()

