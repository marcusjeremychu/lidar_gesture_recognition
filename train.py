import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from joblib import dump, load
import os

classes = ["body", "thumb index", "curved hand", "straight hand"]#0, 1, 2, 3
log_root_path = "./logs"
master_df = pd.DataFrame()

def process_log(class_id, log_file_name):
    global master_df

    log_file = open(log_file_name)
    data = json.load(log_file)
    count = 0
    for clusters in data:
        for key in clusters.keys():
            cluster = clusters[key]
            cluster["class"] = class_id
            frame = pd.DataFrame([list(cluster.values())], columns=list(cluster.keys())).dropna()
            master_df = master_df.append(frame, ignore_index=True)
            count += 1

    print("Added " + str(count) + " instances of class: " + classes[class_id] + "\n")

def main():
    # build dataframe
    class_index = 0
    for log in os.listdir(log_root_path):
        print(log + " links with " + str(class_index))
        fp = log_root_path + "/" + log
        process_log(class_index, fp)
        class_index += 1
    print(master_df)

    # train random forest classifier
    y = master_df["class"]
    x = master_df.drop(columns="class")

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(x_train, y_train)

    y_pred = random_forest.predict(x_test)
    print(metrics.classification_report(y_test, y_pred))
    dump(random_forest, 'random_forest.joblib')

if __name__ == "__main__":
    main()