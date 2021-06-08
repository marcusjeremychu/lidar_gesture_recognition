import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from joblib import dump, load

classes = ["body", "straight hand", "curved hand"] #0, 1, 2
master_df = pd.DataFrame()

def process_log(class_id, log_file_name):
    global master_df

    log_file = open(log_file_name)
    data = json.load(log_file)
    count = 0
    for clusters in data:
        for key in clusters.keys():
            cluster = clusters[key]
            flattened_map = {}

            flattened_map["area"] = cluster["area"]
            flattened_map["length"] = cluster["length"]
            flattened_map["class"] = class_id
            for stat in cluster["residual_stats"].keys():
                flattened_map["residual_" + str(stat)] = cluster["residual_stats"][stat]

            consecutive_angles = cluster["consecutive_angles"]
            flattened_map["consecutive_angles_mean"] = np.mean(consecutive_angles)
            flattened_map["consecutive_angles_range"] = max(consecutive_angles) - min(consecutive_angles)
            flattened_map["consecutive_angles_min"] = min(consecutive_angles)
            flattened_map["consecutive_angles_max"] = max(consecutive_angles)
            flattened_map["consecutive_angles_sum"] = sum(consecutive_angles)
            flattened_map["consecutive_angles_std"] = np.std(consecutive_angles)

            centroid_angles = cluster["centroid_angles"]
            flattened_map["centroid_angles_mean"] = np.mean(centroid_angles)
            flattened_map["centroid_angles_range"] = max(centroid_angles) - min(centroid_angles)
            flattened_map["centroid_angles_min"] = min(centroid_angles)
            flattened_map["centroid_angles_max"] = max(centroid_angles)
            flattened_map["centroid_angles_sum"] = sum(centroid_angles)
            flattened_map["centroid_angles_std"] = np.std(centroid_angles)
            count += 1

            frame = pd.DataFrame([list(flattened_map.values())], columns=list(flattened_map.keys())).dropna()
            master_df = master_df.append(frame, ignore_index=True)
    print("Added " + str(count) + " instances of class: " + classes[class_id])

def main():
    # build dataframe
    process_log(0, "./log_body.txt")
    process_log(1, "./log_straighthand.txt")
    process_log(2, "./log_curvedhand.txt")

    # train random forest classifier
    y = master_df["class"]
    x = master_df.drop(columns="class")
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(x_train, y_train)

    print(x_test)
    y_pred = random_forest.predict(x_test)
    print("Accuracy:", metrics.classification_report(y_test, y_pred))
    dump(random_forest, 'random_forest.joblib')

if __name__ == "__main__":
    main()