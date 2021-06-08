# REFERENCES
# ===================================
# CLUSTERING
# - https://scikit-learn.org/stable/modules/clustering.html#clustering
# - https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html#sklearn.cluster.OPTICS

from math import cos, sin, pi, floor
from rplidar import RPLidar, RPLidarException

from sklearn.cluster import OPTICS
import numpy as np
import matplotlib.pyplot as plt
import pprint
import json
from joblib import dump, load
import pandas as pd

PORT_NAME = '/dev/ttyUSB0'
MAX_RANGE = 700
classes = ["body", "thumb index", "curved hand", "straight hand"] #0, 1, 2, 3
WRITE = False

if WRITE:
    log_file = open("./log.txt","w")

# grabs data from RPLidar A1
def get_data():
    try:
        lidar = RPLidar(PORT_NAME, baudrate=115200)
        for scan in lidar.iter_scans(max_buf_meas=500):
            break
        lidar.stop()
        return scan
    except RPLidarException:
        print("Failed.")
        lidar.reset()
        return []

# rotates an individual point
def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    Negative angle rotates it clockwise.

    source: https://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python
    """
    ox, oy = origin
    px, py = point

    qx = ox + cos(angle) * (px - ox) - sin(angle) * (py - oy)
    qy = oy + sin(angle) * (px - ox) + cos(angle) * (py - oy)
    return qx, qy

# grabs centroid from points
def get_centroid(points):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    return (sum(x) / len(points), sum(y) / len(points))

# apply rotational and translational transforms to cluster of points around centroid to orient around origin/zero degrees
def transform_clusters_to_local(XY, cluster_labels):
    cluster_map = {}
    for point, label in zip(XY, cluster_labels):
        if label == -1:
            continue
        cluster_map.setdefault(label, []).append(point)
    
    # transform to local coordinates, and rotate to 0 degrees
    original_centroid_list = []
    for key in cluster_map.keys():
        points = np.asarray(cluster_map[key])
        centroid = get_centroid(points)
        original_centroid_list.append(centroid)
        centroid_angle = np.arctan(centroid[1] / centroid[0])   
        for i in range(0, len(points)):
            points[i] = points[i] - centroid
            points[i] = rotate((0,0), points[i], -1 * centroid_angle)
     
    return cluster_map, original_centroid_list

# returns a map of cluster IDs to features
def extract_features(cluster_map):
    feature_map = {}
    for cluster_key in cluster_map.keys():
        features = {}
        points = np.asarray(cluster_map[cluster_key])
        centroid = (0, 0) # we've already transformed it so the centroid is centered around origin
        x = [p[0] for p in points]
        y = [p[1] for p in points]

        # Bounding Box Area
        bounding_box_length = max(x) - min(x)
        bounding_box_height = max(y) - min(y)
        bounding_box_area = float(bounding_box_length * bounding_box_height)
        
        # get line between first and last points
        m = (points[-1, 1] - points[0, 1]) / (points[-1, 0] - points[0, 0])
        line = lambda x_line : m * (x_line - points[0,0]) + points[0,1]

        length = 0.0
        consecutive_angle_list = []
        centroid_angle_list = []
        residual_list = []

        # fencepost 
        residual_list.append(float(points[0,1] - line(points[0,0])))
        centroid_angle_list.append(float(np.arctan((points[0,1] - centroid[1]) / (points[0,0] - centroid[0]))))

        # process everything else
        for i in range(0, len(points) - 1):
            p1 = points[i]
            p2 = points[i+1]

            # Real world length (not sure if U shape is accounted for here)
            distance = float(np.linalg.norm(p2 - p1))
            length += distance

            # relative angle between consecutive points 
            theta_between = float(np.arctan((p2[1] - p1[1]) / (p2[0] - p1[0])))
            consecutive_angle_list.append(theta_between)

            # angles between each point relative to the path centroid.
            theta_centroid = float(np.arctan((p2[1] - centroid[1]) / (p2[0] - centroid[0])))
            centroid_angle_list.append(theta_centroid)

            # residual between line (between first and last points) and point
            residual_list.append(float(p2[1] - line(p2[0])))
        
        # build features
        features["area"] = float(bounding_box_area)
        features["length"] = float(length)

        features["residual_mean"] = np.mean(residual_list)
        features["residual_min"] = min(residual_list)
        features["residual_max"] = max(residual_list)
        features["residual_sum"] = sum(residual_list)
        features["residual_std"]= np.std(residual_list)
        features["residual_range"] = features["residual_max"] - features["residual_min"]
        features["residual_rms"] = np.sqrt(np.mean(np.asarray(residual_list) ** 2))

        features["consecutive_angles_mean"] = np.mean(consecutive_angle_list)
        features["consecutive_angles_range"] = max(consecutive_angle_list) - min(consecutive_angle_list)
        features["consecutive_angles_min"] = min(consecutive_angle_list)
        features["consecutive_angles_max"] = max(consecutive_angle_list)
        features["consecutive_angles_sum"] = sum(consecutive_angle_list)
        features["consecutive_angles_std"] = np.std(consecutive_angle_list)

        features["centroid_angles_mean"] = np.mean(centroid_angle_list)
        features["centroid_angles_range"] = max(centroid_angle_list) - min(centroid_angle_list)
        features["centroid_angles_min"] = min(centroid_angle_list)
        features["centroid_angles_max"] = max(centroid_angle_list)
        features["centroid_angles_sum"] = sum(centroid_angle_list)
        features["centroid_angles_std"] = np.std(centroid_angle_list)
        
        feature_map[int(cluster_key)] = features
    return feature_map
    

def main():
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        optics = OPTICS(min_samples=15)
        classifier = load('random_forest.joblib')
        if WRITE: 
            log_file.write("[")
            
        sample_count = 0
        while (True):
            print(sample_count)
            sample_count += 1
            # Grab and Process data
            scan = get_data()
            x = []
            y = []
            for point in scan:
                if point[0]==15 and point[2] < MAX_RANGE:
                    radians = point[1] * pi/180.0
                    x.append(point[2]*np.sin(radians))
                    y.append(point[2]*np.cos(radians))

            # Cluster data
            XY = np.asarray(list(zip(x, y)))
            if (len(XY) < 15):
                print("Too little points")
                continue
            cluster_labels = optics.fit_predict(XY)

            # Process clusters
            cluster_map, original_centroid_list = transform_clusters_to_local(XY, cluster_labels)
            feature_map = extract_features(cluster_map)

            if WRITE:
                json.dump(feature_map, log_file, indent=2)
                log_file.write(",")

            # plot data
            plt.clf()
            for cluster_key in feature_map.keys():
                features = pd.DataFrame([feature_map[cluster_key].values()], columns=list(feature_map[cluster_key].keys())).dropna()
                class_prediction = classifier.predict(features)
                centroid = original_centroid_list[int(cluster_key)]
                plt.text(centroid[0] + 20, centroid[1] + 20, classes[class_prediction[0]])

            plt.plot(0, 0, marker='o', markersize=10, color="red")
            plt.scatter(XY[:, 0], XY[:, 1], c=cluster_labels, s=50, cmap='viridis')
            plt.xlim([-MAX_RANGE, MAX_RANGE])
            plt.ylim([-MAX_RANGE, MAX_RANGE])
            plt.pause(0.01)
        plt.show()
    except KeyboardInterrupt:
        print('Stopping.')
        if WRITE:
            log_file.write("]")
            log_file.close()
    

if __name__ == '__main__':
    main()