import csv
import math
import numpy as np

def read_filtered_data(filename):
    points_by_formation = {}
    with open(filename, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            formation = row['formation']
            try:
                X = float(row['X'])
                Y = float(row['Y'])
                Z = float(row['Z'])
            except ValueError:
                continue
            if any(math.isnan(v) for v in [X, Y, Z]):
                continue
            if formation not in points_by_formation:
                points_by_formation[formation] = []
            points_by_formation[formation].append((X, Y, Z))
    return points_by_formation
