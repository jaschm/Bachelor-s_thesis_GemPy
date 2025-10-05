import csv
import math
import numpy as np

def best_fit_plane(points):
    pts = np.array(points)
    centroid = pts.mean(axis=0)
    A = pts - centroid
    _, _, Vt = np.linalg.svd(A)
    normal = Vt[-1]
    length = np.linalg.norm(normal)
    if length < 1e-12:
        return (None, None), (None, None)
    normal = normal / length
    Nx, Ny, Nz = normal
    return (Nx, Ny, Nz), (centroid[0], centroid[1], centroid[2])

def compute_strike_dip_from_normal(Nx, Ny, Nz):
    horizontal_mag = math.sqrt(Nx*Nx + Ny*Ny)
    if horizontal_mag < 1e-12:
        dip = 90.0
        strike = 0.0
    else:
        dip = math.degrees(math.atan(abs(Nz)/horizontal_mag))
        strike = math.degrees(math.atan2(Nx, Ny))
        if strike < 0:
            strike += 360.0
    return strike, dip

known_orientations = {
    'Sa': (102.4329, 0),
    'Mr': (282.4328, 0),
    'Hk': (282.4328,0),
    'Sr': (282.4328,0),
    'Srmr': (199,0),
    'Si': (175,0),
    'Ka': (175,0),
    'kallio': (175,0),
    'Ki': (175,0),
}

def create_orientation_file(points_by_formation, output_file):
    polarity = 1
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['X','Y','Z','azimuth','dip','polarity','formation'])

        for formation, points in points_by_formation.items():
            if len(points) < 1:
                continue
            pts = np.array(points)
            centroid = pts.mean(axis=0)
            Xc, Yc, Zc = centroid

            if formation in known_orientations:
                strike, dip = known_orientations[formation]
                writer.writerow([Xc, Yc, Zc, strike, dip, polarity, formation])
            else:
                if len(points) < 3:
                    continue
                (Nx, Ny, Nz), (cx, cy, cz) = best_fit_plane(points)
                if Nx is None:
                    continue
                strike, dip = compute_strike_dip_from_normal(Nx, Ny, Nz)
                writer.writerow([cx, cy, cz, strike, dip, polarity, formation])
