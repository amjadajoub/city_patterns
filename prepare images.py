from helper_funcs import database_connection

conn, cur = database_connection.get_connection()
import os
print(os.getcwd())
import matplotlib.pyplot as plt

from shapely import wkt
import cv2


def project_points_n_diminsion(points, transformation_matrix):
    """Project 3D points by a 3D transformation matrix using homogeneous coordinates"""
    points_homo = np.ones((points.shape[0], points.shape[1] + 1))
    points_homo[:, :-1] = points
    points_homo = np.matmul(transformation_matrix, np.transpose(points_homo))
    projected_points = np.transpose(points_homo)[:, :-1]
    return projected_points


def save_image(name, geo_df, bbox_df, img_size):
    min_x = float(bbox_df.bounds.minx)
    min_y = float(bbox_df.bounds.miny)
    max_x = float(bbox_df.bounds.maxx)
    max_y = float(bbox_df.bounds.maxy)

    s_x = img_size[1] / (max_x - min_x)
    s_y = img_size[0] / (max_y - min_y)
    dx = - min_x * s_x
    dy = - min_y * s_y
    transformation_matrix = np.array([[s_x, 0, dx], [0, s_y, dy], [0, 0, 1]])

    geometry = [geom for geom in geo_df.geom]
    coords = []
    for geom in geometry:
        c = geom.exterior.coords.xy if not geom.type == "MultiPolygon" else geom[0].exterior.coords.xy
        coords.append(c)
    temp_image = np.zeros(img_size)
    for coord in coords:
        points_array = np.array([p for p in zip(list(coord[0]), list(coord[1]))])
        local_points = project_points_n_diminsion(points_array, transformation_matrix)
        building_footprint_polygon = [[list(point) for point in local_points]]
        cv2.fillPoly(temp_image, np.array(building_footprint_polygon, dtype=np.int32), 1)
    np.save(name, temp_image)


def get_max_bound(bbox, centroid, threshold = 0):
    y_d = bbox[3] - bbox[1]
    x_d = bbox[2] - bbox[0]
    max = y_d if y_d > x_d else x_d
    max = max + threshold
    bbox = (centroid.x - max / 2, centroid.y - max / 2, centroid.x + max / 2, centroid.y + max / 2)
    return bbox

import geopandas as gpd
def process_cluster(data, cluster, schema, image_size):
    cluster_geom = data.loc[cluster == data['cluster_n']]
    #merged = cluster_geom.unary_union
    bbox = cluster_geom.bounds.values[0]
    centroid = cluster_geom.centroid.values[0]
    max_bound = get_max_bound(bbox, centroid)
    sql = f"""select geom from {schema}.buildings
            where ST_intersects(geom, ST_MakeEnvelope({max_bound[0]}, {max_bound[1]},
            {max_bound[2]}, {max_bound[3]}, 4326))"""
    clipped_geom = gpd.read_postgis(sql, conn, geom_col='geom')
    save_image(f"./images/{schema}/{cluster}.npy", clipped_geom, cluster_geom, image_size)
    print(f"./images/{schema}/{cluster}.npy")




import numpy as np
image_size = (512, 512)
for schema in ["germany", "france","spain","us"]:
    if schema == "germany":
        sql = f"""select * from {schema}.buildings_clusters_001_50_envelope
            where cluster_n is not null
            and st_area(st_envelope) < 1.58947346799971e-05"""
    else:
        sql = f"""select * from {schema}.buildings_clusters_001_50_envelope
            where st_intersects(st_envelope, (select geom from {schema}."AOI"))
          and cluster_n is not null
        and st_area(st_envelope) < 1.58947346799971e-05"""

    data = gpd.read_postgis(sql, conn, geom_col='st_envelope')
    data['cluster_n'][np.isnan(data['cluster_n'])] = 0
    data['cluster_n'] = data['cluster_n'].astype(dtype=np.int)
    for cluster in data['cluster_n'].unique():
        os.makedirs(f"./images/{schema}", exist_ok=True)
        process_cluster(data, cluster, schema, image_size)


