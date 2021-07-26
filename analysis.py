from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

img_shape = (64, 64)
import cv2, os
import skimage


def convert_to_dft(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    img = 20 * np.log(np.abs(fshift))
    img[np.where(np.isnan(img))] = 0
    img[np.where(np.isnan(img))] = 0
    img[np.where(np.isneginf(img))] = 0
    return img


import csv


def write_csv(file_name, l):
    with open(file_name, 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        for x in l:
            wr.writerow([x])


from random import sample
from sklearn.preprocessing import StandardScaler

def read_data(dir, sample_n, frequency_domin=False, multiplyer= 1, image_size=(64,64), normlized = False):
    files = os.listdir(os.path.join(dir, "us"))
    paths = [os.path.join(dir, "us", file) for file in files if file[-4:] == '.npy']
    paths_us = sample(paths, sample_n)

    files = os.listdir(os.path.join(dir, "france"))
    paths = [os.path.join(dir, "france", file) for file in files if file[-4:] == '.npy']
    paths_france = sample(paths, sample_n)

    files = os.listdir(os.path.join(dir, "spain"))
    paths = [os.path.join(dir, "spain", file) for file in files if file[-4:] == '.npy']
    paths_spain = sample(paths, sample_n)

    files = os.listdir(os.path.join(dir, "germany"))
    paths_germany = [os.path.join(dir, "germany", file) for file in files if file[-4:] == '.npy']
    paths_germany = sample(paths_germany, multiplyer * sample_n)

    paths = paths_us + paths_france + paths_spain + paths_germany
    imgs = []
    for p in paths:
        img = np.load(p)
        if img.shape != image_size:
            img = cv2.resize(img, image_size, cv2.INTER_AREA)
        if frequency_domin:
            img = convert_to_dft(img)
        imgs.append(img)
    data = [img.flatten() for img in imgs]
    if normlized:
        scaler = StandardScaler()
        # Fit on training set only.
        scaler.fit(data)
        # Apply transform to both the training set and the test set.
        data = scaler.transform(data)
    return np.array(data), paths


from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


def view_projected_data(projected_data, n_components, labels):
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    if n_components > 2:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = plt.subplot(1, 1, 1)
    for i, txt in enumerate(labels):
        if n_components == 2:
            ax.annotate(txt, (projected_data[i, 0], projected_data[i, 1]))
            ax.plot(projected_data[i, 0], projected_data[i, 1], 'o', color=tuple(colors[int(txt)]))
        elif n_components > 2:
            ax.text(projected_data[i, 0], projected_data[i, 1], projected_data[i, 2], txt)
            ax.scatter(projected_data[i, 0], projected_data[i, 1], projected_data[i, 2], 'o',
                       color=tuple(colors[int(txt)]))


from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix


def build_graph(X):
    num_neighbours = 2
    graph = kneighbors_graph(X, num_neighbours, mode='connectivity', include_self=False)
    return graph


def shortest_path(graph, start_point):
    graph = csr_matrix(graph)
    distances, predecessors = dijkstra(csgraph=graph,
                                       directed=False,
                                       indices=start_point,
                                       return_predecessors=True)
    return distances, predecessors


def unpack_shortest_path(predecessors, i1, i2):
    path = []
    i = i2
    while i != i1:
        path.append(i)
        i = predecessors[i]
    path.append(i1)
    return path


def build_shifted_indicies(data, indices):
    lines = [
        [[data[line_s, 0], data[line_e, 0]],
         [data[line_s, 1], data[line_e, 1]]]
        for line_s, line_e in indices]
    return lines


def plot_graph(data, graph, sample_n):
    indices_ptr = graph.indptr
    indices = graph.indices
    indices_cross = []
    prev = 0
    for i in indices_ptr[1:]:
        for j in indices[prev:i]:
            indices_cross.append((indices[prev], indices[j]))
            prev = i
            if prev == len(indices):
                break

    lines = build_shifted_indicies(data, indices_cross)
    for index, l in enumerate(lines):
        if indices_cross[index][0] < sample_n:
            plt.plot(l[0], l[1], color='r')
        elif sample_n < indices_cross[index][0] < 2*sample_n:
            plt.plot(l[0], l[1], color='g')
        elif 2*sample_n < indices_cross[index][0] < 3*sample_n:
            plt.plot(l[0], l[1], color='b')
        elif indices_cross[index][0] >= 3*sample_n:
            plt.plot(l[0], l[1], color='black')


from collections import deque


def plot_path(data, path):
    path_shifted = deque(path)
    path_shifted.rotate(1)
    lines = build_shifted_indicies(data, zip(path, path_shifted))
    for l in lines[:-1]:
        plt.plot(l[0], l[1], color='yellow', linewidth=4)


from sklearn.cluster import DBSCAN


def clustering(data, eps=1.0, samples=3):
    clusters = DBSCAN(eps, samples).fit(data)
    return clusters


def process_graph(projected_data, start_point, end_point, sample_n):
    graph = build_graph(projected_data)
    distances, predecessors = shortest_path(graph, start_point)
    path = unpack_shortest_path(predecessors, start_point, end_point)
    plot_graph(projected_data, graph, sample_n)
    plot_path(projected_data, path)
    return path


def pca(data, n_components=2):
    decoposition = PCA(n_components=n_components)
    decoposition.fit(data)
    return decoposition


def kernel_pca(data, n_components=2):
    decoposition = KernelPCA(n_components=n_components, kernel='sigmoid')
    decoposition.fit(data)
    return decoposition


def tsne(data, n_components=2):
    decoposition = TSNE(n_components=n_components)
    decoposition.fit(data)
    return decoposition


def closest_node(node, nodes):
    dist_2 = np.sum((nodes - node) ** 2, axis=-1)
    return np.argmin(dist_2)


from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def sample_and_view(projected_data, n_samples, num_images, paths):
    random_images_cls_1 = sample(paths[0:n_samples], num_images)
    random_images_cls_2 = sample(paths[n_samples:2 * n_samples], num_images)
    random_images_cls_3 = sample(paths[2 * n_samples:3 * n_samples], num_images)
    random_images_cls_4 = sample(paths[3 * n_samples:], num_images)
    ax = plt.subplot(1, 1, 1)
    for i, path in enumerate(random_images_cls_1 + random_images_cls_2 + random_images_cls_3 + random_images_cls_4):
        arr_img = skimage.io.imread(paths[i])
        imagebox = OffsetImage(arr_img, zoom=0.03)
        imagebox.image.axes = ax
        ab = AnnotationBbox(imagebox, (projected_data[i, 0], projected_data[i, 1]),
                            xybox=(0., 0.),
                            xycoords='data',
                            boxcoords="offset points",
                            pad=0)
        ax.add_artist(ab)
    ax.set_xlim(np.min(projected_data[:, 0]), np.max(projected_data[:, 0]))
    ax.set_ylim(np.min(projected_data[:, 1]), np.max(projected_data[:, 1]))
    plt.show()

def get_closest_to_cluster_centroid(cluster, data):
    centroid = np.average(cluster, axis=0)
    index = closest_node(centroid, data)
    return index

def get_closest_to_max_point(cluster, data):
    maximum = np.max(cluster, axis=0)
    index = closest_node(maximum, data)
    return index

def get_closest_to_min_point(cluster, data):
    maximum = np.min(cluster, axis=0)
    index = closest_node(maximum, data)
    return index

def view_closest_image(projected_data, files, sample_n,  pattern = 'Spain'):
    closest_to_pattern = get_closest_to_cluster_centroid(projected_data[2 * sample_n:3 * sample_n], projected_data[3 * sample_n:])
    spain_centroid = get_closest_to_cluster_centroid(projected_data[2*sample_n:3*sample_n], projected_data[2*sample_n:3*sample_n])
    plt.subplot(1,2,1)
    plt.suptitle(files[3 * sample_n + closest_to_pattern])
    plt.imshow(skimage.io.imread(files[3 * sample_n + closest_to_pattern]))
    plt.subplot(1,2,2)
    plt.imshow(skimage.io.imread(files[2*sample_n + spain_centroid]))



import pickle
def analyze(dir, method,
            n_components=2,
            reconstuction=False,
            start_point=0,
            end_point=1,
            n_samples_per_cluster=10,
            eps=10,
            sample_n=300,
            num_images=100,
            frequency_domin=False ,
            multiplyer = 1):
    data, files = read_data(dir, sample_n, frequency_domin, multiplyer)


    decomposition = method(data, n_components)
    with open('decomposition.pkl', 'wb') as output:
        pickle.dump(decomposition, output, pickle.HIGHEST_PROTOCOL)

    projected_data = decomposition.fit_transform(data)

    # sample_and_view(projected_data, sample_n, num_images, files)
    reconstructed = None
    if reconstuction:
        reconstructed = decomposition.inverse_transform(projected_data)

    # clusters = clustering(projected_data, eps, n_samples_per_cluster)
    # labels = [label + 1 for label in clusters.labels_]
    labels = [0 for _ in range(sample_n)] + [1 for _ in range(sample_n)] + [2 for _ in range(sample_n)] + [3 for _ in
                                                                                                           range(
                                                                                                               multiplyer * sample_n)]
    # view_projected_data(projected_data, n_components, labels=labels)
    write_csv('temp/paths.csv', files)
    np.save('temp/projected_data.npy', projected_data)
    write_csv('temp/labels.csv', labels)
    # process_graph(projected_data, start_point, end_point)
    plt.show()
    return projected_data, reconstructed




if __name__ == '__main__':
    n_components = 3
    method = tsne
    sample_n = 85
    num_images = 85
    analyze("./images",
            method,
            n_components,
            start_point=0,
            end_point=9,
            n_samples_per_cluster=3,
            eps=100,
            sample_n=sample_n,
            num_images=num_images,
            frequency_domin=False,
            multiplyer = 1)
