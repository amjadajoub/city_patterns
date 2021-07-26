import analysis
import matplotlib.pyplot as plt
import numpy as np


def view_closest_image(projected_data, files, pattern='spain'):
    if pattern == 'us':
        keys = (0, 1)
    elif pattern == 'france':
        keys = (1, 2)
    elif pattern == 'spain':
        keys = (2, 3)
    closest_to_pattern = analysis.get_closest_to_cluster_centroid(projected_data[keys[0] * sample_n:keys[1] * sample_n],
                                                                  projected_data[3 * sample_n:])
    pattern_centroid = analysis.get_closest_to_cluster_centroid(projected_data[keys[0] * sample_n:keys[1] * sample_n],
                                                                projected_data[keys[0] * sample_n:keys[1] * sample_n])
    ax = plt.subplot(1, 2, 1)
    img = np.load(files[keys[0] * sample_n + pattern_centroid])
    plt.imshow(img, cmap="Greys")
    name = files[keys[0] * sample_n + pattern_centroid].split("/")[-1][:-4]
    ax.title.set_text(f"Pattern centroid {pattern.capitalize()}: {name}")
    plt.axis("off")

    ax = plt.subplot(1, 2, 2)
    img = np.load(files[3 * sample_n + closest_to_pattern])
    plt.imshow(img, cmap="Greys")
    name = files[3 * sample_n + closest_to_pattern].split("/")[-1][:-4]
    ax.title.set_text(f"Most similar in Germany:{name}")
    plt.axis("off")
    plt.show()
    return keys[0] * sample_n + pattern_centroid, 3 * sample_n + closest_to_pattern


sample_n = 85
multiplyer = 450
image_size = (32, 32)
frequency_domin = False
normlized = False
n_components = 2
images = np.load("X_500.npy")
import pickle

with open("paths.txt", "rb") as fp:  # Unpickling
    paths = pickle.load(fp)


from tensorflow.keras.models import load_model
model = load_model("encoder_200")
projected_data = model.predict(images)
for pattern in ["us", "france", "spain"]:
    pattern_centroid, closest_to_pattern = view_closest_image(projected_data, paths, pattern=pattern)
    path = analysis.process_graph(projected_data, closest_to_pattern, pattern_centroid, sample_n)
    plt.show()

fig = plt.figure(figsize=(len(path), 1))
from matplotlib.ticker import NullFormatter

plt.Axes(fig, [0., 0., 1., 1.])
for index, i in enumerate(path):
    ax = plt.subplot(1, len(path), index + 1)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.imshow(np.load(paths[i]))

plt.tight_layout()
plt.show()

import analysis
import matplotlib.pyplot as plt
import numpy as np


sample_n = 85
multiplyer = 450
image_size = (64, 64)
frequency_domin = False
normlized = False
n_components = 2


images = np.load("X_500.npy")
import pickle
with open("paths.txt", "rb") as fp:  # Unpickling
    paths = pickle.load(fp)


from tensorflow.keras.models import load_model
model = load_model("encoder_200")
projected_data = model.predict(images)

query_image = np.random.random_integers(0, 500)
dist_2 = np.sum((projected_data - projected_data[query_image, ...]) ** 2, axis=-1)
n = 5
plt.subplot(2,3, 1)
plt.imshow(images[query_image, ...].reshape(image_size), cmap="Greys")
plt.axis("off")
indices = (dist_2).argsort()[2:n+2]
for counter, i in enumerate(list(indices)):
    plt.subplot(2,3, counter+2)
    plt.imshow(images[i, ...].reshape(image_size), cmap="Greys")
    plt.axis("off")
plt.show()

import analysis
import matplotlib.pyplot as plt
import numpy as np

sample_n = 85
multiplyer = 450
image_size = (64, 64)
frequency_domin = False
normlized = False
n_components = 2

images = np.load("X_100.npy")
import pickle

with open("paths.txt", "rb") as fp:  # Unpickling
    paths = pickle.load(fp)

from tensorflow.keras.models import load_model

model = load_model("encoder_200")
decoder = load_model("decoder_200")
projected_data = model.predict(images)

pattern = "us"
if pattern == 'us':
    keys = (0, 1)
elif pattern == 'france':
    keys = (1, 2)
elif pattern == 'spain':
    keys = (2, 3)

pattern_1 = projected_data[keys[0] * sample_n:keys[1] * sample_n]
pattern_max = analysis.get_closest_to_max_point(pattern_1, pattern_1)
pattern_1_index = int(keys[0] * sample_n + pattern_max)

pattern = "france"
if pattern == 'us':
    keys = (0, 1)
elif pattern == 'france':
    keys = (1, 2)
elif pattern == 'spain':
    keys = (2, 3)

pattern_2 = projected_data[keys[0] * sample_n:keys[1] * sample_n]
pattern_min = analysis.get_closest_to_min_point(pattern_2, pattern_2)
pattern_2_index = int(keys[0] * sample_n + pattern_min)

path = analysis.process_graph(projected_data, pattern_1_index, pattern_2_index, sample_n)
print(path)
plt.show()

from matplotlib.ticker import NullFormatter

for i in range(len(path)):
    ax = plt.subplot(5, 4, i + 1)
    plt.imshow(images[path[i]].reshape(image_size), cmap="Greys")
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
plt.show()

import itertools

imgs = []
for i in range(len(path)-1):
    mid_seg = (projected_data[path[i + 1]] - projected_data[path[i]]) / 2
    point = projected_data[path[i]] + mid_seg
    img = decoder.predict(np.array([point]))[0].reshape(image_size)
    imgs.append(images[path[i]].reshape(image_size))
    imgs.append(img)
imgs.append(images[path[-1]].reshape(image_size))

for i, img in enumerate(imgs):
    ax = plt.subplot(6, 6, i + 1)
    plt.imshow(img, cmap="Greys")
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
plt.show()

point_start = images[path[8]].reshape(image_size)

plt.figure(figsize=(24, 10))
ax = plt.subplot(2, 6, 1)
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
ax.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
plt.imshow(point_start, cmap="Greys")
selected_idx = 8
steps = 10
step = (projected_data[path[selected_idx+1]] - projected_data[path[selected_idx]]) / steps
imgs = []
imgs.append(point_start)

for i in range(10):
    ax = plt.subplot(2, 6, i + 2)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
    point = projected_data[path[selected_idx]] + (i + 1) * step
    img = decoder.predict(np.array([point]))[0].reshape(image_size)
    imgs.append(img)
    plt.imshow(img, cmap="Greys")

point_end = images[path[selected_idx+1]].reshape(image_size)
imgs.append(point_end)
ax = plt.subplot(2, 6, 12)
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
ax.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
plt.tight_layout()
plt.imshow(point_end, cmap="Greys")
import time
plt.savefig(f'temp/interpolation_{int(time.time())}.png')
plt.show()


import analysis
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import decomposition
import numpy as np

sample_n = 85
multiplyer = 10
image_size = (128, 128)
frequency_domin = False
normlized = True
components = 6

ax = plt.subplot(1, 1, 1)
X, _ = analysis.read_data("images",
                          sample_n=sample_n,
                          multiplyer=multiplyer,
                          frequency_domin=frequency_domin,
                          image_size=image_size,
                          normlized=normlized)
non_zero_count = np.count_nonzero(X, axis=0)
X = X * 1/non_zero_count

pca = decomposition.PCA(n_components=components)
pc = pca.fit_transform(X)
df = pd.DataFrame({'Variation %': pca.explained_variance_ratio_*100,
             'PC':[f"PC{i+1}" for i in range(components)]})
sns.barplot(x='PC',y="Variation %", data=df, color="g")
plt.show()

for i in range(components):
    ax = plt.subplot(2,3,i+1)
    plt.imshow(pca.components_[i].reshape(image_size), cmap="Greys")
    ax.title.set_text(f"PC{i+1}")
    plt.axis("off")
plt.show()
