import analysis
sample_n = 85
multiplyer = 489
image_size = (64, 64)
frequency_domin = False
normlized = False
import numpy as np
X = np.load("X_500.npy")
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}

plt.rc('font', **font)

plt.figure(figsize=(16,4))
ax = plt.subplot(1,4,1)
us = X[:multiplyer, :]
avg = np.average(us, axis=0)
avg = avg.reshape(image_size)
avg = avg * -1
plt.imshow(avg, cmap="Greys")
plt.axis("off")
ax.title.set_text("New York City")

ax = plt.subplot(1,4,2)
france = X[multiplyer:2*multiplyer, :]
avg = np.average(france, axis=0)
avg = avg.reshape(image_size)
avg = avg * -1
plt.imshow(avg, cmap="Greys")
plt.axis("off")
ax.title.set_text("Paris")

ax = plt.subplot(1,4,3)
spain = X[2*multiplyer:3*multiplyer, :]
avg = np.average(spain, axis=0)
avg = avg.reshape(image_size)
avg = avg * -1
plt.imshow(avg, cmap="Greys")
plt.axis("off")
ax.title.set_text("Barcelona")

ax = plt.subplot(1,4,4)
germany = X[3*multiplyer:, :]
avg = np.average(germany, axis=0)
avg = avg.reshape(image_size)
avg = avg * -1
plt.imshow(avg, cmap="Greys")

plt.axis("off")
ax.title.set_text("Germany")

plt.show()
