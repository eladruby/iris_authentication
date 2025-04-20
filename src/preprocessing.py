import os
import numpy as np
import tensorflow as tf
from iris_url_normalization import IrisURLNormalization

class PreProcessing:
    image_train = np.array([])
    label_train = np.array([])
    unique_train_label = np.array([])
    map_train_label_indices = dict()

    def __init__(self, data_src, TRIPLETS):
        self.TRIPLETS = TRIPLETS
        self.data_src = data_src
        print("Loading the Dataset...")
        self.image_train, self.label_train = self.preprocessing()
        self.unique_train_label = np.unique(self.label_train)
        self.map_train_label_indices = {
            label: np.flatnonzero(self.label_train == label)
            for label in self.unique_train_label
        }
        print("Preprocessing Done. Summary:")
        print("Images trained: ", self.image_train.shape)
        print("Labels trained: ", self.label_train.shape)
        print("Unique labels: ", self.unique_train_label)

    def read_dataset(self):
        count = 0
        directories = os.listdir(self.data_src)
        for directory in directories:
            count += len([file for file in os.listdir(os.path.join(self.data_src, directory))])
        x = [None] * count
        y = [None] * count
        index = 0
        for directory in directories:
            try:
                print("Reading Directory: ", directory)
                for pic in os.listdir(os.path.join(self.data_src, directory)):
                    img = IrisURLNormalization(os.path.join(self.data_src, directory, pic))
                    img = tf.keras.preprocessing.image.img_to_array(img)
                    x[index] = img
                    y[index] = directory
                    index += 1
            except Exception as e:
                print("Error reading directory: ", directory)
                print(e)
        print("Reading Dataset Done.")
        return x, y

    def preprocessing(self):
        x, y = self.read_dataset()
        x, y = zip(*[(xi, yi) for xi, yi in zip(x, y) if yi is not None])
        labels = list(set(y))
        labels.sort(key=int)
        label_dict = dict(zip(labels, range(len(labels))))
        y = np.asarray([label_dict[label] for label in y])
        shuffle_indices = np.random.permutation(len(y))
        x_shuffled = [x[i] for i in shuffle_indices]
        y_shuffled = [y[i] for i in shuffle_indices]
        return np.asarray(x_shuffled), np.asarray(y_shuffled)

    def get_triplets(self):
        label_l, label_r = np.random.choice(self.unique_train_label, 2, replace=False)
        a, p = np.random.choice(self.map_train_label_indices[label_l], 2, replace=False)
        n = np.random.choice(self.map_train_label_indices[label_r])
        return a, p, n

    def get_triplets_batch(self):
        indexs_a, indexs_p, indexs_n = [], [], []
        for _ in range(self.TRIPLETS):
            a, p, n = self.get_triplets()
            indexs_a.append(a)
            indexs_p.append(p)
            indexs_n.append(n)
        anchor_dataset = tf.data.Dataset.from_tensor_slices(self.image_train[indexs_a, :])
        positive_dataset = tf.data.Dataset.from_tensor_slices(self.image_train[indexs_p, :])
        negative_dataset = tf.data.Dataset.from_tensor_slices(self.image_train[indexs_n, :])
        return tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
