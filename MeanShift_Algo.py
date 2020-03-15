
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11], [8, 2], [10, 2], [9, 3]])
colors = 10 * ['g', 'r', 'c', 'b', 'k']


class MeanShift:
    def __init__(self, radius=None, radius_units=100):
        self.radius = radius
        self.radius_units = radius_units

    def fit(self, data):
        if self.radius is None:
            start_centroid = np.average(data, axis=0)
            distances_to_start_centroids = [np.linalg.norm(feature_set-start_centroid)
                                            for feature_set in data]
            self.radius = max(distances_to_start_centroids)/self.radius_units
        print_test = True
        centroids = {}
        scores = [i for i in range(self.radius_units)][::-1]  # reverse the list
        for i in range(len(data)):
            centroids[i] = data[i]
        optimized = False
        while not optimized:
            new_centroids = []
            # for each centroid, identify other points in its bandwidth and calculate a new centroid
            for centroid_classification in centroids:
                centroid = centroids[centroid_classification]
                in_bandwidth = []
                for feature_set in data:
                    distance = np.linalg.norm(feature_set-centroid)
                    if distance == 0:
                        distance == 0.000000001
                    score_index = int(distance/self.radius)  # classify distance
                    if score_index >= self.radius_units - 1:  # if too far away
                        # set it to the largest index with lowest score
                        score_index = self.radius_units - 1
                    to_add = ((scores[score_index]**2)*[feature_set])  # penalize for being too far
                    in_bandwidth += to_add
                    if print_test:
                        print("feature_set:", feature_set)
                        # close points being awarded, far points being penalized
                        print("to_add:", len(to_add), "of", feature_set)
                print_test = False
                new_centroid = np.average(in_bandwidth, axis=0)
                new_centroids.append(tuple(new_centroid))
            uniques = self.extract_unique_elements(new_centroids)

            prev_centroids = dict(centroids)
            print("prev_centroids:", prev_centroids)
            print("new_unique_centroids:", uniques)
            centroids = {}
            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])
            step_optimized = True
            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    print("comparing: ", centroids[i], "and", prev_centroids[i])
                    step_optimized = False
                if not step_optimized:
                    break
            print(100 * '#')
            # when the entire centroids dict do not change
            if step_optimized:
                optimized = True

        self.centroids = centroids
        print("estimated_centroids:", self.centroids)
        print(100 * '#')

    def extract_unique_elements(self, data):
        uniques = sorted(list(set(data)))  # set of tuples
        # need to remove centroids that are really close from each other
        similar_data = []
        for i in range(len(uniques) - 1):
            for j in range(i + 1, len(uniques) - 1):
                if np.linalg.norm(np.array(uniques[i]) - np.array(uniques[j])) <= self.radius:
                    if uniques[j] not in similar_data:
                        similar_data.append(uniques[j])
        for similar_centroid in similar_data:
            uniques.remove(similar_centroid)
        return uniques

    def predict(self, data):
        distances = [np.linalg.norm(data-self.centroids[classification]) for classification in self.centroids]
        prediction = distances.index(min(distances))
        print("predicted classification of ", data, ": class", prediction)
        return prediction


clf = MeanShift()
clf.fit(X)
centroids = clf.centroids
clf.predict([6, 8])

plt.scatter(X[:, 0], X[:, 1], s=150)
for c in centroids:
    plt.scatter(centroids[c][0], centroids[c][1], color='k', marker='*', s=150)
plt.show()

