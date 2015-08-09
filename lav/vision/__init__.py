from __future__ import division
import numpy as np
from numpy import argsort, sqrt


class Vision:

    def __init__(self):
        pass

    @staticmethod
    def compute_visual_similarity(distance, minimum_distance, maximum_distance):

        dist = np.divide(np.longdouble(distance - minimum_distance), np.longdouble(maximum_distance - minimum_distance))
        similarity = 1 - dist

        if similarity == 0:
            similarity = 0.000001  # prevent division issues
        elif similarity > 1:
            similarity = 1  # prevent floating point issues

        return similarity

    @staticmethod
    def find_nearest_neighbours(x, d, k):
        """ find K nearest neighbours of data among D """

        data = d.shape[1]
        k = k if k < data else data
        sqd = sqrt(((d - x[:, :data]) ** 2).sum(axis=0))  # euclidean distances from the other points
        idx = argsort(sqd)  # sorting
        idx = idx[:k]  # return the indexes of K nearest neighbours
        return sqd[idx], idx

    @staticmethod
    def remove_outliers(candidates, column_indice, epsilon, number_of_nearest_neighbours):
        """ Remove outliers adaptively based on distance and a treshold value """

        remaining = len(candidates)
        visual_distance_scores = []

        try:
            for i in range(len(candidates)):
                visual_distance_scores.append(float(candidates[i][column_indice]))
        except Exception, e:
            print str(e)
            print candidates

        dist_min = min(visual_distance_scores)
        dist_max = max(visual_distance_scores)
        ind2remove = []
        # reverse the list, so that we can start removing from the furthest score
        candidates.sort(key=lambda c: c[column_indice], reverse=True)

        for i in range(len(candidates)):
            if float(candidates[i][column_indice]) > (1 + epsilon) * dist_min:
                if remaining > number_of_nearest_neighbours:  # Make sure we have at least some items left.
                    ind2remove.append(i)
                    remaining -= 1
                elif remaining == number_of_nearest_neighbours:
                    break
        # candidates = np.delete(candidates, idx, axis=0) # remove outliers
        candidates = [x for i, x in enumerate(candidates) if i not in ind2remove]
        # candidates = candidates.tolist()
        candidates.reverse()
        return candidates, dist_min, dist_max

    def retrieve_visually_similar_images(self, query_item, candidate_items, number_of_nearest_neighbours, settings):
        """Retrieve visually similar images to the query image using the extracted CNN features."""

        data = []
        for i, item in enumerate(candidate_items):
            data.append(item[int(settings["vggindex"])])

        data = np.array(data)
        data = data.transpose()

        query = [query_item[int(settings["vggindex"])]]
        query = np.array(query)
        query = query.transpose()

        distances, indices = self.find_nearest_neighbours(query, data, number_of_nearest_neighbours)
        indices = indices.tolist()
        visually_similar_images = [candidate_items[i] for i in indices]

        for i, item in enumerate(visually_similar_images):
            item.append(distances[i])

        return visually_similar_images
