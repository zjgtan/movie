# coding: utf8
import numpy as np
import torch

class ML1M:
    def __init__(self, path):
        self.path = path
        self.user_dict = self.load_user_dict(path + "/users.dat")
        self.movie_dict = self.load_movie_dict(path + "/movies.dat")
        self.ratings = self.load_rating_data(path + "/ratings.dat")

        self.num_user_ids = 6041
        self.num_movie_ids = 3953

        self.instances, self.output_types = self.build_instances()

        np.random.shuffle(self.instances)
        
        self.train_set, self.test_set = self.instances[:int(len(self.instances) * 0.7)], self.instances[int(len(self.instances) * 0.7):]

    def load_user_dict(self, filename):
        d = {}

        return d

    def load_movie_dict(self, filename):
        d = {}
        return d

    def load_rating_data(self, filename):
        ratings = []

        for line in file(filename):
            user_id, movie_id, rating, timestamp = map(int, line.rstrip().split("::"))
            ratings.append([user_id, movie_id, float(rating)])

        return ratings

    def build_instances(self):
        
        output_types = [torch.LongTensor, torch.LongTensor, torch.FloatTensor]
        return self.ratings, output_types


def mini_batch_iterator(instances, output_types,  batch_size):
    mini_batch = [[] for _ in range(len(instances[0]))]
    count = 0
    for instance in instances:
        if count == batch_size:
            mini_batch = [output_type(mini_batch[ix]) for ix, output_type in enumerate(output_types)]
            yield mini_batch[:-1], mini_batch[-1]
            count = 0
            mini_batch = [[] for _ in range(len(instances[0]))]

        for ix, elem in enumerate(instance):
            if isinstance(elem, list) != True:
                elem = [elem]
            mini_batch[ix].append(elem)

        count += 1

    if count != 0:
        mini_batch = [output_type(mini_batch[ix]) for ix, output_type in enumerate(output_types)]
        yield mini_batch[:-1], mini_batch[-1]


if __name__ == "__main__":
    ml1m_dataset = ML1M("./ml-1m")

    for batch in mini_batch_iterator(ml1m_dataset.train_set, 
            ml1m_dataset.output_types, 10):
        print batch
        break




        





