# coding: utf8
import torch
import torch.nn as nn

class UserItemRatingRegressor(nn.Module):
    def __init__(self, num_user_ids, num_movie_ids, emb_dim):
        super(UserItemRatingRegressor, self).__init__()

        self.user_id_bias_layer = torch.nn.Embedding(num_user_ids, 1)
        self.movie_id_bias_layer = torch.nn.Embedding(num_movie_ids, 1)
        self.user_id_emb_layer = torch.nn.Embedding(num_user_ids, emb_dim)
        self.movie_id_emb_layer = torch.nn.Embedding(num_movie_ids, emb_dim)
        self.loss_function = nn.MSELoss()

    def forward(self, user_id, movie_id):
        user_id_bias = self.user_id_bias_layer(user_id)
        movie_id_bias = self.movie_id_bias_layer(movie_id)
        user_id_emb = self.user_id_emb_layer(user_id)
        movie_id_emb = self.movie_id_emb_layer(movie_id)

        merge = torch.cat(
                (user_id_bias, 
                    movie_id_bias, 
                    torch.mul(user_id_emb, movie_id_emb)), 2)

        pred = torch.sum(
                input = merge,
                dim = 2, keepdim = False)

        return pred

if __name__ == "__main__":
    user_ids = torch.LongTensor([[1], [2]])
    movie_ids = torch.LongTensor([[1], [2]])
    ratings = torch.FloatTensor([[1.0], [5.0]])

    model = UserItemRatingRegressor(10, 10, 2)
    model(user_ids, movie_ids)


