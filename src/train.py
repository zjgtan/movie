# coding: utf8
from data import *
from model import *
import torch
import sys
import math
import matplotlib.pyplot as plt

dataset = ML1M("./ml-1m")

train_set, test_set = dataset.train_set, dataset.test_set
test_features, test_targets = next(mini_batch_iterator(test_set, dataset.output_types, len(test_set)))

emb_dim = 2
num_epoch = 20
batch_size = 2048
learning_rate = 0.01
lamda2 = 0.01

model = UserItemRatingRegressor(num_user_ids = dataset.num_user_ids, 
        num_movie_ids = dataset.num_movie_ids, 
        emb_dim = emb_dim)

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

train_losses = []
test_losses = []

for epoch in range(num_epoch):
    running_loss = 0.
    num_batchs = 0.
    for ix, (features, targets) in enumerate(mini_batch_iterator(train_set, dataset.output_types, batch_size)):
        model.zero_grad()
        preds = model(*features)
        loss = model.loss_function(preds, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batchs += 1

    train_losses.append(running_loss / num_batchs)

    test_preds = model(*test_features)
    test_loss = model.loss_function(test_preds, test_targets)
    print >> sys.stderr, "epoch: %d, train rmse: %f, test rmse: %f" % (epoch, math.sqrt(running_loss / num_batchs), math.sqrt(loss.item()))
    test_losses.append(loss)
    if epoch >= 1 and loss > test_losses[-2]:
        break

#plt.plot([ix for ix in range(num_epoch)], train_losses)
plt.plot([ix for ix in range(num_epoch)], test_losses)
plt.show()



