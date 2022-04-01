import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
from model import ResNet
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import torchvision.models as models

# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
csv_path = ''
for root, dirs, files in os.walk('.'):
    for name in files:
        if name == 'data.csv':
            csv_path = os.path.join(root, name)
data = pd.read_csv(csv_path, sep=';')
train_data, val_data = train_test_split(data, test_size=0.1, shuffle=False)  # no shuffle

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
train_data, val_data = ChallengeDataset(train_data, 'train'), ChallengeDataset(val_data, 'val')   # image transform for training data is included in Challengedataset
train_data, val_data = t.utils.data.DataLoader(train_data, batch_size=32, shuffle=True), t.utils.data.DataLoader(val_data, batch_size=32, shuffle=False)
#when training,shuffle is acceptable

# create an instance of our ResNet model
model = ResNet()
model = model.cuda()

# using a pre-trained model
model_pre = models.resnet18(pretrained=True)

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# set up the optimizer (see t.optim)
# create an object of type Trainer and set its early stopping criterion
criterion = t.nn.BCELoss()
criterion = criterion.cuda()
optimizer = t.optim.Adam(model.parameters(), lr=0.001)
# optimizer = t.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.01)
trainer = Trainer(model, criterion, optimizer, train_data, val_data, True, 30)    # The best results since now came from one run with patience = 100

# go, go, go... call fit on trainer
res = trainer.fit(100)

# save the model
best_epoch = trainer.best_epoch
trainer.restore_checkpoint(best_epoch)  # load the best parameters
print(best_epoch)

with open("/home/cip/medtech2019/en50oweb/DLex4/src/model.onnx", "wb+") as f:
    trainer.save_onnx(f)

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')
