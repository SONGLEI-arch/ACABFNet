import numpy as np
np.set_printoptions(threshold=np.inf)

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from assess import hist_sum, compute_metrics
# from pytorchtools import EarlyStopping
# from gary2RGB import create_visual_anno

from model import CrossNet

from LEVIR_CDdataset import BuildingChangeDataset
# from datadataset import BuildingChangeDataset
# from CDDdataset import CDDDataset

from index2one_hot import get_one_hot

from poly import adjust_learning_rate_poly

import warnings
warnings.filterwarnings('ignore')

# torch.backends.cudnn.enabled = False
# =========================================================#
# =========================================================#
# =========================================================#
train_data = BuildingChangeDataset(mode='train')
data_loader = DataLoader(train_data, batch_size=4, shuffle=True)

test_data = BuildingChangeDataset(mode='test')
test_data_loader = DataLoader(test_data, batch_size=4, shuffle=False)

Epoch = 200
lr = 1e-4
n_class = 2
# ,head=[1,4,8,12]
net = CrossNet(n_class,[4,8,16,32]).cuda()
# edge = edge_conv2d().cuda()

criterion = nn.BCEWithLogitsLoss().cuda()
# focal = FocalLoss(gamma=2, alpha=0.25).cuda()
optimizer = optim.Adam(net.parameters(), lr=lr)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.9,patience=5)

with open(r'F:\LEVIR_CD\levir_dataset\train.txt', 'a') as f:
    for epoch in range(Epoch):
        # print('lr:', optimizer.state_dict()['param_groups'][0]['lr'])

        torch.cuda.empty_cache()

        new_lr = adjust_learning_rate_poly(optimizer, epoch, Epoch, lr, 0.9)
        print('lr:', new_lr)

        _train_loss = 0

        _hist = np.zeros((n_class, n_class))

        net.train()
        for before, after, change in tqdm(data_loader, desc='epoch{}'.format(epoch), ncols=100):
            before = before.cuda()
            after = after.cuda()

            # ed_change = change.cuda()
            # ed_change = edge(ed_change)
            # lbl = torch.where(ed_change > 0.1, 1, 0)
            # plt.figure()
            # plt.imshow(lbl.data.cpu().numpy()[0][0], cmap='gray')
            # plt.show()
            # lbl = lbl.squeeze(dim=1).long().cpu()
            # lbl_one_hot = get_one_hot(lbl, 2).permute(0, 3, 1, 2).contiguous().cuda()

            change = change.squeeze(dim=1).long()
            change_one_hot = get_one_hot(change, 2).permute(0, 3, 1, 2).contiguous().cuda()

            optimizer.zero_grad()

            pred = net(before, after)
            loss_pred = criterion(pred, change_one_hot)
            loss = loss_pred

            loss.backward()
            optimizer.step()
            _train_loss += loss.item()

            label_pred = F.softmax(pred, dim=1).max(dim=1)[1].data.cpu().numpy()
            label_true = change.data.cpu().numpy()

            hist = hist_sum(label_true, label_pred, 2)

            _hist += hist

        # scheduler.step()

        precision, recall, miou, F1 = compute_metrics(_hist)

        trainloss = _train_loss / len(data_loader)

        print('Epoch:', epoch, ' |train loss:', trainloss, ' |train precision:', precision, ' |train recall:',
              recall, ' |train miou:', miou, ' |train F1:', F1)
        f.write(
            'Epoch:%d|train loss:%0.06f|train precision:%0.06f|train recall:%0.06f|train miou:%0.06f|train F1:%0.06f' % (
                epoch, trainloss, precision, recall, miou, F1))
        f.write('\n')
        f.flush()
        torch.save(net, r'F:\LEVIR_CD\levir_dataset\network\epoch_{}.pth'.format(epoch))


        with torch.no_grad():
            with open(r'F:\LEVIR_CD\levir_dataset\test.txt', 'a') as f1:
                torch.cuda.empty_cache()

                _test_loss = 0

                _hist = np.zeros((n_class, n_class))

                k = 0

                net.eval()
                for before, after, change in tqdm(test_data_loader, desc='epoch{}'.format(epoch), ncols=100):
                    before = before.cuda()
                    after = after.cuda()
                    change = change.squeeze(dim=1).long()
                    change_one_hot = get_one_hot(change, 2).permute(0, 3, 1, 2).contiguous().cuda()

                    pred = net(before, after)

                    loss = criterion(pred, change_one_hot)

                    _test_loss += loss.item()

                    label_pred = F.softmax(pred, dim=1).max(dim=1)[1].data.cpu().numpy()
                    label_true = change.data.cpu().numpy()

                    hist = hist_sum(label_true, label_pred, 2)

                    _hist += hist

                precision, recall, miou, F1 = compute_metrics(_hist)

                testloss = _test_loss / len(test_data_loader)

                print('Epoch:', epoch, ' |test loss:', testloss, ' |test precision:', precision, ' |test recall:',
                      recall, ' |test miou:', miou, ' |test F1:', F1)
                f1.write(
                    'Epoch:%d|test loss:%0.06f|test precision:%0.06f|test recall:%0.06f|test miou:%0.06f|test F1:%0.06f' % (
                    epoch, testloss, precision, recall, miou, F1))
                f1.write('\n')
                f1.flush()
        # scheduler.step(testloss)
