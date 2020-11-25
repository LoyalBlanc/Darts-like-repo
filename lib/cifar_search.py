import argparse
import logging
import os
from time import time

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam, SGD

from lib.dataset import get_cifar10_train_loader, AvgMeter, accuracy_topk
from lib.basic import BasicSearchNetwork, BasicClassifyHead


class BasicSearchCIFAR(nn.Module):
    def __init__(self, device):
        super(BasicSearchCIFAR, self).__init__()

        self.feature_extractor = BasicSearchNetwork("softmax").to(device)
        self.classifier = BasicClassifyHead(self.feature_extractor.out_channel, 10).to(device)

        self.criterion = nn.CrossEntropyLoss().to(device)

    def forward(self, img, lab):
        feature = self.feature_extractor(img)
        result = self.classifier(feature)
        temp_loss = self.criterion(result, lab)
        temp_acc = accuracy_topk(lab, result)[0]
        return temp_loss, temp_acc


def main():
    parser = argparse.ArgumentParser("CIFAR10")
    parser.add_argument('--seed', type=int, default=19, help='random seed')

    parser.add_argument('--data_path', type=str, default='./data', help='location of the data corpus')
    parser.add_argument('--log_path', type=str, default=f'./log/cifar10_search{time()}.log', help='log save path')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')

    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')

    parser.add_argument('--learning_rate', type=float, default=0.005, help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')

    parser.add_argument('--alpha_learning_rate', type=float, default=3e-3, help='learning rate for alpha')
    parser.add_argument('--alpha_weight_decay', type=float, default=1e-3, help='weight decay for alpha')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, filename=args.log_path, filemode='w+')

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    np.random.seed(args.seed)
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.backends.cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)

    train_queue = get_cifar10_train_loader(args.data_path, args.batch_size)

    model = BasicSearchCIFAR(torch.device(f"cuda:0")).cuda()
    model_optimizer = SGD([param for name, param in model.named_parameters() if 'alpha' not in name],
                          args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        model_optimizer, args.epochs, eta_min=args.learning_rate_min, last_epoch=-1)
    alpha_optimizer = Adam([param for name, param in model.named_parameters() if 'alpha' in name],
                           args.alpha_learning_rate, betas=(0.9, 0.999), weight_decay=args.alpha_weight_decay)

    loss_value = AvgMeter()
    top1 = AvgMeter()
    best_acc = [0, 0]
    for epoch in range(1, args.epochs + 1):
        for step, (image, label) in enumerate(train_queue, 1):
            model.train()
            n = label.size(0)
            image = Variable(image, requires_grad=False).cuda(non_blocking=True)
            label = Variable(label, requires_grad=False).cuda(non_blocking=True)

            step_loss, acc = model(image, label)

            model_optimizer.zero_grad()
            alpha_optimizer.zero_grad()
            step_loss.backward()
            model_optimizer.step()
            alpha_optimizer.step()

            loss_value.update(step_loss.item(), n)
            top1.update(acc.item(), n)
            if step % 50 == 0:
                logging.info(f"Step {step}, loss {np.log(loss_value.avg+1e-16):.4f}, top1 {top1.avg:.2%}")
        if top1.avg > best_acc[0]:
            best_acc = [top1.avg, epoch]
        scheduler.step()

        genotype = model.feature_extractor.genotype()
        logging.info(f"Epoch [{epoch}/{args.epochs}], lr {scheduler.get_last_lr()[0]}, "
                     f"Best Accuracy {best_acc[0]:.2%}[{best_acc[1]}]\n{genotype}")
