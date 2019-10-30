import argparse

import torch as t
import torch.nn as nn
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim import Adam
from torchvision import datasets

from models import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--num-epochs', type=int, default=100, metavar='NI',
                        help='num epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=1000, metavar='BS',
                        help='batch size (default: 70)')
    parser.add_argument('--use-cuda', type=bool, default=True, metavar='CUDA',
                        help='use cuda (default: False)')
    parser.add_argument('--learning-rate', type=float, default=0.0005, metavar='LR',
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--mode', type=str, default='vardropout', metavar='M',
                        help='training mode (default: simple)')
    parser.add_argument('--data_path', type=str, default='/home/zhezhihe/data/')
    
    
    args = parser.parse_args()

    writer = SummaryWriter(args.mode)

    assert args.mode in ['simple', 'dropout', 'vardropout'], 'Invalid mode, should be in [simple, dropout, vardropout]'
    Model = {
        'simple': SimpleModel,
        'dropout': DropoutModel,
        'vardropout': VariationalDropoutModel
    }
    Model = Model[args.mode]

    dataset = datasets.MNIST(root=args.data_path,
                             transform=transforms.Compose([
                                 transforms.ToTensor()]),
                             download=True,
                             train=True)
    train_dataloader = t.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    dataset = datasets.MNIST(root=args.data_path,
                             transform=transforms.Compose([
                                 transforms.ToTensor()]),
                             download=True,
                             train=False)
    test_dataloader = t.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = Model()
    if args.use_cuda:
        model.cuda()

    optimizer = Adam(model.parameters(), args.learning_rate, eps=1e-6)

    cross_enropy_averaged = nn.CrossEntropyLoss(reduction='mean')

    for epoch in range(args.num_epochs):
        for iteration, (input, target) in enumerate(train_dataloader):

            input = Variable(input).view(-1, 784)
            target = Variable(target)

            if args.use_cuda:
                input, target = input.cuda(), target.cuda()

            optimizer.zero_grad()

            loss = None
            if args.mode == 'simple':
                loss = model.loss(input=input, target=target, average=True)
            elif args.mode == 'dropout':
                loss = model.loss(input=input, target=target, p=0.4, average=True)
            else:
                likelihood, kld = model.loss(input=input, target=target, train=True, average=True)
                coef = min(epoch / 40., 1.)
                loss = likelihood + kld * coef

            loss.backward()
            optimizer.step()

            if iteration % 50 == 0:
                print('train epoch {}, iteration {}, loss {}'.format(epoch, iteration, loss.cpu().data.numpy()))


            if iteration % 100 == 0:
                loss = 0
                for input, target in test_dataloader:
                    input = Variable(input).view(-1, 784)
                    target = Variable(target)

                    if args.use_cuda:
                        input, target = input.cuda(), target.cuda()

                    if args.mode == 'simple':
                        loss += model.loss(input=input, target=target, average=False).cpu().data.numpy()
                    elif args.mode == 'dropout':
                        loss += model.loss(input=input, target=target, p=0., average=False).cpu().data.numpy()
                    else:
                        loss += model.loss(input=input, target=target, train=False, average=False).cpu().data.numpy()

                loss = loss / (args.batch_size * len(test_dataloader))
                print('-'*60)
                print('valid epoch {}, iteration {}'.format(epoch, iteration))
                print('-'*60)
                print('valid epoch {}, loss {}'.format(epoch, loss))
                print('-'*60)
                
                # logging using the tensorboardX
                writer.add_scalar('data/loss', loss, epoch * len(train_dataloader) + iteration) # val loss
                
                # conditionally log the variational dropout
                if args.mode == 'vardropout':
                    for name, param in model.named_parameters():
                        writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch * len(train_dataloader) + iteration)

    writer.close()
