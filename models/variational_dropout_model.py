import torch.nn as nn
import torch.nn.functional as F

from variational_dropout.variational_dropout import VariationalDropout


class VariationalDropoutModel(nn.Module):
    def __init__(self):
        super(VariationalDropoutModel, self).__init__()

        self.fc = nn.ModuleList([
            VariationalDropout(784, 500),
            VariationalDropout(500, 50),
            nn.Linear(50, 10)
        ])

    def forward(self, input):
        """
        :param input: An float tensor with shape of [batch_size, 784]
        :param train: An boolean value indicating whether forward propagation called when training is performed
        :return: An float tensor with shape of [batch_size, 10]
                 filled with logits of likelihood and kld estimation
        """

        result = input

        if self.training:
            kld_sum = 0 # initialize the KL divergence

            for i, layer in enumerate(self.fc): # enumerate through the entire model
                if i != len(self.fc) - 1:
                    result, kld = layer(result) # get the hidden activation and their KL-divergence
                    result = F.elu(result)
                    kld_sum += kld # accumulate the KL-divergence

            return self.fc[-1](result), kld_sum
        
        # if not in the train mode, directly return the result
        for i, layer in enumerate(self.fc):
            if i != len(self.fc) - 1:
                result = F.elu(layer(result))

        return self.fc[-1](result)

    def loss(self, **kwargs):
        if kwargs['train']:
            self.train() # change training flag
            out, kld = self(kwargs['input'])
            return F.cross_entropy(out, kwargs['target'], size_average=kwargs['average']), kld
        
        self.eval() # change to evaluation
        out = self(kwargs['input'])
        return F.cross_entropy(out, kwargs['target'], size_average=kwargs['average'])
