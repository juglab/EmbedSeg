"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import torch.nn as nn
import EmbedSeg.models.erfnet_3d as erfnet


class BranchedERFNet_3d(nn.Module):
    def __init__(self, num_classes, input_channels =1, encoder=None):
        super().__init__()

        print('Creating branched erfnet 3d with {} classes'.format(num_classes))
        if (encoder is None):
            self.encoder = erfnet.Encoder(sum(num_classes), input_channels)
        else:
            self.encoder = encoder

        self.decoders = nn.ModuleList()
        for n in num_classes:
            self.decoders.append(erfnet.Decoder(n))

    def init_output(self, n_sigma=1):
        with torch.no_grad():
            output_conv = self.decoders[0].output_conv
            print('initialize last layer with size: ',
                  output_conv.weight.size())

            output_conv.weight[:, 0:3, :, :, :].fill_(0)
            output_conv.bias[0:3].fill_(0)

            output_conv.weight[:, 3:3 + n_sigma, :, :, :].fill_(0)
            output_conv.bias[3:3 + n_sigma].fill_(1)

    def forward(self, input, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            output = self.encoder(input)

        return torch.cat([decoder.forward(output) for decoder in self.decoders], 1)


