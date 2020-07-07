from rr.resrep_config import ResRepConfig
from builder import ConvBuilder
from base_config import BaseConfigByEpoch
import torch.nn as nn
from rr.compactor import CompactorLayer

class ResRepBuilder(ConvBuilder):

    def __init__(self, base_config:BaseConfigByEpoch, resrep_config:ResRepConfig, mode='train'):
        super(ResRepBuilder, self).__init__(base_config=base_config)
        self.resrep_config = resrep_config
        assert mode in ['train', 'deploy']
        self.mode = mode

    def Conv2dBN(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
               padding_mode='zeros', use_original_conv=False):
        self.cur_conv_idx += 1
        assert type(kernel_size) is int
        in_channels = int(in_channels)
        out_channels = int(out_channels)

        if self.mode == 'deploy':
            se = self.Sequential()
            se.add_module('conv', super(ResRepBuilder, self).Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                                                    padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode,
                                                                    bias=True))
            return se

        if use_original_conv or self.cur_conv_idx not in self.resrep_config.target_layers:
            self.cur_conv_idx -= 1
            print('layer {}, use original conv'.format(self.cur_conv_idx + 1))
            return super(ResRepBuilder, self).Conv2dBN(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                                       padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode)

        else:

            se = self.Sequential()
            conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False,
                                   padding_mode=padding_mode)
            se.add_module('conv', conv_layer)
            bn_layer = self.BatchNorm2d(num_features=out_channels)
            se.add_module('bn', bn_layer)
            se.add_module('compactor', CompactorLayer(num_features=out_channels, conv_idx=self.cur_conv_idx))
            print('use compactor on conv {} with kernel size {}'.format(self.cur_conv_idx, kernel_size))
            return se