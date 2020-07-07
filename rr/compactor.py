import torch
import torch.nn.init as init
from torch.nn import Conv2d
import numpy as np

class CompactorLayer(torch.nn.Module):

    def __init__(self, num_features, conv_idx, ):
        super(CompactorLayer, self).__init__()
        self.conv_idx = conv_idx
        self.pwc = Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=1,
                          stride=1, padding=0, bias=False)
        identity_mat = np.eye(num_features, dtype=np.float32)
        self.pwc.weight.data.copy_(torch.from_numpy(identity_mat).reshape(num_features, num_features, 1, 1))
        self.register_buffer('mask', torch.ones(num_features))
        init.ones_(self.mask)
        self.num_features = num_features

    def forward(self, inputs):
        return self.pwc(inputs)


    def set_mask(self, zero_indices):
        new_mask_value = np.ones(self.num_features, dtype=np.float32)
        new_mask_value[np.array(zero_indices)] = 0.0
        self.mask.data = torch.from_numpy(new_mask_value).cuda().type(torch.cuda.FloatTensor)

    def set_weight_zero(self, zero_indices):
        new_compactor_value = self.pwc.weight.data.detach().cpu().numpy()
        new_compactor_value[np.array(zero_indices), :, :, :] = 0.0
        self.pwc.weight.data = torch.from_numpy(new_compactor_value).cuda().type(torch.cuda.FloatTensor)

    def get_num_mask_ones(self):
        mask_value = self.mask.cpu().numpy()
        return np.sum(mask_value == 1)

    def get_remain_ratio(self):
        return self.get_num_mask_ones() / self.num_features

    def get_pwc_kernel_detach(self):
        return self.pwc.weight.detach()

    def get_lasso_vector(self):
        lasso_vector = torch.sqrt(torch.sum(self.get_pwc_kernel_detach() ** 2, dim=(1, 2, 3))).cpu().numpy()
        return lasso_vector

    def get_metric_vector(self):
        metric_vector = torch.sqrt(torch.sum(self.get_pwc_kernel_detach() ** 2, dim=(1, 2, 3))).cpu().numpy()
        return metric_vector