import torch
import torch.nn as nn
import torch.nn.functional as F

class feat_decoder(nn.Module):
    def __init__(self, feature_shape_list):
        super(feat_decoder, self).__init__()
        assert len(feature_shape_list) >= 2
        for ele in feature_shape_list:
            assert isinstance(ele, int)
        
        self.mod_list = []

        for i in range(len(feature_shape_list) - 1):
            input_dim = feature_shape_list[i]
            output_dim = feature_shape_list[i + 1]
            self.mod_list.append(nn.utils.weight_norm(nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=1, padding=0)))
            # self.mod_list.append(nn.BatchNorm2d(output_dim))
            if i != len(feature_shape_list) - 2:
                self.mod_list.append(nn.ReLU(inplace=True))
        
        # Initialize conv layers as identity
        for i in range(0, len(self.mod_list), 2):
            nn.init.normal_(self.mod_list[i].weight)
            nn.init.normal_(self.mod_list[i].bias)
        
        self.net = nn.Sequential(*self.mod_list)

    def forward(self, x):
        x = self.net(x)
        return x
