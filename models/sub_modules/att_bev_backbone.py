import numpy as np
import torch
import torch.nn as nn

from opencood.models.fuse_modules.self_attn import AttFusion
from opencood.models.sub_modules.auto_encoder import AutoEncoder


class AttBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        self.compress = False

        if 'compression' in model_cfg and model_cfg['compression'] > 0:
            self.compress = True
            self.compress_layer = model_cfg['compression']

        if 'layer_nums' in self.model_cfg:

            assert len(self.model_cfg['layer_nums']) == \
                   len(self.model_cfg['layer_strides']) == \
                   len(self.model_cfg['num_filters'])

            layer_nums = self.model_cfg['layer_nums']
            layer_strides = self.model_cfg['layer_strides']
            num_filters = self.model_cfg['num_filters']
        else:
            layer_nums = layer_strides = num_filters = []

        if 'upsample_strides' in self.model_cfg:
            assert len(self.model_cfg['upsample_strides']) \
                   == len(self.model_cfg['num_upsample_filter'])

            num_upsample_filters = self.model_cfg['num_upsample_filter']
            upsample_strides = self.model_cfg['upsample_strides']

        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]

        self.blocks = nn.ModuleList()
        self.fuse_modules = nn.ModuleList()
        self.deblocks = nn.ModuleList()

        if self.compress:
            self.compression_modules = nn.ModuleList()

        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]

            fuse_network = AttFusion(num_filters[idx])
            self.fuse_modules.append(fuse_network)
            if self.compress and self.compress_layer - idx > 0:
                self.compression_modules.append(AutoEncoder(num_filters[idx],
                                                            self.compress_layer-idx))

            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx],
                              kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])

            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx],
                                       eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3,
                                       momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1],
                                   stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict, attack = None, attack_src = None, num = 0, sparsity = False, keep_pos = False, delete_list = []):
        spatial_features = data_dict['spatial_features']
        record_len = data_dict['record_len']

        ups = []
        ret_dict = {}
        x = spatial_features
        
        random_att = False
        zero_mask_list = []
        neg_index_list = []
        neg_part_list = []

        for i in range(len(self.blocks)):

            x = self.blocks[i](x)
            if self.compress and i < len(self.compression_modules):
                x = self.compression_modules[i](x)
            
            # 随机施加噪声
            if random_att:
                if x.shape[0] > 1: 
                    # all agents are attackers
                    zero_part = torch.zeros(1, x.shape[1], x.shape[2], x.shape[3]).cuda()
                    perturb = torch.randn(x.shape[0] - 1, x.shape[1], x.shape[2], x.shape[3]).cuda()
                    perturb = torch.cat([zero_part, perturb], dim=0)
                    x = x + perturb
                    # one agent is an attacker
                    # x[1] = x[1] + torch.rand(x.shape[1], x.shape[2], x.shape[3]).cuda()
                    x_fuse = self.fuse_modules[i](x, record_len, delete_list)
                else:
                    x_fuse = self.fuse_modules[i](x, record_len, delete_list)
            else:
            # PGD攻击
                # 如果攻击智能体数不够就直接跳过(此时attack_src是空)
                if attack is None or len(attack_src) == 0:
                    # print(x.shape)
                    x_fuse = self.fuse_modules[i](x, record_len, delete_list)
                
                # attack是list，为n个att，每个att对应一个list，为每个block
                elif isinstance(attack, list):
                    tmp = x.clone()
                    # np.save(f'/GPFS/data/shengyin/OpenCOOD-main/outcome/feature_step_10/normal/sample_{num}_block_{i}.npy', tmp.detach().cpu().numpy())

                    num_att = len(attack_src)
                    
                    # 每个攻击者对应的x施加扰动
                    for j in range(num_att):
                        perturb = attack[i][j]
                        attacker = attack_src[j]
                        # if sparsity:
                        #     zero_mask = (x[attacker] != 0).float()
                        #     zero_mask_list.append(zero_mask)
                        #     perturb = perturb * zero_mask
                        #     x[attacker] = x[attacker] + perturb
                        # if keep_pos:
                        #     neg_index = x[attacker] < 0
                        #     neg_index_list.append(neg_index)
                        #     neg_part = x[attacker][neg_index].detach()
                        #     neg_part_list.append(neg_part)
                        x[attacker] = x[attacker] + perturb
                    
                    tmp = x.clone()
                    # np.save(f'/GPFS/data/shengyin/OpenCOOD-main/outcome/feature_step_10/perturbed/sample_{num}_block_{i}.npy', tmp.detach().cpu().numpy())

                    x_fuse = self.fuse_modules[i](x, record_len, delete_list)

            # 只对第一个block求residual
            if i == 0:
                residual_vector = x_fuse - x[0]

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x

            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x_fuse))
            else:
                ups.append(x_fuse)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return data_dict, zero_mask_list, [neg_index_list, neg_part_list], residual_vector
