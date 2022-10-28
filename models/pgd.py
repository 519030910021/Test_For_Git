# modified from from torchattacks
from asyncio import FastChildWatcher
from codecs import decode
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.ops import diff_iou_rotated_3d

from torchattacks.attack import Attack

from opencood.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor
import matplotlib.pyplot as plt

import os 
import sys 
import ipdb
# sys.path.append('/DB/data/yanghengzhao/adversarial/Rotated_IoU')
# from oriented_iou_loss import cal_diou, cal_giou, cal_iou
# from .iou_utils import cal_iou

class PGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 0.3)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 40)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=40, random_start=True)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, cls_head, reg_head,
                eps=0.1, alpha=0.1, steps=40, 
                attack_mode='others',
                n_att=1, colla_attack=True, 
                noise_attack=False,
                random_start=True,
                project=True,
                save_path=None,
                save_attack=False):
        super().__init__("PGD", model)
        """
            需要支持的功能：
            1. 设置攻击的超参数(eps, alpha, steps, project) -- done
            2. 设置attacker的src和tgt -- done
            3. 设置attacker个数 -- to do
            4. targeted 攻击？ -- to do
            5. agent 之间的迁移 -- done 
            6. random smooth -- to do
            7. multi-agent attack 是否合作 -- to do
        """
        self.eps = eps
        self.alpha = alpha  
        self.gamma = 1
        self.steps = steps
        self.random_start = random_start
        self.attack_mode = attack_mode
        self.noise_attack = noise_attack
        self.project = project

        # 最终得到cls和reg
        self.cls_head = cls_head
        self.reg_head = reg_head

        self.n_att = n_att
        self.colla_attack = colla_attack


    def forward(self, data_dict, anchors, reg_targets, labels, num = 0, sparsity = False, keep_pos = False, attack_target = 'pred'):
        r"""
        Overridden.
        """

        # # feature
        # data_dict = data_dict.clone().detach().to(self.device)

        # 4个参与loss计算的变量
        anchors = anchors.clone().detach().to(self.device)
        reg_targets = reg_targets.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        attack_srcs = self.get_attack_src(data_dict['spatial_features'].shape[0])

        # 如果攻击者数大于可传递信息的智能体数，直接返回空的attack
        if data_dict['spatial_features'].shape[0] - 1 < self.n_att:
            return [torch.zeros(64, 100, 352).cuda(), torch.zeros(128, 50, 176).cuda(), torch.zeros(256, 25, 88).cuda()], attack_srcs

        
        # 是否联合攻击
        if self.colla_attack:
            # 让attacks变成拥有 3 * n_att个元素
            attack = [torch.Tensor(self.n_att, 64, 100, 352).cuda(), torch.Tensor(self.n_att, 128, 50, 176).cuda(), torch.Tensor(self.n_att, 256, 25, 88).cuda()]

            if self.random_start:
                    # Starting at a uniformly random point
                    if not isinstance(self.eps, float):
                        for j in range(3):
                            attack[j].uniform_(-self.eps[j], self.eps[j])
                    else:
                        for a in attack:
                            a.uniform_(-self.eps, self.eps)

            if self.noise_attack: 
                    for _ in range(self.steps):
                        for a in attack:
                            a += self.alpha * torch.randn_like(a) * 3
                        if self.project:
                            if not isinstance(self.eps, float):
                                for j in range(3):
                                    attack[j].clamp_(min=-self.eps[j], max=self.eps[j])
                            else:
                                for a in attack:
                                    a.clamp_(min=-self.eps, max=self.eps)
            else:
                loss_list = []
                for _ in range(self.steps):
                    
                    # require grad
                    for a in attack:
                        a.requires_grad = True
                    
                    # 输入inner model
                    outputs, zero_mask, neg_part, _ = self.model(data_dict, attack = attack, attack_src = attack_srcs, sparsity = sparsity, keep_pos = keep_pos)

                    cost = self.loss(outputs, anchors, reg_targets, labels, attack_target)
                    
                    # 存储cost测试一下
                    tmp = torch.clone(cost).detach().cpu().numpy()
                    loss_list.append(tmp)

                    # Update adversarial images
                    grad_list = torch.autograd.grad(cost, attack,
                                                retain_graph=False, create_graph=False)
                    grad_list = list(grad_list)

                    # FGSN的计算公式
                    for k in range(len(attack)):
                        a = attack[k]
                        a = a.detach() - self.alpha * grad_list[k].sign()
                        attack[k] = a
                    if self.project:
                        if not isinstance(self.eps, float):
                            for j in range(3):
                                attack[j].clamp_(min=-self.eps[j], max=self.eps[j])
                        else:
                                for a in attack:
                                    a.clamp_(min=-self.eps, max=self.eps)

            np.save(f'/GPFS/data/shengyin/OpenCOOD-main/attack_loss_1/loss_sample_{num}.npy',loss_list)
            attacks = attack
        else:
            attacks = []
            for attack_src in attack_srcs:

                attack = [torch.Tensor(1, 64, 100, 352).cuda(), torch.Tensor(1, 128, 50, 176).cuda(), torch.Tensor(1,256, 25, 88).cuda()]

                if self.random_start:
                    # Starting at a uniformly random point
                    if not isinstance(self.eps, float):
                        for i in range(len(attack)):
                            attack[i].uniform_(-self.eps[i], self.eps[i])
                    else:
                        for a in attack:
                            a.uniform_(-self.eps, self.eps)
                if self.noise_attack: 
                    for _ in range(self.steps):
                        for a in attack:
                            a += self.alpha * torch.randn_like(a) * 3
                        if self.project:
                            if not isinstance(self.eps, float):
                                for i in range(len(attack)):
                                    attack[i].clamp_(min=-self.eps[i], max=self.eps[i])
                            else:
                                for a in attack:
                                    a.clamp_(min=-self.eps, max=self.eps)
                else:
                    loss_list = []
                    for p in range(self.steps):
                        
                        # require grad
                        for a in attack:
                            a.requires_grad = True
                        
                        # 输入inner model
                        outputs, zero_mask, neg_part, _ = self.model(data_dict, attack = attack, attack_src = [attack_src], sparsity = sparsity, keep_pos = keep_pos)

                        # TODO: 关于loss函数的适配
                        cost, true_num = self.loss(outputs, anchors, reg_targets, labels, attack_target)

                        # if p == 0:
                        #     np.save(f'/GPFS/data/shengyin/OpenCOOD-main/true_num/'+ attack_target +f'/sample_{num}', true_num.detach().cpu().numpy())
                        
                        if cost.isnan():
                            attack = [torch.zeros(64, 100, 352).cuda(), torch.zeros(128, 50, 176).cuda(), torch.zeros(256, 25, 88).cuda()]
                            break
                        # Update adversarial images
                        grad = torch.autograd.grad(cost, attack,
                                                retain_graph=False, create_graph=False)# allow_unused=True)
                        grad = list(grad)
                        if sparsity:
                            for i in range(len(zero_mask)):
                                grad[i] = grad[i] * zero_mask[i]

                        # grad = [torch.autograd.grad(cost, a,
                        #                         retain_graph=True, create_graph=False)[0] for a in attack] 
                        # 另外一种求梯度的方式
                        # cost.backward()     
                        # torch.zero_grad()
                        # # attack[0].grad

                        # FGSN的计算公式

                        tmp_loss = torch.clone(cost)
                        loss_list.append(tmp_loss.detach().cpu().numpy())

                        for k in range(len(attack)):
                            a = attack[k]
                            a = a.detach() - self.alpha * grad[k].sign()
                            attack[k] = a
                        if self.project:
                            if not isinstance(self.eps, float):
                                for i in range(len(attack)):
                                    attack[i].clamp_(min=-self.eps[i], max=self.eps[i])
                            else:
                                for a in attack:
                                    a.clamp_(min=-self.eps, max=self.eps)

                        if keep_pos:
                            neg_index = neg_part[0]
                            neg_feat = neg_part[1]
                            for i in range(len(neg_index)):
                                attack[i][neg_index[i]] -= neg_feat[i] - 1e-7
                    
                    
                    # np.save(f'/GPFS/data/shengyin/OpenCOOD-main/attack_loss/loss_sample_{num}.npy',loss_list)
                attacks.extend(attack)
            # attacks = torch.cat(attacks, dim=0)
            real_attacks = [torch.Tensor(self.n_att, 64, 100, 352).cuda(), torch.Tensor(self.n_att, 128, 50, 176).cuda(), torch.Tensor(self.n_att, 256, 25, 88).cuda()]
            for block in range(3):
                for att in range(self.n_att):
                    real_attacks[block][att] = attacks[3*att + block]
            attacks = real_attacks
        return attacks, attack_srcs

    def get_attack_src(self, agent_num):

        if agent_num - 1 < self.n_att:
            return []

        attacker = torch.randint(low=1,high=agent_num,size=(self.n_att,))
        tmp = []
        for i in range(len(attacker)):
            tmp.append(attacker[i])

        return tmp
    
    def inference(self, data_dict, attack, attack_src, delete_list = [], num = 0):        
        outputs, _, _, residual_vector = self.model(data_dict, attack = attack, attack_src = attack_src, num = num, delete_list = delete_list)
        return outputs, residual_vector

    def loss_det(self,
            result, # backbone的输出
            anchors, # (1, H, W, 2, 7)
            reg_targets, # (1, H, W, 14) 
            labels, # (1, H, W, 2))
            ):
        spatial_features_2d = result['spatial_features_2d']
        pred_cls = self.cls_head(spatial_features_2d)                                  # (1, 2, H, W)
        pred_loc = self.reg_head(spatial_features_2d)                                  # (1, 14, H, W)
        pred_loc = pred_loc.permute(0, 2, 3, 1).contiguous()                           # (1, H, W, 14)
        return (torch.sum(torch.abs(pred_loc - reg_targets)))

    def loss_(self,
            result, # backbone的输出
            anchors, # (1, H, W, 2, 7)
            reg_targets, # (1, H, W, 14) 
            labels, # (1, H, W, 2) 
            ):
        
        # 生成两个结果
        spatial_features_2d = result                  # ['spatial_features_2d']
        pred_cls = torch.randn(1, 2, 100, 352).cuda()
        pred_loc = torch.randn(1, 14, 100, 352).cuda()
        # pred_cls = self.cls_head(spatial_features_2d) # (1, 2, H, W)
        # pred_loc = self.reg_head(spatial_features_2d) # (1, 14, H, W)

        # 调整shape
        pred_cls = pred_cls.permute(0, 2, 3, 1).contiguous().squeeze(0)                # (H, W, 2)
        pred_cls = pred_cls.view(-1, )  
        pred_cls = torch.sigmoid(pred_cls)                                                # (2*H*W, 1)
        pred_loc = pred_loc.permute(0, 2, 3, 1).contiguous()                           # (1, H, W, 14)
        anchors = anchors.squeeze(0)                                                   # (H, W, 2, 7)
        
        # 先生成真实的pred，再选出foreground
        '''
        delta_to_boxes3d
        inputs: rm (1, H, W, 14), anchor (H, W, 2, 7)
        outputs: (1, H*W*2, 7)
        '''
        
        decoded_pred = VoxelPostprocessor.delta_to_boxes3d(pred_loc, anchors).squeeze(0)
        decoded_target = VoxelPostprocessor.delta_to_boxes3d(reg_targets, anchors).squeeze(0)

        labels = labels.squeeze(0).view(-1)    # (H*W, )
        fg_proposal = labels == 1              # (H*W, )
        bg_proposal = labels == 0              # (H*W, )
        pred = decoded_pred[fg_proposal].unsqueeze(0)       # (1, N, 7)
        target = decoded_target[fg_proposal].unsqueeze(0)   # (1, N, 7)

        # compute IoU
        #这里要将h和l的位置互换一下
        pred[[0],[2],[4]] = pred[[0],[2],[5]]
        target[[0],[2],[4]] = target[[0],[2],[5]]
        # input (1, N, 7), 其中7个特性为(x,y,z,w,h,l,alpha)
        # output (1, N)
        #import ipdb; ipdb.set_trace()
        iou = diff_iou_rotated_3d(pred, target)[0].squeeze(0)

        # loss 
        lamb = 0.2
        # lamb = 1.0
        total_loss = torch.sum(- torch.log(1 - pred_cls[fg_proposal]) * iou) + lamb * torch.sum(- pred_cls[bg_proposal].pow(self.gamma) * torch.log(1 - pred_cls[bg_proposal]))
        return total_loss

    def loss(self,
            result, # backbone的输出
            anchors, # (1, H, W, 2, 7)
            reg_targets, # (B, H, W, 14) 
            labels, # (B, H, W, 2)
            attack_target # pred / gt 
            ):
        
        # 生成两个结果
        spatial_features_2d = result['spatial_features_2d']
        pred_cls = self.cls_head(spatial_features_2d) # (B, 2, H, W)
        pred_loc = self.reg_head(spatial_features_2d) # (B, 14, H, W)

        # 调整shape
        pred_cls = pred_cls.permute(0, 2, 3, 1).contiguous().squeeze(0)                # (H, W, 2)
        pred_cls = pred_cls.view(-1, )  
        pred_cls = torch.sigmoid(pred_cls)                                             # (2*H*W, 1)
        pred_loc = pred_loc.permute(0, 2, 3, 1).contiguous()                           # (1, H, W, 14)
        anchors = anchors.squeeze(0)                                                   # (H, W, 2, 7)
        
        # 先生成真实的pred，再选出foreground
        '''
        delta_to_boxes3d
        inputs: rm (B, H, W, 14), anchor (1, H, W, 2, 7)
        outputs: (B, H*W*2, 7)
        '''
        decoded_pred = VoxelPostprocessor.delta_to_boxes3d(pred_loc, anchors).view(-1, 7)
        decoded_target = VoxelPostprocessor.delta_to_boxes3d(reg_targets, anchors).view(-1, 7)

        # gt/pred不同之处在于label的处理
        if attack_target == 'gt':
            labels = labels.squeeze(0).view(-1)    # (H*W, )
            fg_proposal = labels == 1              # (H*W, )
            bg_proposal = labels == 0              # (H*W, )
        else:
            labels = labels.squeeze(0).view(-1, )
            labels = torch.sigmoid(labels)
            fg_proposal = labels > 0.7             # (H*W, )
            bg_proposal = labels <= 0.7            # (H*W, )  
    

        pred = decoded_pred[fg_proposal].unsqueeze(0)       # (1, N, 7)
        target = decoded_target[fg_proposal].unsqueeze(0)   # (1, N, 7)

        # compute IoU
        #这里要将h和l的位置互换一下
        pred[:,:,[4,5]] = pred[:,:,[5,4]]
        target[:,:,[4,5]] = target[:,:,[5,4]]
        # input (1, N, 7), 其中7个特性为(x,y,z,w,h,l,alpha)
        # output (1, N)
        if fg_proposal.sum() == 0:
            iou = 0
        else:
            iou = diff_iou_rotated_3d(pred, target.float())[0].squeeze(0)

        # loss 
        lamb = 0.2
        # lamb = 1.0
        total_loss = torch.sum(- torch.log(1 - pred_cls[fg_proposal] + 1e-6) * iou) + lamb * torch.sum(- pred_cls[bg_proposal].pow(self.gamma) * torch.log(1 - pred_cls[bg_proposal] + 1e-6))
        return total_loss, fg_proposal.sum()


if __name__ == "__main__":
    pgd = PGD(torch.nn.Conv2d(1,1,1,1).cuda(),
                torch.nn.Conv2d(1,1,1,1).cuda(),
                torch.nn.Conv2d(1,1,1,1).cuda(), 
                eps=0.1, alpha=0.1, steps=40, 
                attack_mode='others',
                n_att=1, colla_attack=True, 
                noise_attack=False,
                random_start=True,
                project=True,
                save_path=None)

    result = torch.Tensor(np.load('/GPFS/data/shengyin/OpenCOOD-main/result.npy')).cuda()
    reg_target = torch.Tensor(np.load('/GPFS/data/shengyin/OpenCOOD-main/reg_target.npy')).cuda()
    labels = torch.Tensor(np.load('/GPFS/data/shengyin/OpenCOOD-main/labels.npy')).cuda()
    anchors = torch.Tensor(np.load('/GPFS/data/shengyin/OpenCOOD-main/anchors.npy')).cuda()
    
    print(pgd.loss_(result=result, reg_targets=reg_target, labels=labels, anchors=anchors))
    
