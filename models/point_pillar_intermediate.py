# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib
# Modified by: Sheng Yin

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.att_bev_backbone import AttBEVBackbone
from opencood.utils.match import HungarianMatcher
from opencood.utils.residual_autoencoder import ResidualAutoEncoderV2ReconstructionLoss
from opencood.utils.bh_procedure import build_bh_procedure
from .pgd import PGD




class PointPillarIntermediate(nn.Module):
    def __init__(self, args):

        # 这里的args是数据
        super(PointPillarIntermediate, self).__init__()

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        # 相对于Fafmodule，这里称为内层model
        self.backbone = AttBEVBackbone(args['base_bev_backbone'], 64)
        
        # 两个（已经训练好）的卷积神经网络用于获得最终的结果
        self.cls_head = nn.Conv2d(128 * 3, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 3, 7 * args['anchor_num'],
                                  kernel_size=1)

    def attack_detection(self, data_dict, dataset, method, attack = None):
        if method == 'match':
            return self.attack_detection_match(data_dict, dataset, attack)
        elif method == 'autoencoder':
            return self.attack_detection_ae(data_dict, dataset, attack)
        elif method == 'multi_test':
            return self.attack_detection_multi_test(data_dict, dataset, attack)
        else:
            print("Please choose an available method among match,autoencoder and multi_test!")
            exit(0)

    def upperbound_generation(self, data_dict, dataset, attack = None, method = 'match_cost'):
        """
        inputs:
            data_dict: origin data which includes other agents' information
            attack: (Dict) have 4 parts -- attack_src, block 1, block 2, block 3
            method: detection method like "match_cost","raw autoencoder" and so on
            dataset: used for the data post_process
        """

        cav_content = data_dict['ego']
        num_agent = cav_content['record_len']
        output_dict = OrderedDict()
        # TODO: 需要测量实际的阈值
        percentiles = {
            50: 0.34508609771728516,
            60: 0.3668681025505066,
            70: 0.39437257647514345,
            80: 0.43134005069732667,
            90: 0.5459123015403747,
            95: 0.7015174746513366}

        # ego-data process
        delete_list = [i for i in range(1, num_agent)]
        ego_output = self.forward(data_dict=cav_content, delete_list=delete_list)
        output_dict['ego'] = ego_output
        with torch.no_grad():
            target_box_tensor, target_score, gt_box_tensor, target_bbox = \
                dataset.post_process(data_dict, output_dict)
        target_box = {'box_tensor':target_bbox, 'score':target_score}

        # 生成 batch_dict
        voxel_features = cav_content['processed_lidar']['voxel_features']
        voxel_coords = cav_content['processed_lidar']['voxel_coords']
        voxel_num_points = cav_content['processed_lidar']['voxel_num_points']
        record_len = cav_content['record_len']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        
        # fuse-data process
        # 如果agent = 1直接返回ego的结果即可
        if num_agent == 1:
            return target_box_tensor, target_score, gt_box_tensor, target_bbox
        else:
            attack = attack.item()
            matcher = HungarianMatcher()
            attack_model = PGD(self.backbone, self.cls_head, self.reg_head)
            attack_src = attack['src']
            if attack_src == []:
                actual_attack = None
            else:
                actual_attack = [torch.tensor(attack[f'block{i}']).cuda() for i in range(1,4)]
            final_delete_list = []
            for agent in range(1, num_agent):
                if agent in attack_src:
                    final_delete_list.append(agent)
            # for agent in range(1, num_agent):
            #     # 生成delete list
            #     delete_list = []
            #     for j in range(1, num_agent):
            #         if j != agent:
            #             delete_list.append(j)

            #     # 得到单独与agent合作的结果
            #     attack_result,_ = attack_model.inference(batch_dict, actual_attack, attack_src,delete_list)
            #     spatial_features_2d = attack_result['spatial_features_2d']
            #     psm = self.cls_head(spatial_features_2d)
            #     rm = self.reg_head(spatial_features_2d)
            #     att_output = {'psm': psm, 'rm': rm}
            #     output_dict['ego'] = att_output
            #     # TODO: 此处的pred_tensor可能为空
            #     with torch.no_grad():
            #         pred_tensor, pred_score, gt_box_tensor, pred_bbox = \
            #             dataset.post_process(data_dict, output_dict)
            #     pred_box = {'box_tensor':pred_bbox, 'score':pred_score}
            #     match_loss, _, _ = matcher(pred_box, target_box)

            #     if match_loss[0] > percentiles[95]:
            #         final_delete_list.append(agent) 

            # 得到最终的结果并返回
            final_output = self.forward(data_dict=cav_content, delete_list=final_delete_list)
            output_dict['ego'] = final_output
            with torch.no_grad():
                box_tensor, score, gt_box_tensor, bbox = \
                    dataset.post_process(data_dict, output_dict)
            return box_tensor, score, gt_box_tensor, bbox
        
    def generate_ae_loss(self, data_dict, attack = None):
        """
        inputs:
            data_dict: origin data which includes other agents' information
            attack: (Dict) have 4 parts -- attack_src, block 1, block 2, block 3
        """

        cav_content = data_dict['ego']
        num_agent = cav_content['record_len']
        # 生成 batch_dict
        voxel_features = cav_content['processed_lidar']['voxel_features']
        voxel_coords = cav_content['processed_lidar']['voxel_coords']
        voxel_num_points = cav_content['processed_lidar']['voxel_num_points']
        record_len = cav_content['record_len']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)

        # fuse-data process
        # 如果agent数为1,直接返回空列表
        if num_agent == 1:
            return []
        else:
            # attack = attack.item()
            ae_loss_list = []
            loss_fn = ResidualAutoEncoderV2ReconstructionLoss().cuda()
            attack_model = PGD(self.backbone, self.cls_head, self.reg_head)
            attack_src = attack['src']
            if attack_src == []:
                actual_attack = None
            else:
                actual_attack = [torch.tensor(attack[f'block{i}']).cuda() for i in range(1,4)]
            for agent in range(1, num_agent):
                # 生成delete list
                delete_list = []
                for j in range(1, num_agent):
                    if j != agent:
                        delete_list.append(j)

                # 得到单独与agent合作的结果
                attack_result, residual_vector = attack_model.inference(batch_dict, actual_attack, attack_src,delete_list)
                tmp_loss = loss_fn(residual_vector)
                tmp_loss = tmp_loss.detach().cpu().numpy()
                if agent not in attack_src:
                    ae_loss_list.append([tmp_loss, 0])
                else:
                    ae_loss_list.append([tmp_loss, 1])

            return ae_loss_list

    def attack_detection_ae(self, data_dict, dataset, attack = None):
        """
        inputs:
            data_dict: origin data which includes other agents' information
            attack: (Dict) have 4 parts -- attack_src, block 1, block 2, block 3
        """

        output_dict = OrderedDict()
        cav_content = data_dict['ego']
        num_agent = cav_content['record_len']
        # 生成 batch_dict
        voxel_features = cav_content['processed_lidar']['voxel_features']
        voxel_coords = cav_content['processed_lidar']['voxel_coords']
        voxel_num_points = cav_content['processed_lidar']['voxel_num_points']
        record_len = cav_content['record_len']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)

        percentiles = {
            50: 16071.2265625,
            60: 17611.601953125,
            70: 19970.0875,
            80: 22846.326953125,
            90: 26177.4796875,
            95: 27980.115624999988}
        
#         percentiles_pred = {
#             16071.2265625
# 17611.60234375
# 19970.08359375
# 22846.32890625
# 26177.4796875
# 27980.11445312499
#         }

        # fuse-data process
        # 如果agent数为1,直接返回空列表
        if num_agent == 1:
            ego_output = self.forward(data_dict=cav_content)
            output_dict['ego'] = ego_output
            with torch.no_grad():
                box_tensor, score, gt_box_tensor, bbox = \
                    dataset.post_process(data_dict, output_dict)
            return box_tensor, score, gt_box_tensor, bbox
        else:
            attack = attack.item()
            loss_fn = ResidualAutoEncoderV2ReconstructionLoss().cuda()
            attack_model = PGD(self.backbone, self.cls_head, self.reg_head)
            attack_src = attack['src']
            if attack_src == []:
                actual_attack = None
            else:
                actual_attack = [torch.tensor(attack[f'block{i}']).cuda() for i in range(1,4)]
            
            final_delete_list = []
            for agent in range(1, num_agent):
                # 生成delete list
                delete_list = []
                for j in range(1, num_agent):
                    if j != agent:
                        delete_list.append(j)

                # 得到单独与agent合作的结果
                attack_result, residual_vector = attack_model.inference(batch_dict, actual_attack, attack_src,delete_list)
                tmp_loss = loss_fn(residual_vector)
                tmp_loss = tmp_loss.detach().cpu().numpy()
                if tmp_loss > percentiles[95]:
                    final_delete_list.append(agent)

            # 得到最终的结果并返回
            final_output = self.forward(data_dict=cav_content, delete_list=final_delete_list)
            output_dict['ego'] = final_output
            with torch.no_grad():
                box_tensor, score, gt_box_tensor, bbox = \
                    dataset.post_process(data_dict, output_dict)
            return box_tensor, score, gt_box_tensor, bbox

    def generate_match_loss(self, data_dict, dataset, attack = None):
        """
        inputs:
            data_dict: origin data which includes other agents' information
            attack: (Dict) have 4 parts -- attack_src, block 1, block 2, block 3
            method: detection method like "match_cost","raw autoencoder" and so on
            dataset: used for the data post_process
        """

        cav_content = data_dict['ego']
        num_agent = cav_content['record_len']
        output_dict = OrderedDict()

        # ego-data process
        delete_list = [i for i in range(1, num_agent)]
        ego_output = self.forward(data_dict=cav_content, delete_list=delete_list)
        output_dict['ego'] = ego_output
        with torch.no_grad():
            target_box_tensor, target_score, gt_box_tensor, target_bbox = \
                dataset.post_process(data_dict, output_dict)
        target_box = {'box_tensor':target_bbox, 'score':target_score}

        # 生成 batch_dict
        voxel_features = cav_content['processed_lidar']['voxel_features']
        voxel_coords = cav_content['processed_lidar']['voxel_coords']
        voxel_num_points = cav_content['processed_lidar']['voxel_num_points']
        record_len = cav_content['record_len']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)

        
        # fuse-data process
        # 如果agent数为1,直接返回空列表
        if num_agent == 1:
            return []
        else:
            # attack = attack.item()
            match_loss_list = []
            matcher = HungarianMatcher()
            attack_model = PGD(self.backbone, self.cls_head, self.reg_head)
            attack_src = attack['src']
            if attack_src == []:
                actual_attack = None
            else:
                actual_attack = [torch.tensor(attack[f'block{i}']).cuda() for i in range(1,4)]
            for agent in range(1, num_agent):
                # 生成delete list
                delete_list = []
                for j in range(1, num_agent):
                    if j != agent:
                        delete_list.append(j)

                # 得到单独与agent合作的结果
                attack_result,_ = attack_model.inference(batch_dict, actual_attack, attack_src,delete_list)
                spatial_features_2d = attack_result['spatial_features_2d']
                psm = self.cls_head(spatial_features_2d)
                rm = self.reg_head(spatial_features_2d)
                att_output = {'psm': psm, 'rm': rm}
                output_dict['ego'] = att_output
                # TODO: 此处的pred_tensor可能为空
                with torch.no_grad():
                    pred_tensor, pred_score, gt_box_tensor, pred_bbox = \
                        dataset.post_process(data_dict, output_dict)
                pred_box = {'box_tensor':pred_bbox, 'score':pred_score}
                match_loss, _, _ = matcher(pred_box, target_box)
                if agent not in attack_src:
                    match_loss_list.append([match_loss, 0])
                else:
                    match_loss_list.append([match_loss, 1])

            return match_loss_list

    def attack_detection_multi_test(self, data_dict, dataset, attack = None):

        # 初始化
        dists = []
        dists.append(np.load("/GPFS/data/shengyin/OpenCOOD-main/generate_match/new_validation/validation_match_cost.npy"))
        dists.append(np.load('/GPFS/data/shengyin/OpenCOOD-main/generate_ae_loss/validation/validation_ae_loss.npy'))
        bh_test = build_bh_procedure(dists=dists)


        cav_content = data_dict['ego']
        num_agent = cav_content['record_len']
        output_dict = OrderedDict()
        # ego-data for match cost
        delete_list = [i for i in range(1, num_agent)]
        ego_output = self.forward(data_dict=cav_content, delete_list=delete_list)
        output_dict['ego'] = ego_output
        with torch.no_grad():
            target_box_tensor, target_score, gt_box_tensor, target_bbox = \
                dataset.post_process(data_dict, output_dict)
        target_box = {'box_tensor':target_bbox, 'score':target_score}

        # 生成 batch_dict
        voxel_features = cav_content['processed_lidar']['voxel_features']
        voxel_coords = cav_content['processed_lidar']['voxel_coords']
        voxel_num_points = cav_content['processed_lidar']['voxel_num_points']
        record_len = cav_content['record_len']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        
        # fuse-data process
        # 如果agent = 1直接返回ego的结果即可
        if num_agent == 1:
            return target_box_tensor, target_score, gt_box_tensor, target_bbox
        # 否则进行multi-test
        else:
            attack = attack.item()

            matcher = HungarianMatcher()
            loss_fn = ResidualAutoEncoderV2ReconstructionLoss().cuda()

            attack_model = PGD(self.backbone, self.cls_head, self.reg_head)
            attack_src = attack['src']
            if attack_src == []:
                actual_attack = None
            else:
                actual_attack = [torch.tensor(attack[f'block{i}']).cuda() for i in range(1,4)]
            final_delete_list = []
            for agent in range(1, num_agent):
                # 生成delete list
                delete_list = []
                for j in range(1, num_agent):
                    if j != agent:
                        delete_list.append(j)

                # 得到单独与agent合作的结果
                attack_result, residual_vector = attack_model.inference(batch_dict, actual_attack, attack_src,delete_list)
                spatial_features_2d = attack_result['spatial_features_2d']
                psm = self.cls_head(spatial_features_2d)
                rm = self.reg_head(spatial_features_2d)
                att_output = {'psm': psm, 'rm': rm}
                output_dict['ego'] = att_output
                
                # 生成 match loss
                with torch.no_grad():
                    pred_tensor, pred_score, gt_box_tensor, pred_bbox = \
                        dataset.post_process(data_dict, output_dict)
                pred_box = {'box_tensor':pred_bbox, 'score':pred_score}
                match_loss, _, _ = matcher(pred_box, target_box)

                # 生成 ae_loss
                ae_loss = loss_fn(residual_vector)

                # 生成结果
                scores = [match_loss[0].detach().cpu().numpy(),ae_loss.detach().cpu().numpy()]
                rejected = bh_test.test(scores)
                if len(rejected) > 0:
                    final_delete_list.append(agent)

            # 得到最终的结果并返回
            final_output = self.forward(data_dict=cav_content, delete_list=final_delete_list)
            output_dict['ego'] = final_output
            with torch.no_grad():
                box_tensor, score, gt_box_tensor, bbox = \
                    dataset.post_process(data_dict, output_dict)
            return box_tensor, score, gt_box_tensor, bbox

    def attack_detection_match(self, data_dict, dataset, attack = None, method = 'match_cost', if_gt = False):
        """
        inputs:
            data_dict: origin data which includes other agents' information
            attack: (Dict) have 4 parts -- attack_src, block 1, block 2, block 3
            method: detection method like "match_cost","raw autoencoder" and so on
            dataset: used for the data post_process
        """

        cav_content = data_dict['ego']
        num_agent = cav_content['record_len']
        output_dict = OrderedDict()
        percentiles = {
            50: 0.34508609771728516,
            60: 0.3668681025505066,
            70: 0.39437257647514345,
            80: 0.43134005069732667,
            90: 0.5459123015403747,
            95: 0.7015174746513366}

        # ego-data process
        delete_list = [i for i in range(1, num_agent)]
        ego_output = self.forward(data_dict=cav_content, delete_list=delete_list)
        output_dict['ego'] = ego_output
        with torch.no_grad():
            target_box_tensor, target_score, gt_box_tensor, target_bbox = \
                dataset.post_process(data_dict, output_dict)
        target_box = {'box_tensor':target_bbox, 'score':target_score}

        # 生成 batch_dict
        voxel_features = cav_content['processed_lidar']['voxel_features']
        voxel_coords = cav_content['processed_lidar']['voxel_coords']
        voxel_num_points = cav_content['processed_lidar']['voxel_num_points']
        record_len = cav_content['record_len']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        
        # fuse-data process
        # 如果agent = 1直接返回ego的结果即可
        if num_agent == 1:
            return target_box_tensor, target_score, gt_box_tensor, target_bbox
        else:
            attack = attack.item()
            matcher = HungarianMatcher()
            attack_model = PGD(self.backbone, self.cls_head, self.reg_head)
            attack_src = attack['src']
            if attack_src == []:
                actual_attack = None
            else:
                actual_attack = [torch.tensor(attack[f'block{i}']).cuda() for i in range(1,4)]
            final_delete_list = []
            for agent in range(1, num_agent):
                # 生成delete list
                delete_list = []
                for j in range(1, num_agent):
                    if j != agent:
                        delete_list.append(j)

                # 得到单独与agent合作的结果
                attack_result,_ = attack_model.inference(batch_dict, actual_attack, attack_src,delete_list)
                spatial_features_2d = attack_result['spatial_features_2d']
                psm = self.cls_head(spatial_features_2d)
                rm = self.reg_head(spatial_features_2d)
                att_output = {'psm': psm, 'rm': rm}
                output_dict['ego'] = att_output
                # TODO: 此处的pred_tensor可能为空
                with torch.no_grad():
                    pred_tensor, pred_score, gt_box_tensor, pred_bbox = \
                        dataset.post_process(data_dict, output_dict)
                pred_box = {'box_tensor':pred_bbox, 'score':pred_score}
                match_loss, _, _ = matcher(pred_box, target_box)

                if match_loss[0] > percentiles[95]:
                    if (not if_gt) or (agent in attack_src):
                        final_delete_list.append(agent) 
                

            # 得到最终的结果并返回
            final_output = self.forward(data_dict=cav_content, delete_list=final_delete_list)
            output_dict['ego'] = final_output
            with torch.no_grad():
                box_tensor, score, gt_box_tensor, bbox = \
                    dataset.post_process(data_dict, output_dict)
            return box_tensor, score, gt_box_tensor, bbox

    def forward(self, data_dict, attack=False, com=True, attack_mode='self', eps=0.1, alpha=0.1,
                 proj=True, attack_target='pred', save_path=None, step=15, noise_attack=False,num = 0,sparsity = False, keep_pos = False, save_attack = False, delete_list = []):

        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)

        # 没有attack的结果
        result, _, _, _ = self.backbone(batch_dict,delete_list = delete_list)
        spatial_features_2d = result['spatial_features_2d']
        psm = self.cls_head(spatial_features_2d)
        rm = self.reg_head(spatial_features_2d)
        no_att_output_dict = {'psm': psm,
                       'rm': rm}
        
        # 定义attack模型PGD
        if isinstance(attack, bool) or attack == "TRUE":
            self.attack = attack or attack == "TRUE"
            if self.attack:
                self.attack_model = PGD(self.backbone, self.cls_head, self.reg_head,eps=eps, alpha=alpha, steps=step,
                                        attack_mode=attack_mode, noise_attack=noise_attack, 
                                        n_att=1, colla_attack=True,  # 兼容
                                        project=proj, save_path=save_path)
                self.attack_target = attack_target

        elif isinstance(attack, str):
            # yaml file 
            from omegaconf import OmegaConf
            attack_conf = OmegaConf.load(attack)
            if attack_conf.attack is not None:
                self.attack = True
                self.attack_model = PGD(self.backbone, self.cls_head, self.reg_head, **attack_conf.attack.pgd)
                self.attack_target = attack_conf.attack.attack_target
            else:
                self.attack = False

        
        anchors = data_dict['anchor_box'] # 每个anchor由7个元素表示
        reg_targets = data_dict['label_dict']['targets'] # (1, H, W, 2*7), 2表示每个像素位置有两个anchor  
        labels = data_dict['label_dict']['pos_equal_one'] # (1, H, W, 2)

        # np.save('anchors.npy', anchors.detach().cpu().numpy())
        # np.save('reg_target.npy', reg_targets.detach().cpu().numpy())
        # np.save('labels.npy', labels.detach().cpu().numpy())
        # np.save('result.npy',result['spatial_features_2d'].detach().cpu().numpy())


        # 有attack的结果
        if self.attack:
            if self.attack_model.attack_mode == 'self':
                print("This occasion is not considered now!")
                exit(0)
            else:
                ref_result = no_att_output_dict

            if self.attack_target == 'gt':
                att_reg_targets = reg_targets
                att_labels = labels
            elif self.attack_target == 'pred':
                att_reg_targets = ref_result['rm'].permute(0, 2, 3, 1).contiguous() # (1, 14, 100, 352)
                att_labels = ref_result['psm'].permute(0, 2, 3, 1).contiguous()     # (1, 2, 100, 352)
                
            else:
                raise NotImplementedError(self.attack_target)

            evasion, attack_src = self.attack_model(batch_dict, anchors, att_reg_targets, att_labels, num = num, sparsity = sparsity, keep_pos = keep_pos, attack_target = self.attack_target)
            if save_attack:
                tmp = {'block1':torch.clone(evasion[0]).detach().cpu().numpy(),'block2':torch.clone(evasion[1]).detach().cpu().numpy(),'block3':torch.clone(evasion[2]).detach().cpu().numpy(),'src':attack_src}
                np.save(save_path + f'/sample_{num}.npy', tmp)

            attack_result,_ = self.attack_model.inference(batch_dict, evasion, attack_src=attack_src, num = num,delete_list=delete_list)
            spatial_features_2d = attack_result['spatial_features_2d']
            psm = self.cls_head(spatial_features_2d)
            rm = self.reg_head(spatial_features_2d)
            att_output_dict = {'psm': psm,
                        'rm': rm}
        else:
            att_output_dict = {'psm': None,
                        'rm': None}
        
        if self.attack:
            return att_output_dict
        else:
            return no_att_output_dict