#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from auto_cad_recon.Dataset.retrieval_dataset import RetrievalDataset

from auto_cad_recon.Model.roca import RetrievalNet

from auto_cad_recon.Method.device import toCuda, toCpu, toNumpy
from auto_cad_recon.Method.time import getCurrentTime
from auto_cad_recon.Method.path import createFileFolder, renameFile, removeFile
from auto_cad_recon.Method.io import saveDataList, loadDataList
from auto_cad_recon.Method.render import renderDataList

from auto_cad_recon.Module.dataset_manager import DatasetManager


def _worker_init_fn_(worker_id):
    torch_seed = torch.initial_seed()
    np_seed = torch_seed // 2**32 - 1
    random.seed(torch_seed)
    np.random.seed(np_seed)
    return True


class RetrievalTrainer(object):

    def __init__(self, scannet_dataset_folder_path,
                 scannet_object_dataset_folder_path,
                 scannet_bbox_dataset_folder_path,
                 scan2cad_dataset_folder_path,
                 scan2cad_object_model_map_dataset_folder_path,
                 shapenet_dataset_folder_path,
                 shapenet_udf_dataset_folder_path):
        self.dataset_manager = DatasetManager(
            scannet_dataset_folder_path, scannet_object_dataset_folder_path,
            scannet_bbox_dataset_folder_path, scan2cad_dataset_folder_path,
            scan2cad_object_model_map_dataset_folder_path,
            shapenet_dataset_folder_path, shapenet_udf_dataset_folder_path)

        self.scannet_scene_name = None
        self.scannet_scene_name_list = self.dataset_manager.getScanNetSceneNameList(
        )

        self.retrieval_net = RetrievalNet().cuda()

        self.retrieval_dataset = RetrievalDataset(self.dataset_manager)
        self.retrieval_dataloader = DataLoader(self.retrieval_dataset,
                                               batch_size=64,
                                               shuffle=False,
                                               num_workers=0,
                                               worker_init_fn=_worker_init_fn_)

        self.lr = 1e-3
        self.step = 0
        self.loss_min = float('inf')
        self.log_folder_name = getCurrentTime()
        self.save_result_idx = 0

        self.optimizer = Adam(self.retrieval_net.parameters(), lr=self.lr)
        self.summary_writer = None
        return

    def loadSummaryWriter(self):
        self.summary_writer = SummaryWriter("./logs/" + self.log_folder_name +
                                            "/")
        return True

    def loadModel(self, model_file_path):
        if not os.path.exists(model_file_path):
            self.loadSummaryWriter()
            print("[WARN][RetrievalTrainer::loadModel]")
            print("\t model_file not exist! start training from step 0...")
            return True

        model_dict = torch.load(model_file_path)

        self.retrieval_net.load_state_dict(model_dict['retrieval_net'])
        self.optimizer.load_state_dict(model_dict['optimizer'])
        self.step = model_dict['step']
        self.loss_min = model_dict['loss_min']
        self.log_folder_name = model_dict['log_folder_name']
        self.save_result_idx = model_dict['save_result_idx']

        self.loadSummaryWriter()
        print("[INFO][RetrievalTrainer::loadModel]")
        print("\t load model success! start training from step " +
              str(self.step) + "...")
        return True

    def saveModel(self, save_model_file_path):
        model_dict = {
            'retrieval_net': self.retrieval_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': self.step,
            'loss_min': self.loss_min,
            'log_folder_name': self.log_folder_name,
            'save_result_idx': self.save_result_idx,
        }

        createFileFolder(save_model_file_path)

        tmp_save_model_file_path = save_model_file_path.split(
            ".pth")[0] + "_tmp.pth"

        torch.save(model_dict, tmp_save_model_file_path)

        removeFile(save_model_file_path)
        renameFile(tmp_save_model_file_path, save_model_file_path)
        return True

    def loadScene(self, scannet_scene_name):
        assert scannet_scene_name in self.scannet_scene_name_list

        if self.scannet_scene_name == scannet_scene_name:
            return True

        self.scannet_scene_name = scannet_scene_name
        self.retrieval_dataset.loadScene(self.scannet_scene_name)
        return True

    def loadSceneByIdx(self, scannet_scene_name_idx):
        assert scannet_scene_name_idx <= len(self.scannet_scene_name_list)

        return self.loadScene(
            self.scannet_scene_name_list[scannet_scene_name_idx])

    def testTrain(self, print_progress=False):
        data = self.retrieval_dataset.__getitem__(0, True)
        data['inputs']['dataset_manager'] = self.dataset_manager
        toCuda(data)
        data = self.retrieval_net(data)

        for data in self.retrieval_dataloader:
            data['inputs']['dataset_manager'] = self.dataset_manager
            data['inputs']['scannet_scene_name'] = data['inputs'][
                'scannet_scene_name'][0]
            toCuda(data)

            data = self.retrieval_net(data)
        return True

    def trainStep(self, data):
        data['inputs']['dataset_manager'] = self.dataset_manager
        data['inputs']['scannet_scene_name'] = data['inputs'][
            'scannet_scene_name'][0]
        toCuda(data)

        self.retrieval_net.train()
        self.retrieval_net.zero_grad()
        self.optimizer.zero_grad()

        data = self.retrieval_net(data)

        losses = data['losses']

        losses_tensor = torch.cat([
            loss if len(loss.shape) > 0 else loss.reshape(1)
            for loss in data['losses'].values()
        ])

        loss_sum = torch.sum(losses_tensor)
        loss_sum_float = loss_sum.detach().cpu().numpy()
        self.summary_writer.add_scalar("Loss/loss_sum", loss_sum_float,
                                       self.step)

        if loss_sum_float < self.loss_min:
            self.loss_min = loss_sum_float
            self.saveModel("./output/" + self.log_folder_name +
                           "/model_best.pth")

        for key, loss in losses.items():
            loss_tensor = loss.detach() if len(
                loss.shape) > 0 else loss.detach().reshape(1)
            loss_mean = torch.mean(loss_tensor)
            self.summary_writer.add_scalar("Loss/" + key, loss_mean, self.step)

        loss_sum.backward()
        self.optimizer.step()
        return True

    def trainScene(self,
                   scannet_scene_name,
                   scene_epoch,
                   print_progress=False):
        self.loadScene(scannet_scene_name)

        for_data = range(scene_epoch)
        if print_progress:
            for_data = tqdm(for_data)
        for _ in for_data:
            torch.cuda.empty_cache()
            for data in self.retrieval_dataloader:
                self.trainStep(data)
                self.step += 1

        self.saveModel("./output/" + self.log_folder_name + "/model_last.pth")
        #  self.saveResult(print_progress)
        return True

    def trainEpoch(self,
                   global_epoch_idx,
                   global_epoch,
                   scene_epoch,
                   print_progress=False):
        scannet_scene_name_list = self.dataset_manager.getScanNetSceneNameList(
        )
        for scannet_scene_name_idx, scannet_scene_name in enumerate(
                scannet_scene_name_list):

            # FIXME: for test only
            #  scannet_scene_name = "scene0474_02"

            if print_progress:
                print("[INFO][RetrievalTrainer::trainScene]")
                print("\t start train on scene " + scannet_scene_name +
                      ", epoch: " + str(global_epoch_idx + 1) + "/" +
                      str(global_epoch) + ", scene: " +
                      str(scannet_scene_name_idx + 1) + "/" +
                      str(len(scannet_scene_name_list)) + "...")
            self.trainScene(scannet_scene_name, scene_epoch, print_progress)
        return True

    def train(self, print_progress=False):
        global_epoch = 10000
        scene_epoch = 100

        for global_epoch_idx in range(global_epoch):
            self.trainEpoch(global_epoch_idx, global_epoch, scene_epoch,
                            print_progress)
        return True

    def testScene(self, scannet_scene_name):
        self.retrieval_net.eval()

        self.retrieval_dataset.loadScene(scannet_scene_name)

        data_list = []
        for data_idx in range(len(self.retrieval_dataset)):
            data = self.retrieval_dataset.__getitem__(data_idx, True)
            data['inputs']['dataset_manager'] = self.dataset_manager
            toCuda(data)
            data = self.retrieval_net(data)
            toCpu(data)
            toNumpy(data)
            data_list.append(data)

        renderDataList(data_list)
        return True

    def test(self, print_progress=False):
        scannet_scene_name_list = self.dataset_manager.getScanNetSceneNameList(
        )

        for i, scannet_scene_name in enumerate(scannet_scene_name_list):
            if print_progress:
                print("[INFO][RetrievalTrainer::test]")
                print("\t start test on scene " + scannet_scene_name + ", " +
                      str(i + 1) + "/" + str(len(scannet_scene_name_list)) +
                      "...")
            self.testScene(scannet_scene_name)
        return True

    def saveSceneResult(self,
                        scannet_scene_name,
                        save_json_file_path,
                        print_progress=False):
        self.retrieval_net.eval()

        self.retrieval_dataset.loadScene(scannet_scene_name)

        data_list = []

        for_data = range(len(self.retrieval_dataset))
        if print_progress:
            print("[INFO][RetrievalTrainer::saveSceneResult]")
            print("\t start save scene result on scene " + scannet_scene_name +
                  " as " + str(self.save_result_idx) + ".json"
                  "...")
            for_data = tqdm(for_data)
        for data_idx in for_data:
            data = self.retrieval_dataset.__getitem__(data_idx, True)
            data['inputs']['dataset_manager'] = self.dataset_manager
            toCuda(data)
            data = self.retrieval_net(data)
            toCpu(data)
            toNumpy(data)
            data_list.append(data)

        saveDataList(data_list, save_json_file_path)
        return True

    def saveResult(self, print_progress=False):
        assert self.scannet_scene_name is not None

        save_folder_path = "./output/" + self.log_folder_name + "/result/" + \
            self.scannet_scene_name + "/" + str(self.save_result_idx) + ".json"
        self.saveSceneResult(self.scannet_scene_name, save_folder_path,
                             print_progress)
        self.save_result_idx += 1
        return True
