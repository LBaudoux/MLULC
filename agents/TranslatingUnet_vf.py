"""
Mnist Main agent, as mentioned in the tutorial
"""
import numpy as np
from sklearn.metrics import confusion_matrix
import shutil
import torch
from torch import nn
import torch.optim as optim
import json

from agents.base import BaseAgent

from datasets.landcover_to_landcover import LandcoverToLandcoverDataLoader

from utils.misc import print_cuda_statistics

from os.path import join

from utils.plt_utils import plt_loss2 as plt_loss
from utils.plt_utils import PltPerClassMetrics

from utils.tensorboardx_utils import tensorboard_summary_writer

class TranslatingUnetVfAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda

        # set the manual seed for torch
        self.manual_seed = self.config.seed
        if self.cuda:
            torch.cuda.manual_seed(self.manual_seed)
            self.device = torch.device("cuda")


        # define data_loader
        self.data_loader = LandcoverToLandcoverDataLoader(config=config,device=self.device,pos_enc=True)

        # Get required param for network initialisation
        input_channels = self.data_loader.input_channels
        output_channels = self.data_loader.output_channels
        self.real_patch_sizes = self.data_loader.real_patch_sizes

        if self.real_patch_sizes[0]//self.real_patch_sizes[1]==10: #self.config.datasets==["2.hdf5","cgls.hdf5"] or self.config.datasets==["2.hdf5","1.hdf5"]:
            from graphs.models.translating_unet import TranslatingUnetOSOten as EncDec
            # define models
            self.models = EncDec(input_channels[0], output_channels[1], self.real_patch_sizes[0])
        elif self.real_patch_sizes[0]//self.real_patch_sizes[1]==2: #self.config.datasets==["2.hdf5","3.hdf5"]:
            from graphs.models.translating_unet import TranslatingUnettwo as EncDec
            # define models
            self.models = EncDec(input_channels[0], output_channels[1], self.real_patch_sizes[0])
        elif self.real_patch_sizes[0]//self.real_patch_sizes[1]==1 :#self.config.datasets==["ocsge_o.hdf5","ocsge_u.hdf5"]:
            from graphs.models.translating_unet import TranslatingUnetsame as EncDec
            # define models
            self.models = EncDec(input_channels[0], output_channels[1], self.real_patch_sizes[0])
        elif  self.real_patch_sizes[0]//self.real_patch_sizes[1]==4:#self.config.datasets==["ocsge_o.hdf5","3.hdf5"] or self.config.datasets==["ocsge_u.hdf5","3.hdf5"]:
            from graphs.models.translating_unet import TranslatingUnetfour as EncDec
            # define models
            self.models = EncDec(input_channels[0], output_channels[1], self.real_patch_sizes[0])
        elif self.real_patch_sizes[0]//self.real_patch_sizes[1]==20: #self.config.datasets==["ocsge_u.hdf5","1.hdf5"]:
            from graphs.models.translating_unet import TranslatingUnettwenty as EncDec
            # define models
            self.models = EncDec(input_channels[0], output_channels[1], self.real_patch_sizes[0])
        elif self.real_patch_sizes[0] // self.real_patch_sizes[1] == 5:  # self.config.datasets==["ocsge_u.hdf5","1.hdf5"]:
            from graphs.models.translating_unet import TranslatingUnetfive as EncDec
            # define models
            self.models = EncDec(input_channels[0], output_channels[1], self.real_patch_sizes[0])
        elif self.real_patch_sizes[0] // self.real_patch_sizes[1]<1:
            from graphs.models.translating_unet import TranslatingUnetReduce as EncDec
            # define models
            self.models = EncDec(input_channels[0], output_channels[1], self.real_patch_sizes[0],self.real_patch_sizes[1])


        # define optimizer
        self.optimizers = optim.Adam(self.models.parameters(), lr=self.config.learning_rate[0])

        # # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_metric = 0


        if self.cuda:
            self.models = self.models.to(self.device)

            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)

        # Summary Writer
        if self.config.tensorboard:
            self.summary_writer, self.tensorboard_process = tensorboard_summary_writer(config,
                                                                                       comment=self.config.exp_name)

    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """

        filename = self.config.checkpoint_dir + file_name
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            for i, d in enumerate(self.config.datasets):
                self.models.load_state_dict(checkpoint['encoder_state_dict_' + d])
                self.optimizers.load_state_dict(checkpoint['encoder_optimizer_' + d][0])
                if self.cuda and torch.cuda.device_count() > 1:
                    print("Let's use", torch.cuda.device_count(), "GPUs!")
                    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
                    self.models[i] = torch.nn.DataParallel(self.models[i])
            self.manual_seed = checkpoint['manual_seed']

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                             .format(self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            self.logger.info("**First time to train**")

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0,
                        historical_storage="/work/scratch/baudoulu/train2012_valid2018_modelsave/"):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's accuracy is the best so far
        :return:
        """

        state = {
            'epoch': self.current_epoch,
            'iteration': self.current_iteration,
            'manual_seed': self.manual_seed
        }
        for i, d in enumerate(self.config.datasets):
            state['encoder_optimizer_' + d] = self.optimizers.state_dict(),
            if torch.cuda.device_count() > 1 and self.cuda:
                state['encoder_state_dict_' + d] = self.models.module.state_dict()
            else:
                state['encoder_state_dict_' + d] = self.models.state_dict()

        # Save the state
        torch.save(state, self.config.checkpoint_dir + file_name)
        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(self.config.checkpoint_dir + file_name, self.config.checkpoint_dir + 'model_best.pth.tar')

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            torch.cuda.empty_cache()
            self.train()
            # self.save_patch_loss(self.model,self.device,self.data_loader.train_loader)
            torch.cuda.empty_cache()
            self.test()
            torch.cuda.empty_cache()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training loop
        :return:
        """
        loss_ref = 1000
        plot_training_loss = {d: [] for d in self.config.datasets}
        plot_validation_loss = {d: [] for d in self.config.datasets}

        self.logger.info("Start training !")
        for epoch in range(1, self.config.max_epoch + 1):
            self.logger.info("Training epoch {}/{} ({:.0f}%):\n".format(epoch, self.config.max_epoch,
                                                                        (epoch - 1) / self.config.max_epoch * 100))
            train_loss = self.train_one_epoch()
            # print(train_loss)
            for d, l in train_loss.items():
                plot_training_loss[d].extend(l)
            torch.cuda.empty_cache()
            if epoch % self.config.validate_every == 0:
                self.logger.info("Validation epoch {}/{} ({:.0f}%):\n".format(epoch, self.config.max_epoch,
                                                                              (
                                                                                          epoch - 1) / self.config.max_epoch * 100))
                validation_loss = self.validate()
                for d, l in validation_loss.items():
                    plot_validation_loss[d].append([self.current_iteration, np.mean(l)])
                tmp = [v for v in validation_loss.values()]
                vl = np.mean([item for elem in tmp for item in elem])
                if vl < loss_ref:
                    self.logger.info("Best model for now  : saved ")
                    loss_ref = vl
                    self.save_checkpoint(is_best=1)
                torch.cuda.empty_cache()
            self.current_epoch += 1
            if epoch > 1 and epoch >= 2 * self.config.validate_every:
                plt_loss()(plot_training_loss, plot_validation_loss, savefig=join(self.config.out_dir, "loss.png"))
        self.logger.info("Training ended!")

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        self.logger.info("training one ep")
        plot_loss = {d: [] for d in self.config.datasets}

        self.models.train()
        # compt=0
        batch_idx = 0

        for data in self.data_loader.train_loader[self.config.datasets[0]][self.config.datasets[1]]:
            print(batch_idx)
            # print(compt/self.data_loader.train_iterations)
            # compt+=1
            pos_enc = data.get("coordenc").to(self.device)
            source_patch = data.get("source_one_hot")
            tv = data.get("target_data")[:, 0]

            self.optimizers.zero_grad()
            if self.config.use_pos:
                rec = self.models(source_patch.float(),pos_enc.float())
            else:
                rec = self.models(source_patch.float())
            loss = nn.CrossEntropyLoss(ignore_index=0)(rec, tv)
            loss.backward()
            self.optimizers.step()
            plot_loss[self.config.datasets[0]].append([self.current_iteration, loss.item()])
            plot_loss[self.config.datasets[1]].append([self.current_iteration, loss.item()])

            batch_idx += 1
            self.current_iteration += 1

        if self.config.tensorboard:
            self.summary_writer.add_scalars("Loss", {"training_loss": np.mean(plot_loss)}, self.current_epoch)
        self.save_checkpoint()
        return plot_loss

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        plot_loss = {d: [] for d in self.config.datasets}
        self.models.eval()
        test_loss = 0
        with torch.no_grad():
            im_save = {d: {j: 0 for j in self.config.datasets} for d in self.config.datasets}
            for data in self.data_loader.valid_loader[self.config.datasets[0]][self.config.datasets[1]]:
                # print(compt/self.data_loader.train_iterations)
                # compt+=1
                pos_enc = data.get("coordenc").to(self.device)
                source_patch = data.get("source_one_hot")
                target_patch = data.get("target_one_hot")
                tv = data.get("target_data")[:, 0]
                if self.config.use_pos:
                    rec = self.models(source_patch.float(), pos_enc.float())
                else:
                    rec = self.models(source_patch.float())
                loss = nn.CrossEntropyLoss(ignore_index=0)(rec, tv)

                if im_save[self.config.datasets[0]][self.config.datasets[1]] == 0:
                    out_img = self.data_loader.plot_samples_per_epoch(source_patch, target_patch, rec, rec,
                                                                      self.config.datasets[0],
                                                                      self.config.datasets[1],
                                                                      self.current_epoch, data.get("coordinate"))
                    im_save[self.config.datasets[0]][self.config.datasets[1]] += 1
                plot_loss[self.config.datasets[0]].append(loss.item())
                plot_loss[self.config.datasets[1]].append(loss.item())

        if self.config.tensorboard:
            self.summary_writer.add_scalars("Loss", {"validation_loss": np.mean(plot_loss)}, self.current_epoch)
            self.summary_writer.add_image('train/generated_image', out_img.transpose(2, 0, 1), self.current_epoch)
        return plot_loss

    def test(self):
        with torch.no_grad():
            ##### Read ground_truth_file
            self.load_checkpoint("model_best.pth.tar")
            self.models.eval()
            nb_it=0

            res_oa = {d: {j: [0, 0] for j in self.config.datasets} for d in self.config.datasets}
            conf_matrix = {
                d: {j: np.zeros((self.data_loader.n_classes[j] + 1, self.data_loader.n_classes[j] + 1)) for j in
                    self.config.datasets} for d in self.config.datasets}

            for data in self.data_loader.test_loader[self.config.datasets[0]][self.config.datasets[1]]:
                # print(compt/self.data_loader.train_iterations)
                # compt+=1
                pos_enc = data.get("coordenc").to(self.device)
                source_patch = data.get("source_one_hot")
                tv = data.get("target_data")[:, 0]

                if self.config.use_pos:
                    rec = self.models(source_patch.float(), pos_enc.float())
                else:
                    rec = self.models(source_patch.float())

                y_pred = torch.argmax(rec, dim=1)

                y_pred = y_pred.int().view(-1).cpu().numpy()
                y_targ = tv.int().view(-1).cpu().numpy()

                y_pred = y_pred[y_targ != 0]
                y_targ = y_targ[y_targ != 0]

                where_id = y_pred == y_targ
                T = np.sum(where_id)
                nb = len(y_pred)

                # print(res_oa[d][d2])
                res_oa[self.config.datasets[0]][self.config.datasets[1]][0] += T
                res_oa[self.config.datasets[0]][self.config.datasets[1]][1] += nb

                labels = range(self.data_loader.n_classes[self.config.datasets[1]] + 1)
                conf_matrix[self.config.datasets[1]][self.config.datasets[1]] += confusion_matrix(y_targ, y_pred, labels=labels)

            # df = pd.DataFrame(shape_info.data)
            # df.to_pickle(join(self.config.out_dir, "regional_carac.pkl"))
            with open(join(self.config.out_dir, "regional_carac.json"), 'w') as fp:
                json.dump(shape_info.data, fp)

            # shape_info.plot_usefull_hist()

            res = {d: {j: res_oa[d][j][0] / (res_oa[d][j][1] + 0.00001) for j in self.config.datasets} for d in
                   self.config.datasets}
            with open(join(self.config.out_dir, "accuracy_assessement.json"), 'w') as fp:
                json.dump(res, fp)

            res = {d: {j: conf_matrix[d][j].tolist() for j in self.config.datasets} for d in self.config.datasets}
            with open(join(self.config.out_dir, "per_class_accuracy_assessement.json"), 'w') as fp:
                json.dump(res, fp)

            PltPerClassMetrics()(conf_matrix, savefig=self.config.out_dir + "/per_class")

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        self.logger.info("Please wait while finalizing the operation.. Thank you")
        torch.cuda.empty_cache()
        if self.config.tensorboard:
            self.tensorboard_process.kill()
            self.summary_writer.close()
        # self.save_checkpoint()
        # self.summary_writer.export_scalars_to_json("{}all_scalars.json".format(self.config.summary_dir))
        self.data_loader.finalize()
