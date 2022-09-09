"""
Approche par couple mais on maxime tout en me temps !
"""
import numpy as np
from sklearn.metrics import confusion_matrix
import shutil
import torch
from torch import nn
import torch.optim as optim
import json

from agents.base import BaseAgent
from graphs.models.universal_embedding import UnivEmb as EncDec

from datasets.landcover_to_landcover import LandcoverToLandcoverDataLoader

from utils.misc import print_cuda_statistics

from os.path import join

from utils.plt_utils import plt_loss2 as plt_loss
from utils.plt_utils import PltPerClassMetrics

from utils.tensorboardx_utils import tensorboard_summary_writer

from utils.misc import timeit

from torch.optim.lr_scheduler import ReduceLROnPlateau

class MultiLULCAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)
        # torch.set_default_dtype(torch.half)

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
            torch.cuda.set_device(self.config.gpu_device)
            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")

        # define data_loader
        self.data_loader = LandcoverToLandcoverDataLoader(config=config,device=self.device,pos_enc=True)

        # Get required param for network initialisation
        input_channels=self.data_loader.input_channels
        output_channels=self.data_loader.output_channels

        #
        resizes= self.config.embedding_dim[1]//np.array(self.data_loader.real_patch_sizes)
        resizes=np.where(resizes==1,None,resizes)
        # define models
        self.models = [EncDec(input_channel,output_channel,mul=self.config.mul,softpos=self.config.softpos,number_feature_map=self.config.number_of_feature_map,embedding_dim=self.config.embedding_dim[0],memory_monger=self.config.memory_monger,up_mode=self.config.up_mode,num_groups=self.config.group_norm,decoder_depth=config.decoder_depth,mode=config.mode,resize=resize,cat=False,pooling_factors=config.pooling_factors,decoder_atrou=self.config.decoder_atrou) for input_channel,output_channel,resize in zip(input_channels,output_channels,resizes)]

        self.coord_model=nn.Sequential(nn.Linear(128, 300),nn.ReLU(inplace=True),nn.Linear(300, self.config.embedding_dim[0]),nn.ReLU(inplace=True))
        self.coord_optimizer = optim.Adam(self.coord_model.parameters(), lr=self.config.learning_rate[0])
        # define optimizer
        self.optimizers = [optim.Adam(net.parameters(), lr=self.config.learning_rate[i]) for i,net in enumerate(self.models)] #eps est obligatoire avec mode half ,eps=0.0001

        # # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_metric = 0


        if self.cuda:
            self.models = [net.to(self.device) for net in self.models]
            self.coord_model = self.coord_model.to(self.device)
            print_cuda_statistics()

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)

        # Summary Writer
        if self.config.tensorboard:
            self.summary_writer, self.tensorboard_process = tensorboard_summary_writer(config,comment=self.config.exp_name)

        if self.cuda and torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.models = [torch.nn.DataParallel(net) for net in self.models]

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
            for i,d in enumerate(self.config.datasets):
                self.models[i].load_state_dict(checkpoint['encoder_state_dict_'+d])
                self.optimizers[i].load_state_dict(checkpoint['encoder_optimizer_'+d])
                self.coord_model.load_state_dict(checkpoint['image_state_dict_' + d])
                self.coord_optimizer.load_state_dict(checkpoint['coord_optimizer_' + d])
                if self.cuda and torch.cuda.device_count() > 1:
                    print("Let's use", torch.cuda.device_count(), "GPUs!")
                    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
                    self.models[i] = torch.nn.DataParallel(self.models[i])
                    self.coord_model = torch.nn.DataParallel(self.coord_model)
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
            state['encoder_optimizer_' + d] = self.optimizers[i].state_dict()
            state['coord_optimizer_' + d] = self.coord_optimizer.state_dict()
            if torch.cuda.device_count() > 1 and self.cuda:
                state['encoder_state_dict_'+d]=self.models[i].module.state_dict()
                state['image_state_dict_' + d] = self.coord_model.module.state_dict()
            else:
                state['encoder_state_dict_' + d] = self.models[i].state_dict()
                state['image_state_dict_' + d] = self.coord_model.state_dict()

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
        plot_training_loss = {d:[] for d in self.config.datasets}
        plot_validation_loss = {d:[] for d in self.config.datasets}
        if self.config.use_scheduler:
            scheduler = {d:ReduceLROnPlateau(self.optimizers[i], 'min',factor=0.5,patience=5)  for i,d in enumerate(self.config.datasets)}

        self.logger.info("Start training !")
        for epoch in range(1, self.config.max_epoch + 1):
            self.logger.info("Training epoch {}/{} ({:.0f}%):\n".format(epoch, self.config.max_epoch,
                                                                        (epoch - 1) / self.config.max_epoch * 100))
            train_loss = self.train_one_epoch()
            # print(train_loss)
            for d,l in train_loss.items():
                plot_training_loss[d].extend(l)
            torch.cuda.empty_cache()
            if epoch % self.config.validate_every == 0:
                self.logger.info("Validation epoch {}/{} ({:.0f}%):\n".format(epoch, self.config.max_epoch,
                                                                              (epoch - 1) / self.config.max_epoch * 100))
                validation_loss = self.validate()
                for d, l in validation_loss.items():
                    plot_validation_loss[d].append([self.current_iteration,np.mean(l)])
                tmp=[v for v in validation_loss.values()]
                vl=np.mean([ item for elem in tmp for item in elem])
                if vl < loss_ref:
                    self.logger.info("Best model for now  : saved ")
                    loss_ref = vl
                    self.save_checkpoint(is_best=1)
                torch.cuda.empty_cache()
                if self.config.use_scheduler:
                    for d in self.config.datasets:
                        scheduler[d].step(plot_validation_loss[d][-1][1])
            self.current_epoch += 1
            if epoch > 1 and epoch >= 2 * self.config.validate_every:
                plt_loss()(plot_training_loss, plot_validation_loss, savefig=join(self.config.out_dir, "loss.png"))
        self.logger.info("Training ended!")


    @timeit
    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        self.logger.info("training one ep")
        plot_loss = {d:[] for d in self.config.datasets}

        [model.train() for model in self.models]
        self.coord_model.train()

        # compt=0
        batch_idx=0
        data_loader =  {source:{target:iter(val) for target,val in targetval.items()} for source,targetval in self.data_loader.train_loader.items()}
        end=False
        while not end:
            for source,targetval in data_loader.items():
                i_source = self.config.datasets.index(source)
                for target,dl in targetval.items():
                    i_target = self.config.datasets.index(target)
                    try:
                        data=next(dl)
                    except:
                        end=True
                        break
                    pos_enc=data.get("coordenc").to(self.device)
                    source_patch=data.get("source_one_hot")
                    target_patch = data.get("target_one_hot")
                    sv=data.get("source_data")[:,0]
                    tv=data.get("target_data")[:,0]
                    self.optimizers[i_source].zero_grad(set_to_none=True)
                    self.coord_optimizer.zero_grad(set_to_none=True)
                    self.optimizers[i_target].zero_grad(set_to_none=True)
                    if self.config.use_pos:
                        pos_enc=self.coord_model(pos_enc.float()).unsqueeze(2).unsqueeze(3)
                        embedding,rec = self.models[i_source](source_patch,full=True,res=pos_enc)
                    else:
                        embedding, rec = self.models[i_source](source_patch, full=True)

                    loss =  nn.CrossEntropyLoss(ignore_index=0)(rec, sv) # self reconstruction loss
                    if self.config.use_pos:
                        embedding2, rec = self.models[i_target](target_patch, full=True,res=pos_enc)
                    else:
                        embedding2, rec = self.models[i_target](target_patch, full=True)
                    loss+= nn.CrossEntropyLoss(ignore_index=0)(rec,tv) # self reconstruction loss
                    loss += torch.nn.MSELoss()(embedding, embedding2 ) # similar embedding loss




                    _, rec = self.models[i_target](embedding)
                    loss += nn.CrossEntropyLoss(ignore_index=0)(rec,tv) # translation loss

                    _, rec = self.models[i_source](embedding2)
                    loss += nn.CrossEntropyLoss(ignore_index=0)(rec, sv) # translation loss

                    loss.backward()
                    self.optimizers[i_source].step()
                    self.optimizers[i_target].step()
                    self.coord_optimizer.step()
                if end:
                    break
                batch_idx += 1
            plot_loss[source].append([self.current_iteration, loss.item()])




            self.current_iteration += 1
        self.save_checkpoint()
        return plot_loss


    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        plot_loss = {d:[] for d in self.config.datasets}
        [model.eval() for model in self.models]
        self.coord_model.eval()

        test_loss = 0
        with torch.no_grad():
            im_save={d:{j:0 for j in self.config.datasets} for d in self.config.datasets}
            data_loader = {source: {target: iter(val) for target, val in targetval.items()} for source, targetval in
                           self.data_loader.valid_loader.items()}
            end = False
            while not end:
                for source, targetval in data_loader.items():
                    i_source = self.config.datasets.index(source)
                    for target, dl in targetval.items():
                        i_target = self.config.datasets.index(target)
                        try:
                            data = next(dl)
                        except:
                            end = True
                            break
                        pos_enc = data.get("coordenc").to(self.device)
                        source_patch = data.get("source_one_hot")
                        target_patch = data.get("target_one_hot")

                        if self.config.use_pos:
                            pos_enc = self.coord_model(pos_enc.float()).unsqueeze(2).unsqueeze(3)
                            embedding, rec = self.models[i_source](source_patch.float(), full=True,res=pos_enc)
                        else:
                            embedding, rec = self.models[i_source](source_patch.float(), full=True)

                        _, trad = self.models[i_target](embedding)

                        loss = nn.CrossEntropyLoss(ignore_index=0)(trad,torch.argmax(target_patch, 1))

                        if im_save[source][target] == 0:
                            out_img = self.data_loader.plot_samples_per_epoch(source_patch, target_patch, trad,embedding,source,target,self.current_epoch,data.get("coordinate"))
                            im_save[source][target]=1
                        if im_save[source][source] == 0:
                            out_img = self.data_loader.plot_samples_per_epoch(source_patch, source_patch, rec,embedding,source,source,self.current_epoch,data.get("coordinate"))
                            im_save[source][source]=1
                        plot_loss[target].append(loss.item())

                if end:
                    break
        return plot_loss


    def test(self):
        with torch.no_grad():
            ##### Read ground_truth_file
            self.load_checkpoint("model_best.pth.tar")
            [model.eval() for model in self.models]
            self.coord_model.eval()


            res_oa={d:{j:[0,0] for j in self.config.datasets} for d in self.config.datasets}
            conf_matrix = {d: {j:np.zeros((self.data_loader.n_classes[j]+1,self.data_loader.n_classes[j]+1)) for j in self.config.datasets} for d in self.config.datasets}

            for source, targetval in self.data_loader.test_loader.items():
                i_source = self.config.datasets.index(source)
                for target, val in targetval.items():
                    i_target = self.config.datasets.index(target)
                    for nb_it,data in enumerate(val):
                        pos_enc = data.get("coordenc").to(self.device)
                        source_patch = data.get("source_one_hot")
                        tv = data.get("target_data")[:, 0]


                        if self.config.use_pos:
                            pos_enc = self.coord_model(pos_enc.float()).unsqueeze(2).unsqueeze(3)
                            embedding, _ = self.models[i_source](source_patch.float(), full=True, res=pos_enc)
                        else:
                            embedding, _ = self.models[i_source](source_patch.float(), full=True)

                        _,trad = self.models[i_target](embedding)



                        y_pred = torch.argmax(trad, dim=1)

                        y_pred = y_pred.int().view(-1).cpu().numpy()
                        y_targ = tv.int().view(-1).cpu().numpy()


                        y_pred =y_pred[y_targ!=0]
                        y_targ=y_targ[y_targ!=0]

                        where_id=y_pred == y_targ
                        T = np.sum(where_id)
                        nb = len(y_pred)


                        res_oa[source][target][0] += T
                        res_oa[source][target][1] += nb

                        labels=range(self.data_loader.n_classes[target]+1)
                        conf_matrix[source][target] += confusion_matrix(y_targ,y_pred,labels=labels)

            res={d:{j:res_oa[d][j][0]/(res_oa[d][j][1]+0.00001) for j in self.config.datasets} for d in self.config.datasets}
            with open(join(self.config.out_dir, "accuracy_assessement.json"), 'w') as fp:
                json.dump(res, fp)

            res={d:{j:conf_matrix[d][j].tolist() for j in self.config.datasets} for d in self.config.datasets}
            with open(join(self.config.out_dir, "per_class_accuracy_assessement.json"), 'w') as fp:
                json.dump(res, fp)

            PltPerClassMetrics()(conf_matrix,savefig=self.config.out_dir+"/per_class")


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
        # self.data_loader.finalize()
