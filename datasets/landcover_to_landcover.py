from torch.utils.data import DataLoader
import imageio
from torch.utils.data import Dataset
from os.path import join
import numpy as np
import json
from os.path import basename
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import Compose
from datasets.transforms import ToOneHot,CoordEnc,FlipTransform,RotationTransform
import h5py
from sklearn.decomposition import PCA
from matplotlib.colors import LinearSegmentedColormap
cmap_dict={"clc.hdf5":[(255,255,255),(230, 0, 77), (255, 0, 0), (204, 77, 242), (204, 0, 0), (230, 204, 204),
                (230, 204, 230), (166, 0, 204), (166, 77, 0), (255, 77, 255), (255, 166, 255), (255, 230, 255),
                (255, 255, 168), (255, 255, 0), (230, 230, 0), (230, 128, 0), (242, 166, 77), (230, 166, 0),
                (230, 230, 77), (255, 230, 166), (255, 230, 77), (230, 204, 77), (242, 204, 166), (128, 255, 0),
                (0, 166, 0), (77, 255, 0), (204, 242, 77), (166, 255, 128), (166, 230, 77), (166, 242, 0),
                (230, 230, 230), (204, 204, 204), (204, 255, 204), (0, 0, 0), (166, 230, 204), (166, 166, 255),
                (77, 77, 255), (204, 204, 255), (230, 230, 255), (166, 166, 230), (0, 204, 242), (128, 242, 230),
                (0, 255, 166), (166, 255, 230), (230, 242, 255)],
           "oso.hdf5":[ (255,255,255),(255, 0, 255), (255, 85, 255), (255, 170, 255), (0, 255, 255), (255, 255, 0),
                    (208, 255, 0),
                    (161, 214, 0), (255, 170, 68), (214, 214, 0), (255, 85, 0), (197, 255, 255), (170, 170, 97),
                    (170, 170, 0), (170, 170, 255), (85, 0, 0), (0, 156, 0), (0, 50, 0), (170, 255, 0),
                    (85, 170, 127),
                    (255, 0, 0), (255, 184, 2), (190, 190, 190), (0, 0, 255)],
           "mos.hdf5":[ (255,255,255),(132,202,157),(230,223,205),(249,245,208),(211,238,251),(180,193,170),(255,205,93),(192,43,64),(187,184,220),(58,147,169),(0,0,0),(138,140,143)],
           "ocsge_o.hdf5":[ (255,255,255),(255, 55, 122), (255, 145, 145), (255, 255, 153), (166, 77, 0), (204, 204, 204),
                    (0, 204, 242),
                    (166, 230, 204), (128, 255, 0), (0, 166, 0), (128, 190, 0), (166, 255, 128), (230, 128, 0),
                    (204, 242, 77), (204, 255, 204)],
           "ocsge_u.hdf5":[ (255,255,255),(255,255,168),(0,128,0),(166,0,204),(0,0,153),(230,0,77),(204,0,0),(90,90,90),(230,204,230),(0,102,255),(255,0,0),(255,75,0),(255,77,255),(64,64,64),(240,240,40)],
           "cgls.hdf5":[ (255,255,255),(46,128,21),(132,151,0),(255,187,34),(255,255,76),(0,150,160),(250,230,160),(180,180,180),(240,150,255),(250,0,0),(240,240,240),(0,50,200),(0,0,128)]}
label_dict={"clc.hdf5":["no data", "cont. urban", "disc urban", "ind/ om", 'road/ rail', 'port', 'airport',
                       'mine', 'dump', 'construction', 'green urban', 'leisure', 'non irrigated crops',
                       'perm irrigated crops', 'rice', 'vineyards', 'fruit', 'olive', 'pastures', 'mixed crops',
                       'complex crops', 'crops + nature', 'agro-forestry', 'broad leaved', 'conifere', 'mixed forest',
                       'natural grass', 'moors', 'sclerophyllous', 'transi wood-shrub', 'sand', 'rocks',
                       'sparsely vege', 'burnt', 'snow', 'marshes', 'peat bogs', 'salt marshes', 'salines',
                       'intertidal flats', 'river', 'lakes', 'lagoons', 'estuaries', 'sea'],
            "oso.hdf5":["no data","dense urban", "sparse urban", "ind and com", "roads", "rapeseeds", "cereals",
                           "protein crops", "soy", "sunflower", "maize", "rice", "tubers", "meadow", "orchards",
                           "vineyards", "Broad-leaved", "coniferous", "lawn", "shrubs", "rocks", "sand", "snow",
                           "water"],
            "mos.hdf5":["no data","forest","semi-natural","crops","water","green urban","ind. housing","col. housing","activities","facilities","transport","Mine/dump"],
            "ocsge_o.hdf5":["no data","built","concrete","mineral","mixed materials","bare soil","water","snow","broad-leaved","neadle-leaved","mixed-trees","shrubs","vine","grass","moss"],
            "ocsge_u.hdf5":["no data","farming","forestry","extraction","fishing","house/ind/com","roads","rails","airports","fluvial transport","logistics/storage","public uti networks","transitionnal","abandoned","no-use"],
            "cgls.hdf5":["no data","closed forest","open forest","shrubland","herbaceous","wetland","moss/lichen","bare/sparse","cropland","built-up","snow","water","ocean"]
            }

class LandcoverToLandcover(Dataset):
    def __init__(self,  path,source,target,list_patch_id,mode="train",transform=None,device="cuda"):
        self.source =source
        # print(self.master_dataset)
        self.target=target
        self.source_dataset_path=join(path,source)
        self.target_dataset_path=join(path,target)
        self.device=device

        self.list_patch_id = list_patch_id
        with open(join(path, "train_test_val_60.json"), 'r') as fp:
            data = json.load(fp)
        self.list_patch_id = list(set.intersection(set(self.list_patch_id), set(data[mode])))

        # self.list_patch_id =self.list_patch_id[:250]
        self.transform=transform

    def __len__(self):
        return len (self.list_patch_id)

    def open_hdf5(self):
        self.source_dataset = h5py.File(self.source_dataset_path, "r", swmr=True, libver='latest')
        self.target_dataset = h5py.File(self.target_dataset_path, "r", swmr=True, libver='latest')

    def __getitem__(self, idx):
        if not hasattr(self, 'source_dataset'):
            self.open_hdf5()
        with torch.no_grad():
            patch_id=self.list_patch_id[idx]
            sample={"patch_id":float(patch_id)}

            tmp=self.source_dataset.get(patch_id)
            sample["source_data"]=torch.tensor(tmp[:].astype(float),dtype=torch.float,device=self.device)#.astype(float)
            tmp2 = self.target_dataset.get(patch_id)
            sample["target_data"] = torch.tensor(tmp2[:].astype(float),dtype=torch.float,device=self.device)

            s1 = sample["source_data"].shape
            s2 = sample["target_data"].shape
            m1 = torch.clip(sample["source_data"], 0, 1)
            m2 = torch.clip(sample["target_data"], 0, 1)

            s = max(s1[1], s2[1])
            m1 = torch.nn.functional.interpolate(m1[None, :], size=(s, s), mode='nearest',
                                                 recompute_scale_factor=False)
            m2 = torch.nn.functional.interpolate(m2[None, :], size=(s, s), mode='nearest',
                                                 recompute_scale_factor=False)
            m=m1*m2

            sample["source_data"] *=torch.nn.functional.interpolate(m, size=(s1[1], s1[2]), mode='nearest', recompute_scale_factor=False)[0]
            sample["target_data"] *= torch.nn.functional.interpolate(m, size=(s2[1], s2[2]), mode='nearest', recompute_scale_factor=False)[0]
            sample["source_data"]=sample["source_data"].long()
            sample["target_data"] = sample["target_data"].long()
            sample["coordinate"] = (tmp.attrs["x_coor"].astype(float), tmp.attrs["y_coor"].astype(float))

            sample["source_name"]=self.source
            sample["target_name"] = self.target

        if self.transform:
            sample = self.transform(sample)
        return sample

class LandcoverToLandcoverDataLoader:
    def __init__(self, config,to_one_hot=True,device="cuda",pos_enc=False,ampli=True,num_workers=4):
        """
        :param config:
        """
        self.config = config
        list_all_datasets = [join(config.data_folder, dataset) for dataset in config.datasets]
        self.input_channels =[]
        self.output_channels =[]
        self.real_patch_sizes=[]
        self.n_classes={}
        self.nb_patch_per_dataset=[]
        id_patch_per_dataset = {}
        for dataset in list_all_datasets:
            with h5py.File(dataset, "r") as f:
                # self.input_channels.append(f.attrs["n_channels"])
                # self.output_channels.append(f.attrs["n_channels"])
                self.real_patch_sizes.append(int(f.attrs["patch_size"]))
                self.n_classes[basename(dataset)]=int(f.attrs["numberclasses"])
                self.input_channels.append(int(f.attrs["numberclasses"])+1)
                self.output_channels.append(int(f.attrs["numberclasses"])+1)
                self.nb_patch_per_dataset.append(len(f.keys()))
                id_patch_per_dataset[dataset]=list(f.keys())

        self.couple_patch_per_dataset = {}
        self.total_couple =0
        for source in list_all_datasets:
            tmp={}
            for target in list_all_datasets:
                if source!=  target:
                    inter= list(set.intersection(set(id_patch_per_dataset[source]), set(id_patch_per_dataset[target])))
                    if len(inter)>0:
                        tmp[basename(target)] = inter
                        self.total_couple+=len(inter)
            self.couple_patch_per_dataset[basename(source)]=tmp

        self.nb_patch_per_dataset=np.array(self.nb_patch_per_dataset)
        self.nb_patch_per_dataset=self.nb_patch_per_dataset/self.nb_patch_per_dataset.sum()

        dic_list_transform={source:{target:[]for target,val in targetval.items()} for source,targetval in self.couple_patch_per_dataset.items()}
        dic_list_train_transform={source:{target:[]for target,val in targetval.items()} for source,targetval in self.couple_patch_per_dataset.items()}
        for source, targetval in self.couple_patch_per_dataset.items():
            for target, val in targetval.items():
                if ampli:
                    dic_list_train_transform[source][target].append(FlipTransform())
                    dic_list_train_transform[source][target].append(RotationTransform([0,90,180,270]))
                if to_one_hot:
                    dic_list_transform[source][target].append(ToOneHot(self.n_classes))
                if pos_enc:
                    dic_list_transform[source][target].append(CoordEnc(self.n_classes.keys()))
                dic_list_train_transform[source][target] = Compose(dic_list_train_transform[source][target] + dic_list_transform[source][target])
                dic_list_transform[source][target] = Compose(dic_list_transform[source][target])


        self.train={source:{target:LandcoverToLandcover(config.data_folder,source,target,val,mode="train",transform=dic_list_train_transform[source][target],device=device) for target,val in targetval.items()} for source,targetval in self.couple_patch_per_dataset.items()}

        self.valid={source:{target:LandcoverToLandcover(config.data_folder,source,target,val,mode="validation",transform=dic_list_transform[source][target],device=device) for target,val in targetval.items()} for source,targetval in self.couple_patch_per_dataset.items()}
        self.test={source:{target:LandcoverToLandcover(config.data_folder,source,target,val,mode="test",transform=dic_list_transform[source][target],device=device) for target,val in targetval.items()} for source,targetval in self.couple_patch_per_dataset.items()}

        if device=="cpu":
            num_workers = num_workers
            pin_memory = True
        else:
            num_workers = 0
            pin_memory = False

        self.train_loader={source:{target:DataLoader(val, batch_size=self.config.train_batch_size, shuffle=True,num_workers=num_workers,pin_memory=pin_memory) for target,val in targetval.items()} for source,targetval in self.train.items()}
        self.valid_loader = {source: {target: DataLoader(val, batch_size=self.config.valid_batch_size, shuffle=True,num_workers=num_workers,pin_memory=pin_memory) for target, val in targetval.items()} for source, targetval in self.valid.items()}
        self.test_loader = {source: {target: DataLoader(val, batch_size=self.config.test_batch_size, shuffle=True,num_workers=num_workers,pin_memory=pin_memory) for target, val in targetval.items()} for source, targetval in self.test.items()}


    def plot_samples_per_epoch(self, inputs,targets,outputs,embedding, dataset_src,dataset_tgt,epoch,coordinate,cmap="original",title=None):



        with torch.no_grad():
            if len(inputs.shape)==4:
                inputs=torch.argmax(inputs[0],dim=0)
                targets = torch.argmax(targets[0], dim=0)
            else:
                inputs = inputs[0]
                targets = targets[0]
            outputs=torch.argmax(outputs[0],dim=0)
            if title is None:
                title = join(self.config.out_dir, "Epoch_{}_Source_{}_Target_{}.png".format(epoch, dataset_src, dataset_tgt))
            else:
                title = join(self.config.out_dir,title)
            f, ax = plt.subplots(2, 2,figsize=(20,20))
            # get discrete colormap
            if cmap=="default":
                cmap_src = plt.get_cmap('RdBu', self.n_classes[dataset_src] + 1)
                cmap_tgt = plt.get_cmap('RdBu', self.n_classes[dataset_tgt] + 1)
            elif cmap=="original":
                cmap_src=LinearSegmentedColormap.from_list(dataset_src,np.array(cmap_dict[dataset_src])/ 255, N=self.n_classes[dataset_src] + 1)
                cmap_tgt=LinearSegmentedColormap.from_list(dataset_tgt,np.array(cmap_dict[dataset_tgt])/ 255, N=self.n_classes[dataset_tgt] + 1)

            m1 = ax[0][0].imshow(inputs.cpu().long().numpy(), cmap=cmap_src, vmin=0 - .5, vmax=self.n_classes[dataset_src] + 0.5)
            ax[0][0].set_title("Source")
            m2 = ax[0][1].imshow(targets.cpu().long().numpy(), cmap=cmap_tgt, vmin=0 - .5, vmax=self.n_classes[dataset_tgt] + 0.5)
            ax[0][1].set_title("target")
            m3 = ax[1][0].imshow(outputs.cpu().long().numpy(), cmap=cmap_tgt, vmin=0 - .5, vmax=self.n_classes[dataset_tgt] + 0.5)
            ax[1][0].set_title("Translation")
            # if embedding[0].shape[0]>2:
            #     emb=embedding[0,:3].cpu().numpy().transpose(1,2,0)
            # elif embedding[0].shape[0]==2:
            # #     emb = (embedding[0, 0].cpu().numpy()+embedding[0, 1].cpu().numpy())/2
            # else :
            #     emb = embedding[0,0].cpu().numpy()
            fmap_dim = embedding.shape[1]  # nchannel
            n_pix = embedding.shape[2]
            # we use a pca to project the embeddings to a RGB space
            pca = PCA(n_components=3)
            pca.fit(np.eye(fmap_dim))
            # we need to adapt dimension and memory allocation to CPU
            fmap_ = embedding[0].cpu().detach().numpy().squeeze().reshape((fmap_dim, -1)).transpose(1, 0)  # yikes
            color_vector = pca.transform(fmap_)
            # we normalize for visibility
            #color_vector = np.maximum(np.minimum(((color_vector - color_vector.mean(1, keepdims=True) + 0.5) / (2 * color_vector.std(1, keepdims=True))), 1),0)
            emb = color_vector.reshape((n_pix, n_pix, 3), order='F').transpose(1,0,2)
            #emb = np.mean(embedding[0].cpu().numpy(), axis=0)
            # m4=  ax[1][1].imshow(emb/np.max(emb),vmin=np.percentile(emb,5),vmax=np.percentile(emb,95),cmap="jet")
            m4 = ax[1][1].imshow(emb)
            ax[1][1].set_title("embedding")
            # tell the colorbar to tick at integers
            f.colorbar(m1, ticks=np.arange(0, self.n_classes[dataset_src] + 1),ax=ax[0][0])
            f.colorbar(m2, ticks=np.arange(0, self.n_classes[dataset_tgt] + 1),ax=ax[0][1])
            f.colorbar(m3, ticks=np.arange(0, self.n_classes[dataset_tgt] + 1),ax=ax[1][0])
            #f.colorbar(m4,ax=ax[1][1])
            f.suptitle('x={},y={}'.format(coordinate[0][0].item(),coordinate[1][0].item()))
        f.savefig(title)
        plt.close(f)

        return imageio.imread(title)

    def finalize(self):
        pass
