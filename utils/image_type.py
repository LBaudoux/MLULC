import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

class ImageType(object):
    def __init__(self, producer, name, resolution, year, no_data_value):
        self.producer = producer
        self.name = name
        self.resolution = str(resolution)
        self.year = str(year)
        self.no_data_value = no_data_value

    def get_as_dict(self):
        return {"producer": self.producer,
                "name": self.name,
                "resolution": self.resolution,
                "year": self.year,
                "no_data_value": self.no_data_value}


class LandCover(ImageType):
    def __init__(self, producer, name, resolution, year, no_data_value, id_labels, labels_name, number_of_channels=1,
                 cmap=None):
        super().__init__(producer, name, resolution, year, no_data_value)
        self.number_of_class = len(id_labels)
        if cmap:
            n_bin = len(labels_name)
            cmap_name = 'my_colormap'
            self.cmap = cmap
            self.matplotlib_cmap = LinearSegmentedColormap.from_list(cmap_name, np.array(cmap) / 255, N=n_bin)
        else:
            self.matplotlib_cmap = plt.get_cmap("viridis")
        self.id_labels = id_labels
        self.labels_name = labels_name
        self.number_of_channels = number_of_channels

    def get_as_dict(self):
        return {"producer": self.producer,
                "name": self.name,
                "resolution": self.resolution,
                "year": self.year,
                "no_data_value": self.no_data_value,
                "number_of_class": self.number_of_class,
                "cmap": self.cmap,
                "id_labels": self.id_labels,
                "number_of_channels": self.number_of_channels}


class Imagerie(ImageType):
    def __init__(self, producer, name, resolution, year, no_data_value, cmap=None):
        super().__init__(producer, name, resolution, year, no_data_value)
        self.cmap = cmap


class OSO(LandCover):
    def __init__(self, year=2018):

        # if year == 2018:
        #     labels_name = ["batis denses", "batis diffus", "zones ind et com", "surfaces routes", "colza", "cereales",
        #                    "protéagineux", "soja", "tournesol", "mais", "riz", "tubercules", "prairies", "vergers",
        #                    "vignes", "feuillus", "coniferes", "pelouses", "landes", "roches", "sable", "neige", "eau"]
        #     labels_name = ["Dense urban", "Sparse urban", "ind and com", "roads", "rapeseeds", "cereals",
        #                    "protein crops", "soy", "sunflower", "maize", "rice", "tubers", "meadow", "orchards",
        #                    "vineyards", "Broad-leaved", "coniferous", "lawn", "shrubs", "rocks", "sand", "snow",
        #                    "water"]
        #     id_labels = range(1, 24)
        #     cmap = [(255, 0, 255), (255, 85, 255), (255, 170, 255), (0, 255, 255), (255, 255, 0),
        #             (208, 255, 0),
        #             (161, 214, 0), (255, 170, 68), (214, 214, 0), (255, 85, 0), (197, 255, 255), (170, 170, 97),
        #             (170, 170, 0), (170, 170, 255), (85, 0, 0), (0, 156, 0), (0, 50, 0), (170, 255, 0),
        #             (85, 170, 127),
        #             (255, 0, 0), (255, 184, 2), (190, 190, 190), (0, 0, 255)]

        # elif year == 2017 or year == 2016:
        id_labels = [11, 12, 31, 32, 34, 36, 41, 42, 43, 44, 45, 46, 51, 53, 211, 221, 222]
        id_labels = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
        labels_name = ["culture ete", "culture hiver", "foret feuillus", "foret coniferes", "pelouses",
                       "landes ligneuses", "urbain dense", "urbain diffus", "zones ind et com", "surfaces routes",
                       "surfaces minerales", "plages et dunes", "eau", "glaciers ou neige", "prairies", "vergers",
                       "vignes"]
        labels_name = ["summer crops", "winter crops", "Broad-leaved", "coniferous", "lawn", "shrubs",
                       "Dense urban", "Sparse urban", "zones ind et com", "roads", "rocks", "sand", "water", "snow",
                       "meadow", "orchards", "vineyards"]
        cmap = [(255, 85, 0), (255, 255, 127), (0, 156, 0), (0, 50, 0), (170, 255, 0),
                (85, 170, 127), (255, 0, 255), (255, 85, 255), (255, 170, 255), (0, 255, 255), (255, 0, 0),
                (255, 184, 2), (0, 0, 255), (190, 190, 190), (170, 170, 0), (170, 170, 255), (85, 0, 0)]
        # else:
        #     raise ValueError("Unknow year {] for OSO".format(year))

        super().__init__("cesbio", "oso", 10, year, 0, id_labels, labels_name, number_of_channels=1, cmap=cmap)


class CLC(LandCover):
    def __init__(self, res=100, year=2018, level=3):
        self.res = res
        self.level = level
        id_labels = [111, 112, 121, 122, 123, 124, 131, 132, 133, 141, 142, 211, 212, 213,
                     221, 222, 223, 231, 241, 242, 243, 244, 311, 312, 313, 321, 322, 323,
                     324, 331, 332, 333, 334, 335, 411, 412, 421, 422, 423, 511, 512, 521,
                     522, 523]

        id_labels = np.unique(np.floor(np.array(id_labels) / (10 ** (3 - level)))).astype(
            np.int16).tolist()

        cmap = [(230, 0, 77), (255, 0, 0), (204, 77, 242), (204, 0, 0), (230, 204, 204),
                (230, 204, 230), (166, 0, 204), (166, 77, 0), (255, 77, 255), (255, 166, 255), (255, 230, 255),
                (255, 255, 168), (255, 255, 0), (230, 230, 0), (230, 128, 0), (242, 166, 77), (230, 166, 0),
                (230, 230, 77), (255, 230, 166), (255, 230, 77), (230, 204, 77), (242, 204, 166), (128, 255, 0),
                (0, 166, 0), (77, 255, 0), (204, 242, 77), (166, 255, 128), (166, 230, 77), (166, 242, 0),
                (230, 230, 230), (204, 204, 204), (204, 255, 204), (0, 0, 0), (166, 230, 204), (166, 166, 255),
                (77, 77, 255), (204, 204, 255), (230, 230, 255), (166, 166, 230), (0, 204, 242), (128, 242, 230),
                (0, 255, 166), (166, 255, 230), (230, 242, 255)]
        labels_name = ["no data", "Continuous urban", "Disc urban", "Ind and com", 'road and rail', 'port', 'airport',
                       'mine', 'dump', 'construction', 'green urb', 'leisure', 'non irrigated crops',
                       'perm irrigated crops', 'rice', 'vineyards', 'fruit', 'olive', 'pastures', 'ann + perm crops',
                       'complex culti', 'mix argi and nat', 'agro-forestry', 'broad leaved', 'conifere', 'mixed forest',
                       'natural grass', 'moors', 'sclerophyllous', 'transi wood-shrub', 'sand', 'rocks',
                       'sparsely vege', 'burnt', 'snow', 'marshes', 'peat bogs', 'salt marshes', 'salines',
                       'intertidal flats', 'river', 'lakes', 'lagoons', 'estuaries', 'sea']
        if level == 1:
            labels_name = ["Batis", "Agri.", "Nat.", "Z.H.", "Eau"]
            cmap = [(230, 242, 255), (230, 0, 77), (255, 230, 166), (77, 255, 0), (230, 230, 255), (0, 0, 255)]
        elif level == 2:
            cmap = [(242, 0, 35), (217, 121, 169), (196, 51, 153), (255, 198, 255), (247, 247, 56), (234, 153, 26),
                    (230, 230, 77), (246, 217, 121), (68, 225, 0), (175, 242, 70), (160, 183, 168), (121, 121, 255),
                    (200, 200, 246), (43, 234, 213), (198, 249, 243)]
            labels_name = ["Urban fabric", "Ind. and commercial", "Mine & construction", 'artifial vegetated', 'arable',
                           'perm. Crops', 'pastures', 'heterogene agri', 'forests', 'scrubs', 'non vegetated natural',
                           'Inland wet', 'Maritime wet', 'inland wat', 'marine wat']

        super().__init__("copernicus", "clc", res, year, 0, id_labels, labels_name, number_of_channels=1, cmap=cmap)


class OCSGE(LandCover):
    def __init__(self, kind="cover", year=2013):

        if kind == "cover":
            labels_name = ["Zones baties", "Zones non baties", "Zones a materiaux mineraux", "Zones a autres materiaux",
                           "Sols nus", "Surfaces d'eau", "Neves et glaciers", "Peuplement de feuillus",
                           "Peuplement de coniferes", "Peuplement mixte", "Formations arbustives et sous-arbrisseaux",
                           "Autres formations ligneuses", "Formations herbacees", "Autres formations non ligneuses"]
            id_labels = [1111, 1112, 1121, 1122, 121, 122, 123, 2111, 2112, 2113, 212, 213, 221, 222]
            cmap = [(255, 55, 121), (255, 145, 145), (255, 209, 201), (166, 77, 0), (204, 204, 204),
                    (0, 204, 242),
                    (166, 230, 204), (128, 255, 0), (0, 166, 0), (128, 190, 0), (166, 255, 128), (230, 128, 0),
                    (204, 242, 77), (204, 255, 204)]

        elif kind == "use":
            pass
        else:
            raise ValueError("Unknow type {] for OSO".format(year))

        super().__init__("ign", "ocsge", 5, year, 0, id_labels, labels_name, number_of_channels=1, cmap=cmap)


class SRTM(Imagerie):
    def __init__(self, cmap=None):
        producer = "NASA"
        name = "SRTM"
        resolution = 30
        year = 2015
        no_data_value = -32000
        super(SRTM, self).__init__(producer, name, resolution, year, no_data_value, cmap=cmap)


class Sentinel2(Imagerie):
    def __init__(self, cmap=None):
        producer = "ESA"
        name = "S2"
        resolution = 10
        year = 2018
        no_data_value = -32000
        super(Sentinel2, self).__init__(producer, name, resolution, year, no_data_value, cmap=cmap)