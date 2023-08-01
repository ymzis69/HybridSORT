# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from ...utils.registry import Registry

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for datasets
It must returns an instance of :class:`Backbone`.
"""

# Person re-id datasets
from .mot17 import MOT17
from .mot20 import MOT20
from .cuhk03 import CUHK03              # 1367 ids, 26264 images
from .dukemtmcreid import DukeMTMC      # 702 ids, 16522 images
from .market1501 import Market1501      # 751 ids, 12936 images
from .msmt17 import MSMT17              # --- ids, 32621 images
from .AirportALERT import AirportALERT
from .iLIDS import iLIDS
from .pku import PKU
from .prai import PRAI
from .prid import PRID
from .grid import GRID
from .saivt import SAIVT
from .sensereid import SenseReID
from .sysu_mm import SYSU_mm
from .thermalworld import Thermalworld
from .pes3d import PeS3D
from .caviara import CAVIARa
from .viper import VIPeR
from .lpw import LPW
from .shinpuhkan import Shinpuhkan
from .wildtracker import WildTrackCrop
from .cuhksysu import CUHKSYSU          # [hgx0913]
from .dancetrack import DanceTrack      # [hgx0911]
from .cuhksysu_dancetrack import CUHKSYSU_DanceTrack        # [hgx0914]
from .cuhksysu_mot17 import CUHKSYSU_MOT17
from .cuhksysu_mot20 import CUHKSYSU_MOT20


# Vehicle re-id datasets
from .veri import VeRi
from .vehicleid import VehicleID, SmallVehicleID, MediumVehicleID, LargeVehicleID
from .veriwild import VeRiWild, SmallVeRiWild, MediumVeRiWild, LargeVeRiWild

__all__ = [k for k in globals().keys() if "builtin" not in k and not k.startswith("_")]
