from segmentation.run_segmentation import Segmentation
from segmentation.configs.config import config



if __name__ == '__main__':
    seg_worker = Segmentation(config,['20190412130806', '20190718213917'])
    seg_worker.run_segmentation('20190412130806')
    seg_worker.run_segmentation('20190718213917')