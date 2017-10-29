import torch
import torch.optim as optim
from pyroTrain import train, test, tile_plot, normalize_sn_imgs
from pyroModel import TissueNormalizer
import numpy as np
import logging as log

torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    log.basicConfig(filename='stainnorm.log',
                    format='%(asctime)s %(message)s',
                    datefmt='%Y/%m/%d %I:%M:%S %p',
                    level=log.DEBUG)
    log.info('__START__')

    torch.cuda.device(0)

    model = torch.load('trained_models/keep/tissue_sn_ep034.t7')
    # model = TissueNormalizer()
    model.cuda()
  
    # print(model.get_training_params())
    # optimizer = optim.SGD(model.get_training_params(), lr=0.001, momentum=0.9, weight_decay=0.00001)
    # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00001)
  
#    train(120, model, optimizer,
#          mse_weight=1.0,
#          #cce_weight=0.3,
#          #preview_plot=plot_examples_mnist,
#          #savepath=SAVE_DIR+'training_mixf03_n32.png'
#    )

#    test(
#        model,
#        tile_plot,
#        savepath='/home/temp/bug/resultsDump/pyt_sn_test{:03d}.png'
#    )

    normalize_sn_imgs(model)

    log.info('__DONE__')   
