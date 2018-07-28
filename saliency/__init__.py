# from saliency.vanilla import *
# from saliency.guided import *
# from saliency.deconv import *
# from saliency.excitation_bp import *
# from saliency.test import *
# from saliency.perturbation import *
from saliency.perturbation_2 import *
# from saliency.lime import *
from .types import *
import torch
import torchvision
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.stats



types_map = {
    #SaliencyMethod.VANILLA : VanillaSaliency,
    #SaliencyMethod.GUIDED : GuidedSaliency,
    #SaliencyMethod.DECONV : DeconvSaliency,
    #SaliencyMethod.TEST : TestSaliency,
    #SaliencyMethod.PERTURBATION : PerturbationSaliency,
    SaliencyMethod.PERTURBATION_2: Perturbation_2Saliency
    #SaliencyMethod.LIME: LimeSaliency
    #SaliencyMethod.EXCITATIONBP : ExcitationBPSaliency
}


def generate_saliency(model, input, target, type = SaliencyMethod.PERTURBATION_2):
    saliencies = {}

    output = types_map[type](model).generate_saliency(input, target)
    saliencies[MapType.ORIGINAL] = output
    saliencies[MapType.INPUT] = input
    # np.savetxt('output.txt', output.detach().numpy())

    # if type == SaliencyMethod.PERTURBATION_2:
    #     # print('IN PERTURBATION 2!!@!@!!')
    #     # print('TARGET ACQUIRED!')
    #     # print(target)
    #     np.savetxt('output.txt', output.detach().numpy())
    #     np.savetxt('raw_hp_saliency.txt', output.view(40, 40, 5)[:, :, 0].detach().numpy())
    #     np.savetxt('raw_enemytank_saliency.txt', output.view(40, 40, 5)[:, :, 1].detach().numpy())
    #     np.savetxt('raw_size_saliency.txt', output.view(40, 40, 5)[:, :, 2].detach().numpy())
    #     np.savetxt('raw_type_saliency.txt', output.view(40, 40, 5)[:, :, 3].detach().numpy())
    #     np.savetxt('raw_frenemy_saliency.txt', output.view(40, 40, 5)[:, :, 4].detach().numpy())
    #     #generate_graph(output, 'output')
    #     #print(model)


    pos_map = output.clamp(min = 0)
    # mean = torch.mean(pos_map)
    # stdv = torch.std(pos_map)
    # n = math.sqrt(8000)
    # print('pos_map.max()')
    # print(pos_map.max())
    # print('pos_map.norm()')
    # print(pos_map.norm())
    # print('min')
    # print(pos_map.min())
    # print('np max')
    # print(np.amax(pos_map.detach().numpy()))
    #pos_map = pos_map / pos_map.max()
    # pos_map = torch.log2(pos_map)
    saliencies[MapType.POSITIVE] = pos_map
    # pos_map = (pos_map - mean)/stdv
    # if type == SaliencyMethod.PERTURBATION_2:
    #     np.savetxt('pos_map.txt', pos_map.detach().numpy())

    neg_map = output.clamp(max = 0)
    neg_map *= -1
    # print('neg_map.max()')
    # print(neg_map.max())
    # print('min')
    # print(neg_map.min())
    # print(np.amax(neg_map.detach().numpy()))
    # neg_map = neg_map / neg_map.max()
    # mean = torch.mean(neg_map)
    # stdv = torch.std(neg_map)
    # neg_map = (neg_map - mean)/stdv
    #neg_map = torch.log2(neg_map)

    saliencies[MapType.NEGATIVE] = neg_map
    # if type == SaliencyMethod.PERTURBATION_2:
    #     np.savetxt('neg_map.txt', neg_map.detach().numpy())


    abs_map = output.abs()
    #abs_map = torch.log(abs_map)
    # print('abs_map.max()')
    # print(abs_map.max())
    # print('min')
    # print(abs_map.min())
    # print(np.amax(abs_map.detach().numpy()))
    #abs_map = abs_map / abs_map.max()
    # mean = torch.mean(neg_map)
    # stdv = torch.std(neg_map)
    #abs_map = (abs_map - mean)/(stdv/n)
    #abs_map =
    #abs_map = torch.log2(abs_map)
    # if type == SaliencyMethod.PERTURBATION_2:
    #     np.savetxt('abs_map.txt', abs_map.detach().numpy())
        # generate_graph(abs_map, 'abs_map')
        #generate_graph(abs_map.pow(0.4), 'abs_map pow 1.4')

    #abs_map = abs_map.pow(2)
    saliencies[MapType.ABSOLUTE] = abs_map


    return saliencies

# def generate_graph(y, name):
#     # z = list(y[0].size())
#     # x = torch.range(1, z[0])
#     plt.plot(y.squeeze().detach().numpy(), 'k')
#     plt.title(str(name))
#     mean = np.mean(y.detach().numpy())
#     median = np.median(y.detach().numpy())
#     print('Mean: '+str(mean) + ' Median: '+str(median))
#     # if mean > median:
#     #     plt.title('Skewed right/positively skewed')
#     # else:
#     #     plt.title('Skewed left/negatively skewed')
#     plt.show()
