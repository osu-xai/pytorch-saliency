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
    # SaliencyMethod.PERTURBATION : PerturbationSaliency,
    SaliencyMethod.PERTURBATION_2: Perturbation_2Saliency
    #SaliencyMethod.LIME: LimeSaliency
    #SaliencyMethod.EXCITATIONBP : ExcitationBPSaliency
}

#TARGET
#Q4: 0
#Q1: 1
#Q3: 2
#Q2: 3

#reward_idx
#Combined : None
#City Damaged: 0
#City Destroyed: 1
#Enemy Damaged: 2
#Enemy Destroyed: 3
#Friend Damaged: 4
#Friend Destroyed: 5


def generate_saliency(model, input, target, reward_idx, type = SaliencyMethod.PERTURBATION_2):
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

    #insert normalizing values
    a = 1
    b = 1
    c = 1
    d = 1
    e = 1
    #combined
    if target ==[0] and reward_idx==None: #Q4
        a = 378.9333
        b = 131.1732
        c = 254.2251
        d = 394.9073
        e = 374.2599
    elif target ==[2] and reward_idx==None: #Q3
        a = 315.2669
        b = 130.183
        c = 295.6078
        d = 384.6593
        e = 378.9721
    elif target ==[3] and reward_idx==None:#Q2
        a = 290.0625
        b = 142.4051
        c = 303.7997
        d = 395.7168
        e = 372.9528
    elif target ==[1] and reward_idx==None:#Q1
        a = 438.089
        b = 122.7646
        c = 255.0126
        d = 381.8545
        e = 381.5173
    #city damaged
    if target ==[0] and reward_idx==0: #Q4
        a = 166.7246
        b = 44.3321
        c = 79.457
        d = 175.2672
        e = 66.1829
    elif target ==[2] and reward_idx==0: #Q3
        a = 133.2067
        b = 63.4743
        c = 113.6081
        d = 191.8235
        e = 81.4166
    elif target ==[3] and reward_idx==0:#Q2
        a = 145.5428
        b = 32.8623
        c = 93.6012
        d = 199.7039
        e = 88.9888
    elif target ==[1] and reward_idx==0:#Q1
        a = 190.7863
        b = 56.2004
        c = 69.8642
        d = 182.1354
        e = 83.7192
    #city Destroyed
    if target ==[0] and reward_idx==1: #Q4
        a = 233.726
        b = 27.5724
        c = 111.7337
        d = 295.8649
        e = 111.1669
    elif target ==[2] and reward_idx==1: #Q3
        a = 169.9383
        b = 34.0689
        c = 137.8365
        d = 279.4826
        e = 119.1189
    elif target ==[3] and reward_idx==1:#Q2
        a = 159.3541
        b = 48.5927
        c = 152.2855
        d = 291.3912
        e = 127.0064
    elif target ==[1] and reward_idx==1:#Q1
        a = 216.9263
        b = 38.6506
        c = 114.419
        d = 292.5494
        e = 97.1712
    #enemy Damaged
    if target ==[0] and reward_idx==2: #Q4
        a = 113.8627
        b = 42.9254
        c = 48.5306
        d = 42.9917
        e = 27.0735
    elif target ==[2] and reward_idx==2: #Q3
        a = 122.9391
        b = 47.1815
        c = 30.5211
        d = 42.3685
        e = 36.4769
    elif target ==[3] and reward_idx==2:#Q2
        a = 118.762
        b = 43.7181
        c = 38.9658
        d = 50.3948
        e = 25.3126
    elif target ==[1] and reward_idx==2:#Q1
        a = 120.4472
        b = 43.1375
        c = 39.8917
        d = 48.0924
        e = 29.0898
    #enemy destroyed
    if target ==[0] and reward_idx==3: #Q4
        a = 360.408
        b = 58.664
        c = 54.0745
        d = 81.931
        e = 65.4311
    elif target ==[2] and reward_idx==3: #Q3
        a = 332.3288
        b = 55.3854
        c = 57.3984
        d = 73.2916
        e = 85.6882
    elif target ==[3] and reward_idx==3:#Q2
        a = 247.7114
        b = 48.0571
        c = 60.8469
        d = 74.6827
        e = 93.2744
    elif target ==[1] and reward_idx==3:#Q1
        a = 267.1299
        b = 60.3665
        c = 57.6483
        d = 77.1343
        e = 94.9222
    #friend damaged
    if target ==[0] and reward_idx==4: #Q4
        a = 109.7634
        b = 23.3136
        c = 39.4049
        d = 77.0253
        e = 119.5732
    elif target ==[2] and reward_idx==4: #Q3
        a = 128.5024
        b = 22.756
        c = 53.7883
        d = 74.2072
        e = 110.967
    elif target ==[3] and reward_idx==4:#Q2
        a = 87.4328
        b = 23.4764
        c = 38.3117
        d = 67.0994
        e = 101.9074
    elif target ==[1] and reward_idx==4:#Q1
        a = 147.2241
        b = 24.8987
        c = 49.7508
        d = 71.5709
        e = 114.2507
    #friend destroyed
    if target ==[0] and reward_idx==5: #Q4
        a = 188.6819
        b = 31.7709
        c = 38.9313
        d = 88.254
        e = 146.6498
    elif target ==[2] and reward_idx==5: #Q3
        a = 190.7987
        b = 37.5586
        c = 59.8844
        d = 75.7767
        e = 135.4588
    elif target ==[3] and reward_idx==5:#Q2
        a = 110.6491
        b = 38.7114
        c = 37.2908
        d = 88.0543
        e = 141.4961
    elif target ==[1] and reward_idx==5:#Q1
        a = 175.3254
        b = 32.1663
        c = 47.2841
        d = 87.8841
        e = 135.9592



    #print(reward_idx)




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


    #if type == SaliencyMethod.PERTURBATION_2:
        #np.savetxt('abs_map.txt', abs_map.detach().numpy())
        #generate_graph(abs_map, 'abs_map')
        #generate_graph2(abs_map, 'abs_map')
        #generate_graph(abs_map.pow(0.4), 'abs_map pow 1.4')
        # print('HP max' +str(abs_map.view(40, 40, 5)[:, :, 0].max()))
        # print('agent max' +str(abs_map.view(40, 40, 5)[:, :, 1].max()))
        # print('size max' +str(abs_map.view(40, 40, 5)[:, :, 2].max()))
        # print('type max' +str(abs_map.view(40, 40, 5)[:, :, 3].max()))
        # print('frenemy max' +str(abs_map.view(40, 40, 5)[:, :, 4].max()))
        # print('\n')

    # print(a)
    # print('AAAAA')

    #abs_map = abs_map.pow(2)
    abs_map = abs_map.view(40, 40, 5)
    abs_map[:, :, 0] = abs_map[:, :, 0]/a
    abs_map[:, :, 1] = abs_map[:, :, 1]/b
    abs_map[:, :, 2] = abs_map[:, :, 2]/c
    abs_map[:, :, 3] = abs_map[:, :, 3]/d
    abs_map[:, :, 4] = abs_map[:, :, 4]/e
    abs_map = abs_map.view(1, 8000)
    saliencies[MapType.ABSOLUTE] = abs_map




    return saliencies

def generate_graph(y, name):
    # z = list(y[0].size())
    # x = torch.range(1, z[0])
    plt.plot(y.squeeze().detach().numpy(), 'k')
    #plt.title(str(name))
    mean = np.mean(y.detach().numpy())
    median = np.median(y.detach().numpy())
    print('Mean: '+str(mean) + ' Median: '+str(median))
    if mean > median:
        plt.title('Skewed right/positively skewed')
    else:
        plt.title('Skewed left/negatively skewed')
    plt.show()

def generate_graph2(y, name):
    # z = list(y[0].size())
    # x = torch.range(1, z[0])
    y = y.view(40, 40, 5)
    plt.plot(y[:, :, 0].view(-1).squeeze().detach().numpy(), 'ro')
    plt.title('HP'+'\n'+' Mean: '+str(y[:, :, 0].mean())+' Median: '+str(y[:, :, 0].median())+ ' Stdv: '+str(y[:, :, 0].std()))
    plt.show()

    plt.plot(y[:, :, 1].view(-1).squeeze().detach().numpy(), 'ro')
    plt.title('Tank'+'\n'+' Mean: '+str(y[:, :, 1].mean())+' Median: '+str(y[:, :, 1].median())+ ' Stdv: '+str(y[:, :, 1].std()))
    plt.show()

    plt.plot(y[:, :, 2].view(-1).squeeze().detach().numpy(), 'ro')
    plt.title('Size'+'\n'+' Mean: '+str(y[:, :, 2].mean())+' Median: '+str(y[:, :, 2].median())+ ' Stdv: '+str(y[:, :, 2].std()))
    plt.show()

    plt.plot(y[:, :, 3].view(-1).squeeze().detach().numpy(), 'ro')
    plt.title('Type'+'\n'+' Mean: '+str(y[:, :, 3].mean())+' Median: '+str(y[:, :, 3].median())+ ' Stdv: '+str(y[:, :, 3].std()))
    plt.show()

    plt.plot(y[:, :, 4].view(-1).squeeze().detach().numpy(), 'ro')
    plt.title('Frenemy'+'\n'+' Mean: '+str(y[:, :, 4].mean())+' Median: '+str(y[:, :, 4].median())+ ' Stdv: '+str(y[:, :, 4].std()))
    plt.show()
