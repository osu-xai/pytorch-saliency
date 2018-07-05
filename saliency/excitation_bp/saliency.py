import torch
from saliency.saliency import Saliency
import numpy

try:
    import excitationbp as eb
except:
    print('Cannot import excitation bp')


class ExcitationBPSaliency(Saliency):

    def __init__(self, model):
        super(ExcitationBPSaliency, self).__init__(model)


    def generate_saliency(self, input, target):
        eb.use_eb(True)
        inputs = Variable(torch.Tensor(input))
        prob_outputs_zero = Variable(torch.zeros(1,4))
        zero_id = target
        prob_outputs_zero.data[0:1,zero_id] += 1
        prob_inputs_zero = eb.excitation_backprop(net, inputs, prob_outputs_zero, contrastive=True)
        ebX = prob_inputs_zero.view(inputs.shape).data.abs()
        #print(input)
        #torch.set_printoptions(threshold=10000)
        #print(input.numpy().shape)
        #f = open('input.txt', 'w+')
        #f.write(str(input.numpy()))
        #f.close()

        return ebX
