import torch
import torch.nn as nn

from saliency.saliency import Saliency

class GuidedSaliency(Saliency):
    """Class for computing guided saliency"""
    def __init__(self, model):
        super(GuidedSaliency, self).__init__(model)


    def guided_relu_hook(self, module, grad_in, grad_out):
        return (torch.clamp(grad_in[0], min=0.0), )


    def generate_saliency(self, input, target):
        input.requires_grad = True

        self.model.zero_grad()

        for module in self.model.modules():
            if type(module) == nn.ReLU:
                module.register_backward_hook(self.guided_relu_hook)

        output = self.model(input)

        grad_outputs = torch.zeros_like(output)

        grad_outputs[:, target] = 1

        output.backward(gradient = grad_outputs)

        input.requires_grad = False
        


        return (input.grad.clone()[0] * input)
