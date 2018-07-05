import torch
import torch.nn as nn

from saliency.saliency import Saliency

class DeconvSaliency(Saliency):
    """docstring for DeconvSaliency."""
    def __init__(self, model):
        super(DeconvSaliency, self).__init__(model)


    def guided_relu_hook(self, module, grad_in, grad_out):
        return (torch.nn.functional.relu(grad_in[0]), )


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
        #print(input)

        return (input.grad.clone()[0] * input)
