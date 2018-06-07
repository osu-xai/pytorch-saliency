from copy import deepcopy


class Saliency(object):
    """ Abstract class for saliency """

    def __init__(self, model):
        self.model = deepcopy(model)
        self.model.eval()


    def generate_saliency(self, model, input, target):
        raise "Method not implemented!"
