from saliency.vanilla import *
from saliency.guided import *
from saliency.deconv import *
from saliency.excitation_bp import *
from saliency.test import *
from saliency.perturbation import *
from saliency.perturbation_2 import *
from saliency.lime import *
from .types import *



types_map = {
    SaliencyMethod.VANILLA : VanillaSaliency,
    SaliencyMethod.GUIDED : GuidedSaliency,
    SaliencyMethod.DECONV : DeconvSaliency,
    SaliencyMethod.TEST : TestSaliency,
    SaliencyMethod.PERTURBATION : PerturbationSaliency,
    SaliencyMethod.PERTURBATION_2: Perturbation_2Saliency,
    SaliencyMethod.LIME: LimeSaliency
    #SaliencyMethod.EXCITATIONBP : ExcitationBPSaliency
}


def generate_saliency(model, input, target, type = SaliencyMethod.VANILLA):
    saliencies = {}

    output = types_map[type](model).generate_saliency(input, target)
    saliencies[MapType.ORIGINAL] = output
    saliencies[MapType.INPUT] = input


    pos_map = output.clamp(min = 0)
    pos_map = pos_map / pos_map.max()
    saliencies[MapType.POSITIVE] = pos_map

    neg_map = output.clamp(max = 0)
    neg_map *= -1
    neg_map = neg_map / neg_map.max()
    saliencies[MapType.NEGATIVE] = neg_map

    abs_map = output.abs()
    abs_map = abs_map / abs_map.max()
    saliencies[MapType.ABSOLUTE] = abs_map

    return saliencies
