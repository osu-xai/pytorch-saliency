import torch
from saliency.saliency import Saliency
import numpy as np
from scipy.ndimage import label
import torchvision


class Perturbation_2Saliency(Saliency):

    def __init__(self, model):
        super(Perturbation_2Saliency, self).__init__(model)


    def generate_saliency(self, input, target):

        #self.model.zero_grad()

        output = self.model(input)

        #index: 0 layer: HP
        # index: 1 layer: Ship
        # index: 2 layer: Small Towers
        # index: 3 layer: Big Towers
        # index: 4 layer: Small Cities
        # index: 5 layer: Big Cities
        # index: 6 layer: Friend
        # index: 7 layer: Enemy


        #return (input.grad.clone()[0] * input)
        input2 = input.clone()
        image = np.zeros((40, 40))
        input2 = input2.view(40, 40, 8)

        #logical or of the input images to get the original image
        #only do or over small big cities and towers to get proper coordinates of cities and towers
        #otherwise it will include enemy ship too if enemy channel is included which will cause object to
        #be merged with tower or city object
        for i in range(2,6):
            if i!= 0:
                image = np.logical_or(image, input2[:, :, i].numpy())*1

        #get the number of objects in the image
        labeled_array, num_objects = label(image)

        indices = []
        for i in range(num_objects):
            indices.append(np.argwhere(labeled_array == i+1))



        #special stuff for ship channel
        ship_image = input2[:, :, 1].clone().numpy()
        ship_image[19][20] = 0
        ship_image[20][20] = 0
        ship_image[21][20] = 0
        ship_array, ships = label(ship_image)
        ship_indices = []
        for i in range(ships):
            ship_indices.append(np.argwhere(ship_array == i+1))
        #self.generate_file(ship_array, 'mod_ship_array')

        hp_array, hp = label(input2[:, :, 0].numpy())
        hp_indices = []
        for i in range(hp):
            hp_indices.append(np.argwhere(hp_array == i+1))


        values = torch.zeros(40*40*5)
        values = values.view(40, 40, 5)
        input2 = input.clone()
        input2 = input2.view(40, 40, 8)

        # index: 0 layer: HP
        # index: 1 layer: Ship
        # index: 2 layer: Small Towers
        # index: 3 layer: Big Towers
        # index: 4 layer: Small Cities
        # index: 5 layer: Big Cities
        # index: 6 layer: Friend
        # index: 7 layer: Enemy

        #output layers:
        #0 HP
        #1 agent
        #2 size
        #3 type
        # friend/enemy

        #here i refers to the output salient layers
        #print('num_objects', num_objects)

        for i in range(5):
            if i==0:# output HP
                for j in range(hp):
                    for k in range(hp_indices[j].shape[0]):
                        x = hp_indices[j][k][0]
                        y = hp_indices[j][k][1]
                        temp = 0.3*input2[:, :, 0][x][y]
                        input2[:, :, 0][x][y] += temp
                    perturbed_output = self.model(input2.view(1, 12800))
                    gradient = (perturbed_output - output)/temp
                    input2 = input.clone()
                    input2 = input2.view(40, 40, 8)

                    for k in range(hp_indices[j].shape[0]):
                        x = hp_indices[j][k][0]
                        y = hp_indices[j][k][1]
                        values[:, :, 0][x][y] = gradient[:, target]
            elif i==1:#output agent
                for j in range(ships):
                    for k in range(ship_indices[j].shape[0]):
                        x = ship_indices[j][k][0]
                        y = ship_indices[j][k][1]
                        if input2[:, :, 1][x][y] == 1:
                            input2[:, :, 1][x][y] = 0
                    perturbed_output = self.model(input2.view(1, 12800))
                    gradient = (perturbed_output - output)
                    input2 = input.clone()
                    input2 = input2.view(40, 40, 8)

                    for k in range(ship_indices[j].shape[0]):
                        x = ship_indices[j][k][0]
                        y = ship_indices[j][k][1]
                        values[:, :, 1][x][y] = gradient[:, target]
            elif i==2: #output size
                #print('in i== 2')
                for l in range(2, 6):
                    #print('layer: ',l)
                    for j in range(num_objects):
                    #    print('object: ',j)
                        s = 0
                        for k in range(indices[j].shape[0]):
                            x = indices[j][k][0]
                            y = indices[j][k][1]
                            # print('x: '+str(x)+' y: '+str(y))
                            # print('Value of input: '+str(input2[:, :, i][x][y]))
                    #        print(input2[:, :, l][x][y])
                            if l == 2 or l==4: #small tower/city
                                if input2[:, :, l][x][y] == 1:
                                    s = 1
                                    input2[:, :, l][x][y] = 0
                                    input2[:, :, l+1][x][y] = 1
                            else: #big tower/city
                                if input2[:, :, l ][x][y] == 1:
                                    s = 1
                                    input2[:, :, l][x][y] = 0
                                    input2[:, :, l-1][x][y] = 1

                        perturbed_output = self.model(input2.view(1, 12800))
                        gradient = (perturbed_output - output)
                        #print(saliency)
                        input2 = input.clone()
                        input2 = input2.view(40, 40, 8)
                        if s==1:
                            for k in range(indices[j].shape[0]):
                                x = indices[j][k][0]
                                y = indices[j][k][1]
                                values[:, :, 2][x][y] = gradient[:, target]
                #print(saliency[0][target])
            elif i==3: #output type
                for l in range(2, 6):
                    for j in range(num_objects):
                        s = 0
                        for k in range(indices[j].shape[0]):
                            x = indices[j][k][0]
                            y = indices[j][k][1]
                            # print('x: '+str(x)+' y: '+str(y))
                            # print('Value of input: '+str(input2[:, :, i][x][y]))
                            if l == 2 or l == 3: #small tower/city
                                if input2[:, :, l][x][y] == 1:
                                    s = 1
                                    input2[:, :, l][x][y] = 0
                                    input2[:, :, l+2][x][y] = 1
                            else: #big tower/city
                                if input2[:, :, l ][x][y] == 1:
                                    s = 1
                                    input2[:, :, l][x][y] = 0
                                    input2[:, :, l-2][x][y] = 1

                        perturbed_output = self.model(input2.view(1, 12800))
                        gradient = (perturbed_output - output)
                        #print(saliency)
                        input2 = input.clone()
                        input2 = input2.view(40, 40, 8)
                        if s==1:
                            for k in range(indices[j].shape[0]):
                                x = indices[j][k][0]
                                y = indices[j][k][1]
                                values[:, :, 3][x][y] = gradient[:, target]

            else:# output frenemy
                for l in range(6, 8):
                    for j in range(num_objects):
                        s = 0
                        for k in range(indices[j].shape[0]):
                            x = indices[j][k][0]
                            y = indices[j][k][1]

                            if l == 6:
                                if input2[:, :, l][x][y] == 1:
                                    s = 1
                                    input2[:, :, l][x][y] = 0
                                    input2[:, :, l+1][x][y] = 1
                            else:
                                if input2[:, :, l][x][y] == 1:
                                    s = 1
                                    input2[:, :, l][x][y] = 0
                                    input2[:, :, l-1][x][y] = 1
                        perturbed_output = self.model(input2.view(1, 12800))
                        gradient = (perturbed_output - output)
                        #print(s)
                        input2 = input.clone()
                        input2 = input2.view(40, 40, 8)
                        if s==1:
                            for k in range(indices[j].shape[0]):
                                x = indices[j][k][0]
                                y = indices[j][k][1]
                                values[:, :, 4][x][y] = gradient[:, target]






                    #for l in range(6):
                    #    torchvision.utils.save_image(saliency[:, :, l], "Image perturbed: "+str(i) + "/" + "object perturbed: "+str(j)+ "/" + str(l) + ".png", normalize=True)




        return (values.view(1, 40*40*5))

    def generate_file(self, array, name):
        f = open(str(name)+'.txt', 'w')
        f.write('\n\n\n')

        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                f.write(str(array[i,j]))
            f.write('\n')
        f.close()
