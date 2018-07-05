import torch
from saliency.saliency import Saliency
import numpy as np
from scipy.ndimage import label
import torchvision
from torch.autograd import Variable
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import copy


class LimeSaliency(Saliency):

    def __init__(self, model):
        super(LimeSaliency, self).__init__(model)


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
        input3 = input.clone()

        #logical or of the input images to get the original image
        #only do or over small big cities and towers to get proper coordinates of cities and towers
        #otherwise it will include enemy ship too if enemy channel is included which will cause object to
        #be merged with tower or city object
        for i in range(2,6):
                image = np.logical_or(image, input2[:, :, i].numpy())*1

        #get the number of objects in the image
        labeled_array, num_objects = label(image)

        indices = []
        for i in range(num_objects):
            indices.append(np.argwhere(labeled_array == i+1))
        #self.generate_file(labeled_array, 'labeled_array')

        # print('object 1\n')
        # print(indices[0])
        #print('object 2\n')
        #print(indices[1])
        #print('object 3\n')
        #print(indices[2])
        #print('object 4\n')
        #print(indices[3])
        #print('object 5\n')
        #print(indices[4])

        #hp
        hp_array, hp = label(input2[:, :, 0].numpy())
        hp_indices = []
        for i in range(hp):
            hp_indices.append(np.argwhere(hp_array == i+1))
        #self.generate_file(hp_array, 'hp_array')

        #ship
        #remove agent because we don't want to perturb that
        ship_image = input2[:, :, 1].clone().numpy()
        ship_image[19][20] = 0
        ship_image[20][20] = 0
        ship_image[21][20] = 0
        ship_array, ships = label(ship_image)
        # print('ships ', ships )
        ship_indices = []
        for i in range(ships):
            ship_indices.append(np.argwhere(ship_array == i+1))
        #self.generate_file(ship_array, 'ship_array')



        values = torch.zeros(40*40*5)
        values = values.view(40, 40, 5)
        input2 = input.clone()
        input2 = input2.view(40, 40, 8)
        features = self.generate_features(input2)
        #print(features)
        outputs = output[:, target].data.numpy()

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
        #4 friend/enemy

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
                    feature = self.generate_features(input2)
                    features = np.concatenate((features, feature))
                    outputs = np.concatenate((outputs, perturbed_output[:, target].data.numpy()))
                    input2 = input.clone()
                    input2 = input2.view(40, 40, 8)

            elif i==1:#output agent
                for j in range(ships):
                    for k in range(ship_indices[j].shape[0]):
                        x = ship_indices[j][k][0]
                        y = ship_indices[j][k][1]
                        if input2[:, :, 1][x][y] == 1:
                            input2[:, :, 1][x][y] = 0
                    perturbed_output = self.model(input2.view(1, 12800))
                    feature = self.generate_features(input2)
                    features = np.concatenate((features, feature))
                    outputs = np.concatenate((outputs, perturbed_output[:, target].data.numpy()))
                    input2 = input.clone()
                    input2 = input2.view(40, 40, 8)

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



                        #print(saliency)

                        if s==1:
                            perturbed_output = self.model(input2.view(1, 12800))
                            feature = self.generate_features(input2)
                            features = np.concatenate((features, feature))
                            outputs = np.concatenate((outputs, perturbed_output[:, target].data.numpy()))
                        input2 = input.clone()
                        input2 = input2.view(40, 40, 8)
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


                        #print(saliency)

                        if s==1:
                            perturbed_output = self.model(input2.view(1, 12800))
                            feature = self.generate_features(input2)
                            features = np.concatenate((features, feature))
                            outputs = np.concatenate((outputs, perturbed_output[:, target].data.numpy()))
                        input2 = input.clone()
                        input2 = input2.view(40, 40, 8)

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

                        if s==1:
                            perturbed_output = self.model(input2.view(1, 12800))
                            feature = self.generate_features(input2)
                            features = np.concatenate((features, feature))
                            outputs = np.concatenate((outputs, perturbed_output[:, target].data.numpy()))
                        input2 = input.clone()
                        input2 = input2.view(40, 40, 8)


        #print(features)
        #print(outputs)
        linear_model = LinearRegressionModel(21, 1)
        linear_model.train()
        criterion = nn.L1Loss()
        optimiser = torch.optim.SGD(linear_model.parameters(), lr = 0.01)
        epochs = 5000
        for epoch in range(epochs):
            inputs = Variable(torch.from_numpy(features).float())
            labels = Variable(torch.from_numpy(outputs))
            optimiser.zero_grad()
            pred = linear_model.forward(inputs)
            loss = criterion(pred, labels)
            loss.backward()
            optimiser.step()
            #print('epoch {}, loss {}'.format(epoch,loss.item()))
        #train_loss = eval_net(features, outputs, linear_model)
        # print('train_loss: %.5f ' %
        #       (train_loss))

        # weights_ =  []
        # for name, param in linear_model.parameters() :
        #     weights_.append(param.data.numpy())
        #new_model = copy.deepcopy(linear_model)
        weights = linear_model.linear.weight.clone()
        weights = weights.data.numpy()
        #print(weights)



        # print('weights')
        # print(weights_)
        # weights_ = np.asarray(weights_)

        values = self.plot_saliency(weights, input3)

        return (values.view(1, 40*40*5))


#0 HP
#1 enemy ship
#2 size
#3 type
#4 friend/enemy

    def generate_features(self, input):
        hp_array, hp = label(input[:, :, 0].numpy())
        hp_indices = []
        for i in range(hp):
            hp_indices.append(np.argwhere(hp_array == i+1))
        image = np.zeros((40, 40))
        for i in range(2,6):
                image = np.logical_or(image, input[:, :, i].numpy())*1
        feature = np.zeros((1, 21))
        #put hp of agent. agent will always be object 3 - 1
        feature[0][20] = input[:, :, 0][hp_indices[2][0][0]][hp_indices[2][0][1]]
        #print(feature[0][20])
        #self.generate_file(hp_array, 'feature_hp_array')
        #zero out the agent
        ship_image = input[:, :, 1].clone().numpy()
        ship_image[19][20] = 0
        ship_image[20][20] = 0
        ship_image[21][20] = 0
        ship_image, _ = label(ship_image)
        #self.generate_file(ship_array, 'mod_ship_array')
        #self.generate_file(image, 'image')
        #self.generate_file(ship_image, 'ship_image')
        counter = 0
        #slicing the hp_array quadrant vise
        for i in range(2):
            for j in range(2):
                #array = hp_array[0 + 20*i :19 + 20*i, 0 + 20*j :19 + 20*j]
                #labeled_array, num_objects = label(array)
                indices = np.argwhere(image[0 + 20*i :20 + 20*i, 0 + 20*j :20 + 20*j] > 0)
                # print(indices)
                # print('\n\n')
                # print(indices[0][0])
                # print(indices[0][1])
                x = indices[0][0] + 20*i
                y = indices[0][1] + 20*j
                # print('x ',x)
                # print('y ',y)
                #first feature will be HP
                feature[0][counter + 0] = input[:, :, 0][x][y]
                #second feature will be checking prescence of enemy ship
                _, objs = label(ship_image[0 + 20*i :20 + 20*i, 0 + 20*j :20 + 20*j])
                feature[0][counter + 1] = (1 if objs>0 else 0)
                #third feature check size 1 if big 0 if small
                feature[0][counter + 2] = (1 if input[:, :, 3][x][y] == 1
                                            or input[:, :, 5][x][y] == 1 else 0)
                #fourth feature will check type. 1 if city 0 if tower
                feature[0][counter + 3] = (1 if input[:, :, 4][x][y] == 1
                                            or input[:, :, 5][x][y] == 1 else 0)
                #fifth feature will check frie\nd/enemy. 1 if friend 0 if enemy
                feature[0][counter + 4] = (1 if input[:, :, 6][x][y] == 1 else 0)


                counter += 5
        return feature

    def generate_file(self, array, name):
        f = open(str(name)+'.txt', 'w')
        f.write('\n\n\n')

        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                f.write(str(array[i,j]))
            f.write('\n')
        f.close()

    def plot_saliency(self, feature, input3):
        print('in plot ')
        values = torch.zeros(40*40*5)
        values = values.view(40, 40, 5)
        input3 = input3.view(40,40,8)
        feature = torch.from_numpy(feature).float()
        print('feature: ')
        print(feature)
        #this will give you dimensions of only objects
        image = np.zeros((40, 40))
        for i in range(2,6):
                image = np.logical_or(image, input3[:, :, i].numpy())*1
        labeled_array, num_objects = label(image)
        self.generate_file(image, 'image')
        ship_image = input3[:, :, 1].clone().numpy()
        ship_image[19][20] = 0
        ship_image[20][20] = 0
        ship_image[21][20] = 0
        ship_image, _ = label(ship_image)
        self.generate_file(ship_image, 'ship_image')
        counter = 0
        #slicing the hp_array quadrant vise
        for i in range(2):
            for j in range(2):
                #array = hp_array[0 + 20*i :19 + 20*i, 0 + 20*j :19 + 20*j]
                #labeled_array, num_objects = label(array)
                indices = np.argwhere(image[0 + 20*i :20 + 20*i, 0 + 20*j :20 + 20*j] > 0)
                #second feature will be checking prescence of enemy ship
                print('i ',i)
                print('j ',j)
                print(indices)
                print('\n\n')
                # print(indices[0][0])
                # print(indices[0][1])
                #first take care of HP
                for k in range(indices.shape[0]):
                    x = indices[k][0] + 20*i
                    y = indices[k][1] + 20*j
                    print('x ',x)
                    print('y ',y)
                    #first feature will be HP
                    values[:, :, 0][x][y] = feature[0][counter + 0]
                    values[:, :, 2][x][y] = feature[0][counter + 2]
                    values[:, :, 3][x][y] = feature[0][counter + 3]
                    values[:, :, 4][x][y] = feature[0][counter + 4]

                #second feature will be checking prescence of enemy ship
                _, objs = label(ship_image[0 + 20*i :20 + 20*i, 0 + 20*j :20 + 20*j])
                enemytank_indices = np.argwhere(ship_image[0 + 20*i :20 + 20*i, 0 + 20*j :20 + 20*j]>0)
                if objs > 0:
                    print('objs ')
                    print(objs)
                    for k in range(enemytank_indices.shape[0]):
                        x = enemytank_indices[k][0] + 20*i
                        y = enemytank_indices[k][1] + 20*j
                        print('x ',x)
                        print('y ',y)
                        values[:, :, 1][x][y] = feature[0][counter + 1]
                        values[:, :, 0][x][y] = feature[0][counter + 0]


                # #third feature check size 1 if big 0 if small
                # feature[0][counter + 2] = (1 if input[:, :, 3][x][y] == 1
                #                             or input[:, :, 5][x][y] == 1 else 0)
                # #fourth feature will check type. 1 if city 0 if tower
                # feature[0][counter + 3] = (1 if input[:, :, 4][x][y] == 1
                #                             or input[:, :, 5][x][y] == 1 else 0)
                # #fifth feature will check friend/enemy. 1 if friend 0 if enemy
                # feature[0][counter + 4] = (1 if input[:, :, 6][x][y] == 1 else 0)


                counter += 5

        values[:, :, 0][19][20] = feature[0][20]
        values[:, :, 0][20][20] = feature[0][20]
        values[:, :, 0][21][20] = feature[0][20]
        return values


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out

def eval_net(x, y, model):
    correct = 0
    total = 0
    total_loss = 0
    model.eval() # set model to evaluation mode
    criterion = nn.L1Loss()
    for i, (x1, y1) in enumerate(zip(x, y)):
        inputs = Variable(torch.from_numpy(x1).float())
        labels = Variable(torch.from_numpy(y1))
        pred = model.forward(inputs)
        total += labels.size(0)
        #correct += (pred == labels.data).sum()
        loss = criterion(pred, labels)
        total_loss += loss.item()
        #total_loss += loss.item()
    model.train() # set model back to train mode
    return total_loss / total
