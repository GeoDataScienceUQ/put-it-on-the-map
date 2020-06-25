import torch
import torch.nn as nn 


def outputSize(in_size, kernel_size, stride, padding):
    output = int((in_size - kernel_size + 2*(padding)) / stride) + 1
    return(output)


class SimpleCNN(nn.Module):   
    def __init__(self):
        super(SimpleCNN, self).__init__()

        init_ft_length=61
        nfeatures=3
        conv1 = [8, 5, 1, 2]
        max1 = [2, 2, 1]
        conv2 = [8, 3, 1, 1]
        max2 = [2, 2, 1]
        conv3 = [8, 3, 1, 1]

        self.cnn_layers = nn.Sequential(
            nn.BatchNorm2d(num_features=nfeatures),
            nn.Conv2d(in_channels=nfeatures, out_channels=conv1[0], kernel_size=conv1[1], stride=conv1[2], padding=conv1[3]),
            nn.BatchNorm2d(num_features=conv1[0]),
            #ReLU(inplace=True),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=max1[0], stride=max1[1], padding=max1[2]),
            nn.Conv2d(in_channels=conv1[0], out_channels=conv2[0], kernel_size=conv2[1], stride=conv2[2], padding=conv2[3]),
            nn.BatchNorm2d(num_features=conv2[0]),
            #ReLU(inplace=True),
            nn.PReLU(),
            nn.Dropout(p=0.1),
            nn.MaxPool2d(kernel_size=max2[0], stride=max2[1], padding=max2[2]),
            nn.Conv2d(in_channels=conv2[0], out_channels=conv3[0], kernel_size=conv3[1], stride=conv3[2], padding=conv3[3]),
            nn.BatchNorm2d(num_features=conv3[0]),
            #ReLU(inplace=True),
            nn.PReLU(),
        )
        
        ft_length=outputSize(
                        outputSize(
                            outputSize(
                                outputSize(
                                    outputSize(init_ft_length, conv1[1], conv1[2], conv1[3]), 
                                    max1[0], max1[1], max2[2]),
                                conv2[1], conv2[2], conv2[3]),
                            max2[0], max2[1], max2[2]),
                        conv3[1], conv3[2], conv3[3])
        self.linear_layers = nn.Sequential(
            nn.Linear((conv3[0]*ft_length*ft_length), (conv3[0]*ft_length*ft_length)//2),
            nn.Linear((conv3[0]*ft_length*ft_length)//2, 4)
        )
        
        #self.softmax = nn.LogSoftmax(dim=1)

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
#         x = self.softmax(x)
        return x

# custom loss function
class Interval_Loss(torch.nn.Module):
    
    def __init__(self):
        super(Interval_Loss,self).__init__()
        
#     def forward(self,x,y):
#         y_shape = y.size()[1]
#         x_added_dim = x.unsqueeze(1)
#         x_stacked_along_dimension1 = x_added_dim.repeat(1,NUM_WORDS,1)
#         diff = torch.sum((y - x_stacked_along_dimension1)**2,2)
#         totloss = torch.sum(torch.sum(torch.sum(diff)))
#         return totloss
    def forward(self, output, target):
        if torch.mean((output - target)**2) < 2500:
            return (torch.mean((output - target)**2))
        else:
            return (torch.mean((output - target)**2)+10000)

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss