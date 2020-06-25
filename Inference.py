import pandas as pd
import numpy as np
import pickle
import time
import multiprocessing
from tqdm import notebook
import ast
import sys
import itertools

import torch
import torch.nn as nn 
from torchvision import transforms, models

###local scripts
from Custom_dataloader_geozones import MapDataset, Rescale, CenterCrop, Normalize, ToTensor

class CoordinatesFromLSH():
    """Retrieve the coordinates from """
    def __init__( self, lsh, features ):
        self.lsh = lsh
        self.features = features
        # self.Coordinates = list(range(len(self.features)))
    
    def get_similar_item_image_coordinates(self, i_features):
        try:
            # print(i_features)
            response = self.lsh.query(self.features[i_features].flatten(), 
                            num_results=1, distance_func='hamming')
            # self.Coordinates[i_features] = 
            # print('Coordinates: [{}/{}]'.format(i_features+1, len(self.features)), end="\r")
            return ast.literal_eval(response[0][0][1])
        except IndexError as error:
            print('Coordinate not found index: {}'.format(i_features))
            return [0, 0, 0, 0]

    
    def get_coordinates_multiprocessing(self):
        paramlist = list(range(len(self.features)))
        #Generate processes equal to the number of cores
        print('Getting coordinates...')
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

        # for _ in notebook.tqdm(pool.map_async(self.get_similar_item_image_coordinates, paramlist).get(), 
        #                         total=len(paramlist)):
        #     pass
        
        ##printing the progress bar but slower
        # for _ in notebook.tqdm(pool.imap_unordered(self.get_similar_item_image_coordinates, paramlist), 
        #                         total=len(paramlist)):
        #     pass

        Coordinates = list(notebook.tqdm(pool.map_async(
            self.get_similar_item_image_coordinates, paramlist).get(), total=len(paramlist)))

        # pool.map( self.get_similar_item_image_coordinates, paramlist )

                                    #map(), map_async().get(), imap(), imap_unordered().get
        pool.close()
        pool.join()
        return Coordinates


def main_inference(path_test_CSV, path_test_image_folder, path_CNN, path_LSH, path_submission_CSV, test_batch_size):
    
    startTime = time.time()
    
    ###======== Load CNN model ============
    ###create model dropping last FC layer
    model = models.resnet18(progress=True)
    model = nn.Sequential(*list(model.children())[:-1])
    ###model on GPU if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    ###load previously trained model parameters
    model.load_state_dict(torch.load(path_CNN, map_location=device))
    print("Model loaded from {}".format(path_CNN))
    # print(model)
    
    ###======== Load LSH model ============
    lsh = pickle.load(open(path_LSH,'rb'))
    print("LSH loaded from {}".format(path_LSH))
    
    ###======== Test set dataloader ============
    testDF = pd.read_csv(path_test_CSV)
    testDF.drop(columns=["Unnamed: 0"],inplace=True)
    testDF['Geo_zone'] = np.nan
    test_map_data = MapDataset(testDF, path_test_image_folder, transform= transforms.Compose([
                                                       Rescale(225),
                                                       CenterCrop((224,224)),
                                                       Normalize(alpha=0., beta=1.),
                                                       ToTensor(),
                                                   ]))
    test_loader = torch.utils.data.DataLoader(test_map_data,
                                              batch_size=test_batch_size,
                                              shuffle=False,
                                              num_workers=multiprocessing.cpu_count(),
                                              drop_last=False,
                                              pin_memory=True)
    print('Test data loader created, number of cores: {}'.format(multiprocessing.cpu_count()))

    ###======= Test set forward pass========
    total_step = len(test_loader)
    test_features = []
    for i_batch, sample_batch in enumerate(test_loader):
        with torch.no_grad():
            test_X = sample_batch['image'].float().to(device)
            outputs = model(test_X)
            test_features.extend(outputs.detach().cpu().numpy())
        print('Batch: [{}/{}]'.format(i_batch+1, total_step), end="\r")
    print('Test features created!')
    
    ###========= Retrieve coordinates from LSH domain ============

    cooFromLSH = CoordinatesFromLSH(lsh, test_features)
    Coordinates = cooFromLSH.get_coordinates_multiprocessing()
    print('Coordinates retrieved!')
    ###copy coordinates to test dataframe
    # coordinates_transposed = zip(Coordinates)
    testDF[["llcrnrlon", "llcrnrlat", "urcrnrlon", "urcrnrlat"]] = pd.DataFrame(Coordinates)
    print(testDF.head())
    print("Test set - finding coordinates, time: {}h {}min {}s".format((time.time()-startTime)//3600, 
                                                                      ((time.time()-startTime)%3600)//60,
                                                                      (time.time()-startTime)%60))
    testDF[["id","llcrnrlon","llcrnrlat","urcrnrlon","urcrnrlat"]].to_csv(path_submission_CSV)
    print('CSV file ready to submit, saved: {}'.format(path_submission_CSV))