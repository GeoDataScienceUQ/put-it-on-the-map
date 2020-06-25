# import random
import numpy as np

def create_geo_zones(Lon, Lat, lon_dim=10, lat_dim=10):
    zones =[] 
    izone = 0
    for index in range (len(Lon)):
        zoneFound = False
        for zone in zones:
            if abs(zone[0]-Lon[index]) <= lon_dim and abs(zone[1]-Lat[index]) <= lat_dim:
                zoneFound = True
                break
        if not zoneFound:
            zones.append([Lon[index], Lat[index], izone])
            izone += 1
        # if (index+1) % 1000 == 0:
        #     print('Creating zones: Row [{}/{}]'.format(index+1, len(Lon)), end="\r")
    return zones

def classify_geographic_zones(zones, row, lon_dim=10, lat_dim=10):
    # min_dist = np.inf
    for zone in zones:
        if abs(zone[0]-row['llcrnrlon']) <= lon_dim and abs(zone[1]-row['llcrnrlat']) <= lat_dim:
            return zone[2]
        # tmp_min_dist = abs(zone[0]-row['llcrnrlon']) + abs(zone[1]-row['llcrnrlat'])
        # if tmp_min_dist < min_dist:
        #     min_dist = tmp_min_dist
        #     zone_to_return = zone[2]
    # return(zone_to_return)
    print('Out of Zone')
    return 0

def previousclassify_geographic_zones(row):
    if row['llcrnrlon']<-100. and row['llcrnrlat']>30.:
        return(0) #AmNW
    elif row['llcrnrlon']<-30. and row['llcrnrlat']>30.:
        return(1) #AmNE
    elif row['llcrnrlon']<-30. and row['llcrnrlat']>0.:
        return(2) #AmC
    elif row['llcrnrlon']<-30. and row['llcrnrlat']<=0.:
        return(3) #AmS
    elif row['llcrnrlon']<90. and row['llcrnrlat']>45.:
        return(4) #EuN
    elif row['llcrnrlon']<60. and row['llcrnrlat']>20.:
        return(5) #Med
    elif row['llcrnrlon']<60. and row['llcrnrlat']<=20.:
        return(6) #Af
    elif row['llcrnrlat']>35.:
        return(7) #AsE
    elif row['llcrnrlat']>5.:
        return(8) #AsS
    elif row['llcrnrlat']>-20.:
        return(9) #Ind
    else:
        return(10) #Aus

