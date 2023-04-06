import sys
sys.path.append('../')

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import networkit as nk
from tqdm import tqdm
import networkx as nx

from benji_girgs import generation, utils
import geopandas as gpd

from geopy import distance

print('done importing')

fn = './flightlist_20190601_20190630.csv'

df = pd.read_csv(fn)

df = df.loc[:, ['aircraft_uid', 'origin', 'destination', 'day']]
df.shape
df = df.dropna()

airports = list(np.union1d(df.origin.unique(), df.destination.unique()))

adf = pd.read_csv('./airport-codes_csv.csv')

missing_airports = set([a for a in airports if not a in adf.ident.values])
present_airports = [a for a in airports if a in adf.ident.values]
airports = present_airports
a, b = df.origin.apply(lambda x: x in missing_airports), df.destination.apply(lambda x: x in missing_airports)


df = df[~(a|b)]
df['airport_pair'] = df.apply(lambda x: set([x.origin, x.destination]), axis=1)
df = df[df['airport_pair'].apply(len) == 2]
df['airport_pair'] = df.airport_pair.apply(frozenset)


adf_mini = adf.set_index('ident')
adf_mini = adf_mini.loc[airports]

# reversed s.t. latitude then longitude
adf_mini['coordinates'] = adf_mini.coordinates.apply(lambda x: tuple(reversed([float(y) for y in x.split(', ')])))

adf_mini['lat'] = adf_mini['coordinates'].apply(lambda x: x[0])
adf_mini['long'] = adf_mini['coordinates'].apply(lambda x: x[1])


print('about to squaring')
n = len(adf_mini)
square = pd.DataFrame(
    np.zeros((n, n)),
    index=adf_mini.index, columns=adf_mini.index
)

def get_distance(col):
    end_latlon = adf_mini.loc[col.name, 'coordinates']
    series = adf_mini.coordinates.apply(distance.distance, args=(end_latlon,), ellipsoid='WGS-84')
    return series.apply(lambda x: x.km)

distances = square.progress_apply(get_distance, axis=1).T

distances.to_csv('airport_distances.csv')