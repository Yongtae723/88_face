import pandas as pd
from scipy.spatial.transform import Rotation
import numpy as np


x_columns_names = []
y_columns_names = []
z_columns_names = []
columns_names = []
for point in range(468):
    x_columns_names.append(f"x_{point}")
    y_columns_names.append(f"y_{point}")
    z_columns_names.append(f"z_{point}")
    
columns_names = x_columns_names + y_columns_names + z_columns_names

#%%
def rotation_landmark(dataframe, max_degree = 5 , mode = "train"):
    random_degree = np.random.randn(1,3) * max_degree
    rot = Rotation.from_euler("xyz", random_degree, degrees=True)
    positions = np.stack([
            dataframe[x_columns_names].values -.5 ,
            dataframe[y_columns_names].values-.5 , 
            dataframe[z_columns_names].values] , 1).astype(np.float16)
    if mode == "train":
        positions = rot.apply(positions)
    return positions.flatten()

    
    
# %%
