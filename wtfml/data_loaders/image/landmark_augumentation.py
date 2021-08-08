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
def rotation_landmark(dataframe, max_degree = 10):
    random_degree = np.random.randn(1,3) * max_degree
    rot = Rotation.from_euler("xyz", random_degree, degrees=True)
    for point in range(468):
        position = dataframe[[f"x_{point}",f"y_{point}",f"z_{point}"]] -.5
        dataframe[[f"x_{point}",f"y_{point}",f"z_{point}"]] = rot.apply(position) + .5
    return dataframe[columns_names].list()
        
    
    
# %%