##

import os
import pandas as pd

# Storing CSV metadata file Downloaded from ADNI into a data frame
df_ADNI = pd.read_csv('data/idaSearch_2_27_2018.csv')


# Creating a list of the image paths
saved_paths = []
for subdir, dirs, files in os.walk('/cortex/data/MRI/ADNI'):
    for file in files:
        path = os.path.join(subdir, file)
        if path.endswith('.nii'):   # [x for x in saved_paths if "nii" in x]
            saved_paths.append(path)
            #print path
#print(len(saved_paths))


# Creating the metaData data frame and saving Subject IDs and Labels
columns1 = ['Subject ID', 'Lab']
df_metaData = pd.DataFrame(columns=columns1)
df_metaData['Subject ID'] = df_ADNI['Subject ID']
df_metaData['Lab'] = df_ADNI['DX Group']

# Creating a data frame with Patients ID's from within the Image Paths
columns2 = ['Subject ID', 'Path']
df_paths = pd.DataFrame(columns=columns2)
df_paths['Path'] = saved_paths
df_split_path = pd.DataFrame([x.split('/') for x in saved_paths])
df_paths['Subject ID'] = df_split_path.iloc[:, 6]


# Merging the metaData and paths data frame to match between the Subject IDs and paths
df_metaData = pd.merge(df_metaData, df_paths, how='right', left_on=['Subject ID'], right_on=['Subject ID'])

# Deleting the subject ID column
del df_metaData['Subject ID']

# Saving the metaData data frame into a csv file
#df_metaData.to_csv('data/metadata_t.csv', sep='\t')
df_metaData.to_csv('data/metadata.csv')
print(df_metaData)



