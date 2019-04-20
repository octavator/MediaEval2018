import os
import numpy as np
import pandas as pd

def getInceptionV3(df):

  # ## Inception Features
  
  inception_path = './dev-set/dev-set/dev-set_features/InceptionV3/'
  values = []
  for video in df["video"]:
    video_name = video.strip(".webm")
    
    ## take only first frame // Todo : take first & last, or avg of them all etc.
    example_inception_path = inception_path + video_name + '-0.txt'
    inceptions_values = open(example_inception_path, 'r').read()
    sample_inception_dictionary = parse_inception_feature(inceptions_values)
    
    # Lets convert this to a vector with 1000 elements
    # print("should sum up to 1: ", sum(expand_inception_feature(sample_inception_dictionary)))
    value_to_save = expand_inception_feature(sample_inception_dictionary)
    values.append(value_to_save)

  # We will store the data in a dictionary, where they keys are the names of the files.
  # inception_features = pd.DataFrame(expanded_dict).T
  df["inception"] = values
  print(df.head(10))
  return []


# we will convert the keys using pathlib like this
#example_inception_path.with_suffix('.webm').name.replace('-0', '')

# Each of these files is a single line, containing mappings from indices to values -> function to parse this into a dictionary

def parse_inception_feature(dataPath):
    pairs = dataPath.strip().split(' ')
    pairs = [i.split(':') for i in pairs]
    return {int(k): float(v) for k, v in pairs}

def expand_inception_feature(d):
  # 1000 = size of inception feature vector
  feature = np.zeros(1000)
  for k, v in d.items():
    feature[k] = v
  return feature
