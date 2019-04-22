import pandas as pd
from collections import Counter
import pyprind
from string import punctuation
from keras.preprocessing.text import Tokenizer

def getVideosCaptions():
  video_captions = list(open('../dev-set/dev-set/dev-set_video-captions.txt', 'r'))
  video_captions = [i.split('\t') for i in video_captions]
  video_captions = [[a, b.strip()] for a, b in video_captions]
  video_captions = pd.DataFrame(video_captions, columns=['video', 'caption'])
  return video_captions

def getPredAndCaptions(ground_truth, video_captions):
  ground_truth_captions = video_captions.merge(ground_truth, left_on='video', right_on='video')
  return ground_truth_captions

def countWordsOccur(df_cap):
  counts = Counter()
  pbar = pyprind.ProgBar(len(df_cap['caption']), title='Counting word occurrences')
  for i, cap in enumerate(df_cap['caption']):
      # replace punctuations with space, convert words to lower case 
      text = ''.join([c if c not in punctuation else ' ' for c in cap]).lower()
      df_cap.loc[i,'caption'] = text
      pbar.update()
      counts.update(text.split())


  tokenizer = Tokenizer(num_words=len(counts))
  tokenizer.fit_on_texts(list(df_cap.caption.values)) #fit a list of captions to the tokenizer
  
  one_hot_res = tokenizer.texts_to_matrix(list(df_cap.caption.values),mode='binary')
  return one_hot_res

def printCaptionsSamples(ground_truth_captions):
  ### Most Memorable Short Term Videos
  captions_number = 20

  top_short_term_captions = (ground_truth_captions
                            .sort_values('short-term_memorability',
                                          ascending=False)['caption'])

  print('Most memorable short term videos:')
  for text in top_short_term_captions[:captions_number]:
    print(text)

  # use fancy indexing to reverse array
  ### Least memorable short term videos
  print('Least memorable short term videos:')
  for text in (top_short_term_captions)[::-1][:captions_number]:
    print(text)

  ### Most Memorable Long Term Captions

  top_long_term_captions = (ground_truth_captions
                            .sort_values('long-term_memorability',
                                        ascending=False)['caption'])
  print('Most memorable long term videos:')

  for text in top_long_term_captions[:captions_number]:
    print(text)

  ### Least Memorable Short Term Captions
  # use fancy indexing to reverse array
  print('Least memorable long term videos:')
  for text in (top_long_term_captions)[::-1][:captions_number]:
    print(text)