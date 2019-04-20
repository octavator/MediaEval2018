import seaborn as sns
import matplotlib.pyplot as plt

def drawDensityPlot(ground_truth):
  for label in ['short-term_memorability', 'long-term_memorability']:
    sns.distplot(ground_truth[label], hist=False, label=label)
  plt.xlabel('memorability')
  plt.title('memorability density plot')

# No obvious correlation, some vids are 1 in long term but O.7 - 1 prob bcs there are more raters (annotations)
# for short term so sample is bigger and tends more towards central tendency
def drawCorrelationPlot(ground_truth):
  (ground_truth[['short-term_memorability','long-term_memorability']]
    .plot(kind='scatter',
          x='short-term_memorability',
          y='long-term_memorability'))

  #Zoom in on cluttered top-right section
  (ground_truth[['short-term_memorability', 'long-term_memorability']]
    .plot(kind='scatter',
          x='short-term_memorability',
          y='long-term_memorability',
          xlim=[0.7, 1],
          ylim=[0.6, 1]))

def drawHistPlot(ground_truth):
  bins = 15
  figs = ground_truth[['short-term_memorability', 'long-term_memorability']].hist(bins=bins)

  for fig in figs[0]:
    fig.set_xlabel('memorability')
    fig.set_ylim(0, 1800)
    fig.set_xlim(0, 1)
    fig.set_xlabel('memorability')
    fig.set_ylim(0, 1800)
    fig.set_xlim(0, 1)

def drawPlots(ground_truth):
  drawDensityPlot(ground_truth)
  drawHistPlot(ground_truth)
  drawCorrelationPlot(ground_truth)
  plt.show()

  # I put None at the end of any cell where the last statement is related to a plot
  # Just removes some noise
  # Remove the None to see what happens (????)

