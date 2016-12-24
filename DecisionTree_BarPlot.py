import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


raw_data = {'Feature_Set': ['Feature Set 1', 'Feature Set 2', 'Feature Set 3 (Best)'],
        'Decision_Tree': [85.5537720706, 90.6902086677, 94.3820224719]}
df = pd.DataFrame(raw_data, columns = ['Feature_Set', 'Decision_Tree'])


# Create the general blog and the "subplots" i.e. the bars
f, ax1 = plt.subplots(1, figsize=(10,5))

# Set the bar width
bar_width = 0.25

# positions of the left bar-boundaries
bar_l = [i+1 for i in range(len(df['Decision_Tree']))] 

# positions of the x-axis ticks (center of the bars as bar labels)
tick_pos = [i+(bar_width/2) for i in bar_l] 

# Create a bar plot, in position bar_1

barlist = ax1.bar(bar_l,
                  # using the pre_score data
                  df['Decision_Tree'], 
                  # set the width
                  width=bar_width,
                  # with the label pre score
                  #label='Decision_Tree', 
                  # with alpha 0.5
                  alpha=0.5)
    
barlist[0].set_color('r')
barlist[1].set_color('b')
barlist[2].set_color('g')

    
# set the x ticks with names
plt.xticks(tick_pos, df['Feature_Set'])

# Set the label and legends
ax1.set_ylabel("Accuracy of Decision Tree")
ax1.set_xlabel("Different Feature Sets")
ax1.set_title('Decision_Tree')
#plt.legend(loc='upper left')

rects = ax1.patches

# Now make some labels
labels = [85.55, 90.69, 94.38]

for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width()/2, height - 10, label, ha='center', va='bottom')


# Set a buffer around the edge
plt.xlim([min(tick_pos)-bar_width, max(tick_pos)+bar_width])

plt.show()
