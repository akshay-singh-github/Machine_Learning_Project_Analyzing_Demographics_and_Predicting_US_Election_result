import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

raw_data = {'Feature_Set': ['Feature Set 1', 'Feature Set 2', 'Feature Set 3 (Best)'],
        'SVM': [91.1717495987, 93.5794542536, 95.1845906902]}
df = pd.DataFrame(raw_data, columns = ['Feature_Set', 'SVM'])


# Create the general blog and the "subplots" i.e. the bars
f, ax1 = plt.subplots(1, figsize=(10,5))

# Set the bar width
bar_width = 0.25

# positions of the left bar-boundaries
bar_l = [i+1 for i in range(len(df['SVM']))] 

# positions of the x-axis ticks (center of the bars as bar labels)
tick_pos = [i+(bar_width/2) for i in bar_l] 

# Create a bar plot, in position bar_1
barlist = ax1.bar(bar_l,
                  # using the pre_score data
                  df['SVM'], 
                  # set the width
                  width=bar_width,
                  alpha=0.5)
    
barlist[0].set_color('r')
barlist[1].set_color('b')
barlist[2].set_color('g')
# set the x ticks with names
plt.xticks(tick_pos, df['Feature_Set'])

# Set the label and legends
ax1.set_ylabel("Accuracy of SVM")
ax1.set_xlabel("Different Feature Sets")
ax1.set_title('SVM')

rects = ax1.patches

# Now make some labels
labels = [91.17, 93.57, 95.18]

for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width()/2, height - 10, label, ha='center', va='bottom')


# Set a buffer around the edge
plt.xlim([min(tick_pos)-bar_width, max(tick_pos)+bar_width])

plt.show()
