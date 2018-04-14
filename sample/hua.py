import numpy as np
import string
import matplotlib.pyplot as plt
import matplotlib as mpl

f = open("siftkeypoints.txt") 
line = f.readline()
y = []
while line:
    y.append(string.atoi(line[0:len(line)-1]))                
    line = f.readline()
f.close()
print "pass1"
list=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
for num in y:
 if num<4000:
  index=num/100
  list[index]=list[index]+1
 else:
  list[40]=list[40]+1
print "pass2" 
width = 100
ind = np.linspace(100,4200,41)
fig = plt.figure(1)

ax  = fig.add_subplot(111)
ax.bar(ind-width,list,width,color='green')

    # Set the ticks on x-axis

ax.set_xticks(ind)
plt.show()