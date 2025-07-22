"""Okay, the main purpose of this file is to put the code we've been using to calculate the new loss value"""

# Import loads
from AutoEncoders import GRAEAnchor
import numpy as np

# Setting things up
PATH = "demonstration_utils/"
seeds = np.load(PATH + "seeds/emb.npy")
anchors = np.load(PATH + "seeds/anchors.npy")
labels = np.load(PATH + "seeds/labels.npy")
train_emb = np.load(PATH + "seeds/train_emb.npy") 
trainA = np.load(PATH + "seeds/trainA.npy")
trainB = np.load(PATH + "seeds/trainB.npy")
testA = np.load(PATH + "seeds/testA.npy")
testB = np.load(PATH + "seeds/testB.npy")

train_labels = np.load(PATH + "seeds/train_labels.npy")
test_labels = np.load(PATH + "seeds/test_labels.npy")

# Train the autoencoders
AutoEncA = GRAEAnchor(lam=100, anchor_lam=100, relax=False, n_components = 2) #NOTE: try changing anchor_lam to other values! 
AutoEncA.fit(trainA, train_emb, anchors)

AutoEncB = GRAEAnchor(lam=100, anchor_lam=100, relax=False, n_components = 2)
AutoEncB.fit(trainB, train_emb, anchors)



#The code bellow works with a list passed in.  But I don't have it to the point that it can subtract two lists from each other.
def small_scale(percent, threshold,list):
    number=len(list)
    target=round(number*(percent/100))
    needed_index=[]
    global i
    i=0
    for x in range(len(list)):
        if list[x]>=threshold:
            print(list[x], x)
            needed_index.append(x)
            i+=1
    if i<target:
        print("Not enough values are in the percentage disered, please select another one and try again")
        print(f"Cureently there are {i} values in the threshold of {threshold}")
        return
    elif i==target:
        print("Continuing")
    else:
        print("Too many values fit the percent you gave.  Please lower the amount you want in the percent or raise the percent.")
        return
    print(needed_index)

    #Okay we've found the values at this point, now it's time to remove them.
    for x in range(len(needed_index)):
        t=needed_index.pop()
        list.pop(t)
    return(list)




a_list=[5,88,10,6,21,14,34,87,9,]
print(a_list)
a_list.sort()
print(a_list)