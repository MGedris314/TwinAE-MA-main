"""Notes regardining this file:  In the demonstration book, we use these a couple of times.
Small scale is still a work in progress.  We have the full idea in the demo notebook and it works.  I'm trying to speed things up by putting it in 
a seperate file to just have the function call.  I'm still figuring out what parts I need to have to bring it over.  I think I'm close, I'll take a look 
into it a bit later.

both mod anchor calls work, and they work (as far as I can tell) the way they need to.  The best result, and most effective, comes when you make the call
similar to this:
mod_anchors_B(Anchor_list=[mod_anchors_A(Anchor_list=[anchors],remove_points=points_A)],remove_points=points_B)
Returns the anchors sorted numerically (don't know if this will do anything to effect stuff) which you can then pass back in to other functions as updated
anchor points.
"""
import numpy as np

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
        print(f"Cureently there are {i} values in the threshold of {threshold}%")
        return
    elif i==target:
       print("")
    else:
        print("Too many values fit the percent you gave.  Please lower the amount you want in the percent or raise the percent.")
        return
    print(needed_index)

    #Okay we've found the values at this point, now it's time to remove them.
    for x in range(len(needed_index)):
        t=needed_index.pop()
        list.pop(t)
    return(list)

def mod_anchors_A(Anchor_list,remove_points):
    #This program is expecting both lists to be numpy lists and to have two dimensions.
    sort_anchors=np.sort(Anchor_list[0])
    sort_remove=np.sort(remove_points)
    for x in range(len(sort_anchors)):
        for y in range(len(sort_remove)):
            if sort_anchors[x,0]>=sort_remove[y]:
                sort_anchors[x,0]=sort_anchors[x,0]-1
    return(np.sort(sort_anchors))

def mod_anchors_B(Anchor_list,remove_points):
    anchors=Anchor_list[0]
    #This program is expecting both lists to be numpy lists and to have two dimensions.
    sort_anchors = anchors[anchors[:, 1].argsort()]
    sort_remove=np.sort(remove_points)
    for x in range(len(sort_anchors)):
        for y in range(len(sort_remove)):
            if sort_anchors[x,1]>=sort_remove[y]:
                sort_anchors[x,1]=sort_anchors[x,1]-1
    return(sort_anchors)
