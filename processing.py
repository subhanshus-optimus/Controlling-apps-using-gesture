def preprocessing(path):

    import cv2
    import numpy as np
    import os
    
    
    #    path = "./frames/silhouettes/"
    #path = "./completeDataset/"
    x=[]
    y=[]    
    allFiles = os.listdir(path)
    allFiles.sort()
    class_0 = 0
    class_1 = 0
    class_2=0
    count=0
    for file in allFiles:
        print(file)
        
        if (file.startswith("palm")):
            class_0 =class_0+1
            
            I = cv2.imread(path+file,0)
            I=cv2.resize(I,(128,128), interpolation = cv2.INTER_AREA)
            x.append(I)
            y.append(0)
            
        elif(file.startswith("fist")):
            class_1= class_1+1
            I = cv2.imread(path+file,0)
            I=cv2.resize(I,(128,128), interpolation = cv2.INTER_AREA)
            x.append(I)
            y.append(1)
        elif(file.startswith("noi")):
            class_2= class_2+1
            I = cv2.imread(path+file,0)
            I=cv2.resize(I,(128,128), interpolation = cv2.INTER_AREA)
            x.append(I)
            y.append(2)
        count+=1 
    print(count)
    print(path+allFiles[0])
    return np.asarray(x),np.asarray(y)
            
            