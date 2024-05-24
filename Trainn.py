import os 
import cv2
import numpy as np
import matplotlib.image as mpimg

# === 

data1 = os.listdir('Data/Ajanta Caves/')

data2 = os.listdir('Data/alai_darwaza/')

data3 = os.listdir('Data/alai_minar/')

data4 = os.listdir('Data/basilica_of_bom_jesus/')

data5 = os.listdir('Data/Charar-E- Sharif/')


# ===  

data6 = os.listdir('Data/charminar/')

data7 = os.listdir('Data/Chhota_Imambara/')

data8 = os.listdir('Data/Ellora Caves/')

data9 = os.listdir('Data/Fatehpur Sikri/')

data10 = os.listdir('Data/Gateway of India/')

# === 

data11 = os.listdir('Data/golden temple/')

data12 = os.listdir('Data/hawa mahal pics/')

data13 = os.listdir('Data/Humayun_s Tomb/')

data14 = os.listdir('Data/India gate pics/')

data15 = os.listdir('Data/iron_pillar/')


# === 


data16 = os.listdir('Data/jamali_kamali_tomb/')

data17 = os.listdir('Data/Khajuraho/')

data18 = os.listdir('Data/lotus_temple/')

data19 = os.listdir('Data/mysore_palace/')

data20 = os.listdir('Data/qutub_minar/')


# === 


data21 = os.listdir('Data/Sun Temple Konark/')

data22 = os.listdir('Data/tajmahal/')

data23 = os.listdir('Data/tanjavur temple/')

data24 = os.listdir('Data/victoria memorial/')




dot1= []
labels1 = []
for img in data1:
    try:
        # print(img)
        img_1 = mpimg.imread('Data/Ajanta Caves/' + "/" + img)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(0)
    except:
        None
        
for img in data2:
    try:
        img_2 = mpimg.imread('Data/alai_darwaza/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(1)
    except:
        None

for img in data3:
    try:
        img_2 = mpimg.imread('Data/alai_minar'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(2)
    except:
        None
        
        
for img in data4:
    try:
        img_2 = mpimg.imread('Data/basilica_of_bom_jesus/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(3)
    except:
        None


        
for img in data5:
    try:
        img_2 = mpimg.imread('Data/Charar-E- Sharif/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(4)
    except:
        None

############        
for img in data6:
    try:
        img_2 = mpimg.imread('Data/charminar/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(5)
    except:
        None
        
for img in data7:
    try:
        img_2 = mpimg.imread('Data/Chhota_Imambara/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(6)
    except:
        None

for img in data8:
    try:
        img_2 = mpimg.imread('Data/Ellora Caves/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(7)
    except:
        None

for img in data9:
    try:
        img_2 = mpimg.imread('Data/Fatehpur Sikri/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(8)
    except:
        None

        
for img in data10:
    try:
        img_2 = mpimg.imread('Data/Gateway of India/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(9)
    except:
        None

# =======
        
for img in data11:
    try:
        img_2 = mpimg.imread('Data/golden temple/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(10)
    except:
        None

for img in data12:
    try:
        img_2 = mpimg.imread('Data/hawa mahal pics/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(11)
    except:
        None

for img in data13:
    try:
        img_2 = mpimg.imread('Data/Humayun_s Tomb/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(12)
    except:
        None

for img in data14:
    try:
        img_2 = mpimg.imread('Data/India gate pics/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(13)
    except:
        None

for img in data15:
    try:
        img_2 = mpimg.imread('Data/iron_pillar/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(14)
    except:
        None

# ===========
        
for img in data16:
    try:
        img_2 = mpimg.imread('Data/jamali_kamali_tomb/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(15)
    except:
        None

for img in data17:
    try:
        img_2 = mpimg.imread('Data/Khajuraho/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(16)
    except:
        None

for img in data18:
    try:
        img_2 = mpimg.imread('Data/lotus_temple/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(17)
    except:
        None

for img in data19:
    try:
        img_2 = mpimg.imread('Data/mysore_palace/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(18)
    except:
        None

for img in data20:
    try:
        img_2 = mpimg.imread('Data/qutub_minar/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(19)
    except:
        None

# ========
        
for img in data21:
    try:
        img_2 = mpimg.imread('Data/Sun Temple Konark/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(20)
    except:
        None

for img in data22:
    try:
        img_2 = mpimg.imread('Data/tajmahal/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(21)
    except:
        None

for img in data23:
    try:
        img_2 = mpimg.imread('Data/tanjavur temple/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(22)
    except:
        None


# =========
        
for img in data24:
    try:
        img_2 = mpimg.imread('Data/victoria memorial/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(23)
    except:
        None

        

import pickle
with open('dot.pickle', 'wb') as f:
    pickle.dump(dot1, f)
    
with open('labels.pickle', 'wb') as f:
    pickle.dump(labels1, f)        
        
