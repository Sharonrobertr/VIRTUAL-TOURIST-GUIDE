#======================== IMPORT PACKAGES ===========================

import numpy as np
import matplotlib.pyplot as plt 
from tkinter.filedialog import askopenfilename
import cv2
import numpy as np
import matplotlib.image as mpimg
import os
import seaborn as sns
import base64
import cv2
import streamlit as st
# ================ Background image ===

st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:36px;">{"Virtual Tour"}</h1>', unsafe_allow_html=True)


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('1.jpg')



#====================== READ A INPUT IMAGE =========================

fileneme = st.file_uploader("Upload a image")

if fileneme is None:
    
    st.text("Kindly upload input image....")

else:

    #====================== READ A INPUT IMAGE =========================
    
    
    # filename = askopenfilename()
    img = mpimg.imread(fileneme)
    plt.imshow(img)
    plt.title('Original Image') 
    plt.axis ('off')
    plt.show()
    
    st.image(img,caption="Original Image")
    
    
# #====================== READ A INPUT IMAGE =========================

# filename = askopenfilename()
# img = mpimg.imread(filename)
# plt.imshow(img)
# plt.title('Original Image')
# plt.axis ('off')
# plt.show()


    #============================ PREPROCESS =================================
    
    #==== RESIZE IMAGE ====
    
    resized_image = cv2.resize(img,(224,224))
    img_resize_orig = cv2.resize(img,((50, 50)))
    
    fig = plt.figure()
    plt.title('RESIZED IMAGE')
    plt.imshow(resized_image)
    plt.axis ('off')
    plt.show()
    
    #==== GRAYSCALE IMAGE ====
    
    
    
    SPV = np.shape(img)
    
    try:            
        gray1 = cv2.cvtColor(img_resize_orig, cv2.COLOR_BGR2GRAY)
        
    except:
        gray1 = img_resize_orig
       
    fig = plt.figure()
    plt.title('GRAY SCALE IMAGE')
    plt.imshow(gray1)
    plt.axis ('off')
    plt.show()
    
    
    #============================ 3.FEATURE EXTRACTION ====================
    
    
    # === MEAN MEDIAN VARIANCE ===
    
    mean_val = np.mean(gray1)
    median_val = np.median(gray1)
    var_val = np.var(gray1)
    Test_features = [mean_val,median_val,var_val]
    
    
    print()
    print("----------------------------------------------")
    print("FEATURE EXTRACTION --> MEAN, VARIANCE, MEDIAN ")
    print("----------------------------------------------")
    print()
    print("1. Mean Value     =", mean_val)
    print()
    print("2. Median Value   =", median_val)
    print()
    print("1. Variance Value =", var_val)
    
    # ==== LBP =========
    
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt
       
          
    def find_pixel(imgg, center, x, y):
        new_value = 0
        try:
            if imgg[x][y] >= center:
                new_value = 1
        except:
            pass
        return new_value
       
    # Function for calculating LBP
    def lbp_calculated_pixel(imgg, x, y):
        center = imgg[x][y]
        val_ar = []
        val_ar.append(find_pixel(imgg, center, x-1, y-1))
        val_ar.append(find_pixel(imgg, center, x-1, y))
        val_ar.append(find_pixel(imgg, center, x-1, y + 1))
        val_ar.append(find_pixel(imgg, center, x, y + 1))
        val_ar.append(find_pixel(imgg, center, x + 1, y + 1))
        val_ar.append(find_pixel(imgg, center, x + 1, y))
        val_ar.append(find_pixel(imgg, center, x + 1, y-1))
        val_ar.append(find_pixel(imgg, center, x, y-1))
        power_value = [1, 2, 4, 8, 16, 32, 64, 128]
        val = 0
        for i in range(len(val_ar)):
            val += val_ar[i] * power_value[i]
        return val
       
       
    height, width, _ = img.shape
       
    img_gray_conv = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
       
    img_lbp = np.zeros((height, width),np.uint8)
       
    for i in range(0, height):
        for j in range(0, width):
            img_lbp[i, j] = lbp_calculated_pixel(img_gray_conv, i, j)
    
    plt.imshow(img_lbp, cmap ="gray")
    plt.title("LBP")
    plt.show()
    
    
    #============================ 5. IMAGE SPLITTING ===========================
    
    
    #==== TRAIN DATA FEATURES ====
    
    import pickle
    
    with open('dot.pickle', 'rb') as f:
        dot1 = pickle.load(f)
      
    
    import pickle
    with open('labels.pickle', 'rb') as f:
        labels1 = pickle.load(f) 
    
    
    from sklearn.model_selection import train_test_split
    
    x_train, x_test, y_train, y_test = train_test_split(dot1,labels1,test_size = 0.2, random_state = 101)
    
    print("---------------------------------")
    print("Image Splitting")
    print("---------------------------------")
    print()
    print("1. Total Number of images =", len(dot1))
    print()
    print("2. Total Number of Test  =", len(x_test))
    print()
    print("3. Total Number of Train =", len(x_train))    
    
    
    # ====================== CLASSIFICATION ================
    
    # ==== DIMENSION  ==
    
    from keras.utils import to_categorical
    from tensorflow.keras.models import Sequential
    
    
    y_train1=np.array(y_train)
    y_test1=np.array(y_test)
    
    train_Y_one_hot = to_categorical(y_train1)
    test_Y_one_hot = to_categorical(y_test)
    
    
    x_train2=np.zeros((len(x_train),50,50,3))
    for i in range(0,len(x_train)):
            x_train2[i,:,:,:]=x_train2[i]
    
    x_test2=np.zeros((len(x_test),50,50,3))
    for i in range(0,len(x_test)):
            x_test2[i,:,:,:]=x_test2[i]
    
    
    
    # ==================== CNN - 2D =================================
    
    from keras.layers import Dense, Conv2D
    from keras.layers import Flatten
    from keras.layers import MaxPooling2D
    
    from keras.layers import Dropout
    
     
    # initialize the model
    model=Sequential()
    
    
    #CNN layes 
    model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
    model.add(MaxPooling2D(pool_size=2))
    
    model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    
    model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    
    model.add(Dropout(0.2))
    model.add(Flatten())
    
    model.add(Dense(500,activation="relu"))
    
    model.add(Dropout(0.2))
     
    model.add(Dense(1,activation="softmax"))
    
    #summary the model 
    model.summary()
    
    #compile the model 
    model.compile(loss='binary_crossentropy', optimizer='adam')
    
    print("--------------------------------------------------")
    print("           CNN ---> 2D                  ")
    print("-------------------------------------------------")
    print()
    
    history = model.fit(x_train2,y_train1,batch_size=50, epochs=5)
    print("--------------------------------------------------")
    print(" Performance Analysis - CNN -2D")
    print("-------------------------------------------------")
    print()
    loss=history.history['loss']
    loss=max(loss)
    loss=abs(loss)
    acc_cnn=100-loss
    print()
    
    print("1) Accuracy     =", acc_cnn ,'%')
    print()
    print("2) Error Rate   =", loss ,'%')
    print()
    
    
    #=============================== PREDICTION =================================
    
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
    
    print()
    print("-----------------------")
    print("       PREDICTION      ")
    print("-----------------------")
    print()
    
    
    Total_length = len(data1) + len(data2) + len(data3) + len(data4) + len(data5) + len(data6) + len(data7) + len(data8) + len(data9) + len(data10) + len(data11) + len(data12) + len(data13) + len(data14) + len(data15) + len(data16) + len(data17) + len(data18) + len(data19) + len(data20) + len(data21) + len(data22) + len(data23) + len(data24)
    
    temp_data1  = []
    for ijk in range(0,3745):
        # print(ijk)
        temp_data = int(np.mean(dot1[ijk]) == np.mean(gray1))
        temp_data1.append(temp_data)
    
    temp_data1 =np.array(temp_data1)
    
    zz = np.where(temp_data1==1)
    
    if labels1[zz[0][0]] == 0:
        print('-----------------------------------')
        print(' IDENTIFIED = Ajanta Caves')
        print('-----------------------------------')
        res="Aganta Caves"
    
    elif labels1[zz[0][0]] == 1:
        print('----------------------------------')
        print(' IDENTIFIED = Alari Darwaza')
        print('----------------------------------')
        res="alai_darwaza"
        
    elif labels1[zz[0][0]] == 2:
        print('----------------------------------')
        print(' IDENTIFIED = Alari Minar')
        print('----------------------------------')
        res="alai_minar"
            
        
    elif labels1[zz[0][0]] == 3:
        print('----------------------------------')
        print(' IDENTIFIED = Basilica of bom jesus')
        print('----------------------------------')    
        res="basilica_of_bom_jesus"
        
        
    elif labels1[zz[0][0]] == 4:
        print('----------------------------------')
        print(' IDENTIFIED = Charar-E- Sharif')
        print('----------------------------------')
        res="Charar-E- Sharif"        
    
    elif labels1[zz[0][0]] == 5:
        print('----------------------------------')
        print(' IDENTIFIED = charminar')
        print('----------------------------------') 
        res="charminar"
        
        
    
    elif labels1[zz[0][0]] == 6:
        print('----------------------------------')
        print(' IDENTIFIED = Chhota_Imambara')
        print('----------------------------------')        
        res="Chhota_Imambara"
        
         
        
    elif labels1[zz[0][0]] == 7:
        print('----------------------------------')
        print(' IDENTIFIED = Ellora Caves')
        print('----------------------------------')   
        res="Ellora Caves"     
        
        
    elif labels1[zz[0][0]] == 8:
        print('----------------------------------')
        print(' IDENTIFIED = Fatehpur Sikri')
        print('----------------------------------')
        res="Fatehpur Sikri"        
        
        
    elif labels1[zz[0][0]] == 9:
        print('----------------------------------')
        print(' IDENTIFIED = Gateway of India')
        print('----------------------------------')
        res="Gateway of India"       
        
        
        
        
        
    elif labels1[zz[0][0]] == 10:
        print('----------------------------------')
        print(' IDENTIFIED = golden temple')
        print('----------------------------------')
        res="golden temple"         
        
        
    elif labels1[zz[0][0]] == 11:
        print('----------------------------------')
        print(' IDENTIFIED = hawa mahal pics')
        print('----------------------------------')
        res="hawa mahal pics"        
        
    elif labels1[zz[0][0]] == 12:
        print('----------------------------------')
        print(' IDENTIFIED = Humayun_s Tomb')
        print('----------------------------------') 
        res="Humayun_s Tomb"       
        
    elif labels1[zz[0][0]] == 13:
        print('----------------------------------')
        print(' IDENTIFIED = India gate pics')
        print('----------------------------------')
        res="India gate pics"            
        
        
    elif labels1[zz[0][0]] == 14:
        print('----------------------------------')
        print(' IDENTIFIED = iron_pillar')
        print('----------------------------------')
        res="iron_pillar"        
        
    elif labels1[zz[0][0]] == 15:
        print('----------------------------------')
        print(' IDENTIFIED = jamali_kamali_tomb')
        print('----------------------------------')
        res="jamali_kamali_tomb"      
        
        
    elif labels1[zz[0][0]] == 16:
        print('----------------------------------')
        print(' IDENTIFIED = Khajuraho')
        print('----------------------------------')
        res="Khajuraho"        
        
    elif labels1[zz[0][0]] == 17:
        print('----------------------------------')
        print(' IDENTIFIED = lotus_temple')
        print('----------------------------------')
        res="lotus_temple"        
        
        
    elif labels1[zz[0][0]] == 18:
        print('----------------------------------')
        print(' IDENTIFIED = mysore_palace')
        print('----------------------------------')
        res="mysore_palace"        
        
    elif labels1[zz[0][0]] == 19:
        print('----------------------------------')
        print(' IDENTIFIED = qutub_minar')
        print('----------------------------------')
        res="qutub_minar"      
        
        
        
    elif labels1[zz[0][0]] == 20:
        print('----------------------------------')
        print(' IDENTIFIED = Sun Temple Konark')
        print('----------------------------------')
        res="Sun Temple Konark"        
        
    elif labels1[zz[0][0]] == 21:
        print('----------------------------------')
        print(' IDENTIFIED = tajmahal')
        print('----------------------------------')
        res="tajmahal"       
        
        
    elif labels1[zz[0][0]] == 22:
        print('----------------------------------')
        print(' IDENTIFIED = tanjavur temple')
        print('----------------------------------')
        res="tanjavur temple"        
        
    elif labels1[zz[0][0]] == 23:
        print('----------------------------------')
        print(' IDENTIFIED = victoria memorial')
        print('----------------------------------')
        res="victoria memorial"       
    
    final = "Identified " + res
    st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:26px;">{final}</h1>', unsafe_allow_html=True)
     
    
     
    
    ###
    
    # import pandas as pd
    
    # data=pd.read_excel("Excel Data.xlsx")
        
    # x1=data['Name']
    
    # for i in range(0,len(data)):
    #     if x1[i]==res:
    #         idx=i
    #     # else:
    #     #     idx=7
        
    # history = data['History']
    
    # history=history[idx]    
        
    # st.write('----------------------------------')
    # st.write(' History',history)
    # st.write('----------------------------------')    
    
    
    
    # for i in range(0,len(data)):
    #     if x1[i]==res:
    #         idx=i
    #     # else:
    #     #     idx=7
        
    # cul = data['Cultural Details']
    
    # cul=cul[idx]    
        
    # st.write('----------------------------------')
    # st.write(' Cultural Details',cul)
    # st.write('----------------------------------')    


    
    col1, col2 = st.columns(2)
    
    with col1:
            
        aa = st.button("History")
        
        if aa:
            
            import pandas as pd
            
            data=pd.read_excel("Excel Data.xlsx")
                
            x1=data['Name']
            
            for i in range(0,len(data)):
                if x1[i]==res:
                    idx=i
                # else:
                #     idx=7
                
            history = data['History']
            
            history=history[idx]    
                
            st.write('----------------------------------')
            st.write(' History',history)
            st.write('----------------------------------')    
            # st.success('Successfully Registered !!!')
        # else:
            
            # st.write('Registeration Failed !!!')     
    
    with col2:
            
        aa = st.button("About")
        
        if aa:
            
            import pandas as pd
            
            data=pd.read_excel("Excel Data.xlsx")
                
            x1=data['Name']
            
            
            for i in range(0,len(data)):
                if x1[i]==res:
                    idx=i
                # else:
                #     idx=7
                
            cul = data['Cultural Details']
            
            cul=cul[idx]    
                
            st.write('----------------------------------')
            st.write(' Cultural Details',cul)
            st.write('----------------------------------')    






