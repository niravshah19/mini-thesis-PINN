import pandas as pd
import numpy as np
import sys
sys.path.insert(0,'D:\Mini Thesis Codes')      ##Plotting & Kinametics file
import Plotting as plot
import os
import random
import tensorflow as tf
import Kinematics

##Return mesh and bc
def read_geometry():    
    
    ##define global variable for mesh and bounary_condition(bc)
    global mesh,bc 
    
    ##Reading the mesh information
    df_mesh = pd.read_csv(filepath_or_buffer = "Mesh.txt",skiprows=1, header=None) 
    mesh = df_mesh.to_numpy(dtype=np.float32)         
    #print(mesh)
    
    ##Reading the B.C
    df_bc = pd.read_csv(filepath_or_buffer = "Nirav_Boundary conditions.txt",
                    header=0,squeeze=True)#,usecols=['X','N: 464','Total force'])
    #print(df_bc) 

    df_bc = df_bc.values.tolist()
    b=[]
    for l in range(len(df_bc)):
        x=[]
        s = df_bc[l].split()
        for i in range(len(s)):
            x.append(float(s[i]))
        b.append(x)
    bc = np.array(b,dtype=np.float32)

###Get array of corodinates from array containig node id
def coordinates_from_nodeID(nodeID):
    coord = []
    for i,ID in enumerate(nodeID):
        ID = int(ID)-1
        coord.append(mesh[ID,1:])   
    coord = np.array(coord,dtype=np.float32)   
    #print(coord.shape)
    return coord 

##Check for direction of outward normal
def normal_direction(coord,n):
    r = 3           #Define search radius
    stop = True
    i = 0
    temp = None
    reverse = False
    while(stop): 
        point = coord+ pow(-1,i) * r * n
        #print(f"coord:{coord}    n:{n}      point;{point}")
        
        for l in range(X.shape[0]):
            dist = np.sqrt(pow((X[l,0]-point[0]),2) + pow((X[l,1]-point[1]),2))
            if(dist < r and X[l,0]!=coord[0] and X[l,1]!=coord[1]):
                temp = X[l,:]
                stop = False
                break
        if (type(temp)!=type(None)):
            #print(temp)
            break
        if(i == 1):
            break
        i += 1
    if(i!=1):
        reverse = True
    return reverse
    

##Calculate normal on the curve
def normal(coord):
    #print(coord)
    tangent_slope = np.gradient(coord[:,1],coord[:,0],edge_order=2)
    np.nan_to_num(tangent_slope,nan=1.0e+37, copy=False)
    for i,t in enumerate(tangent_slope):
        if(np.abs(t)<1.0e-6):
            tangent_slope[i]=0.0
    #print("\nTangent_slope")
    #print(tangent_slope)
    normal_slope = (-1) * tangent_slope**(-1)
    np.nan_to_num(normal_slope,copy=False)
    for i,n in enumerate(normal_slope):
        if(np.abs(n)<1.0e-6):
            normal_slope[i]=0.0
    normal = []
    for i,slope in enumerate(normal_slope):
        if(np.abs(slope)<1.0e-5):
            if(i!=0 and np.abs(normal_slope[i-1])<1.0e-5):
                normal.append(normal[-1])
            else:
                n = np.array([1,0])
                if(normal_direction(coord[i,:],n)):
                    n = np.array([-1,0]) 
                normal.append(n) 
        elif(np.abs(slope)>1e+5):
            if(i!=0 and  np.abs(normal_slope[i-1])>=1e+5):
                normal.append(normal[-1])
            else:
                n = np.array([0,1])
                if(normal_direction(coord[i,:],n)):
                     n = np.array([0,-1]) 
                normal.append(n)
        else:
            n = np.array([1 , slope])
            n = n/(np.linalg.norm(n))
            if(normal_direction(coord[i,:],n)):
                n = (-1)*n/(np.linalg.norm(n))
            normal.append(n)
    normal = np.array(normal,dtype=np.float32)
    #print(normal)
    
    return normal

##Return Dirichlet and Neumann boundary mesh
def training_data():  
    
    ##Define Gobal paramters
    global X_d,N_d,X_n,N_n,u_d,X,NODE_ID_D, NODE_ID_N, UD_List, D_List, ND_List

    X = mesh[:,1:]
    D_List = []
    UD_List = []

    ##Dirichlet Boundary Condition    
    df_DBottom = pd.read_csv(filepath_or_buffer="DBottom.txt",skiprows=1,header=None)
    DBottom = df_DBottom.to_numpy(dtype=np.float32)
    DBottom = DBottom.reshape(DBottom.shape[0]*DBottom.shape[1])
    DBottom = DBottom[np.logical_not(np.isnan(DBottom))]
    X_DBottom = coordinates_from_nodeID(DBottom)
    D_List.append(X_DBottom)

    df_DLeft = pd.read_csv(filepath_or_buffer="DLeft.txt",skiprows=1,header=None)
    DLeft = df_DLeft.to_numpy(dtype=np.float32)
    DLeft = DLeft.reshape(DLeft.shape[0]*DLeft.shape[1])
    DLeft = DLeft[np.logical_not(np.isnan(DLeft))]
    X_DLeft = coordinates_from_nodeID(DLeft)
    D_List.append(X_DLeft)

    df_DTop = pd.read_csv(filepath_or_buffer="DTop.txt",skiprows=1,header=None)
    DTop = df_DTop.to_numpy(dtype=np.float32)
    DTop = DTop.reshape((DTop.shape[0]*DTop.shape[1]))
    DTop = DTop[np.logical_not(np.isnan(DTop))]
    X_DTop = coordinates_from_nodeID(DTop)
    D_List.append(X_DTop)

    df_DRight = pd.read_csv(filepath_or_buffer="DRight.txt",skiprows=1,header=None)
    DRight = df_DRight.to_numpy(dtype=np.float32)
    DRight = DRight.reshape(DRight.shape[0]*DRight.shape[1])
    DRight = DRight[np.logical_not(np.isnan(DRight))]
    X_DRight = coordinates_from_nodeID(DRight)
    D_List.append(X_DRight)
    #print(D_List)

    NODE_ID_D = np.concatenate((DBottom,DLeft,DTop,DRight))
    X_d = np.vstack((X_DBottom,X_DLeft,X_DTop,X_DRight))
    
    #Calculate displacement at the boundary
    t = bc[step,1]
    #print(t)
    u = []
    for i in range(X_d.shape[0]): 
        if(X_d[i,0]==-52.5):
            u.append([-t,0])
        elif(X_d[i,0]==52.5):
            u.append([t,0])
        elif(X_d[i,1]==52.5):
            u.append([0,t])
        elif(X_d[i,1]==-52.5):
            u.append([0,-t])
    u_d = np.array(u,dtype=np.float32)
    #print(u_d)

    for i in range(len(D_List)):
        X_b = D_List[i]
        u_b = []
        for i in range(X_b.shape[0]): 
            if(X_b[i,0]==-52.5):
                u_b.append([-t,0])
            elif(X_b[i,0]==52.5):
                u_b.append([t,0])
            elif(X_b[i,1]==52.5):
                u_b.append([0,t])
            elif(X_b[i,1]==-52.5):
                u_b.append([0,-t])
        u_b = np.array(u_b,dtype=np.float32)
        UD_List.append(u_b)
    #print(UD_List)

    ##Neuman Boundary Condition
    
    df_NBLeft = pd.read_csv(filepath_or_buffer="NBLeft.txt",skiprows=1,header=None)
    NBLeft = df_NBLeft.to_numpy(dtype=np.float32)
    NBLeft = NBLeft.reshape(NBLeft.shape[0]*NBLeft.shape[1])
    NBLeft = NBLeft[np.logical_not(np.isnan(NBLeft))]
    X_NBLeft = coordinates_from_nodeID(NBLeft)

    df_NBRight = pd.read_csv(filepath_or_buffer="NBRight.txt",skiprows=1,header=None)
    NBRight = df_NBRight.to_numpy(dtype=np.float32)
    NBRight = NBRight.reshape(NBRight.shape[0]*NBRight.shape[1])
    NBRight = NBRight[np.logical_not(np.isnan(NBRight))]
    X_NBRight = coordinates_from_nodeID(NBRight)
    
    df_NTLeft = pd.read_csv(filepath_or_buffer="NTLeft.txt",skiprows=1,header=None)
    NTLeft = df_NTLeft.to_numpy(dtype=np.float32)
    NTLeft = NTLeft.reshape(NTLeft.shape[0]*NTLeft.shape[1])
    NTLeft = NTLeft[np.logical_not(np.isnan(NTLeft))]
    X_NTLeft = coordinates_from_nodeID(NTLeft)
    
    df_NTRight = pd.read_csv(filepath_or_buffer="NTRight.txt",skiprows=1,header=None)
    NTRight = df_NTRight.to_numpy(dtype=np.float32)
    NTRight = NTRight.reshape(NTRight.shape[0]*NTRight.shape[1])
    NTRight = NTRight[np.logical_not(np.isnan(NTRight))]
    X_NTRight = coordinates_from_nodeID(NTRight)

    NODE_ID_N =np.concatenate((NBLeft,NBRight,NTLeft,NTRight))
    X_n = np.vstack((X_NBLeft,X_NBRight,X_NTLeft,X_NTRight))
    #print(X_n)
    #print(N_List)

    ##Calcualtion for normal 
    N_DBottom = normal(X_DBottom)
    N_DLeft = normal(X_DLeft)
    N_DTop = normal(X_DTop)
    N_DRight = normal(X_DRight)
    N_d = np.vstack((N_DBottom,N_DLeft,N_DTop,N_DRight))
    ND_List = [N_DBottom,N_DLeft, N_DTop, N_DRight]
    #print(ND_List)

    N_NBLeft = normal(X_NBLeft)
    N_NBRight = normal(X_NBRight)
    N_NTLeft = normal(X_NTLeft)
    N_NTRight = normal(X_NTRight)
    N_n = np.vstack((N_NBLeft,N_NBRight,N_NTLeft,N_NTRight))
    #print(N_n)

    #For checking normals 
    import matplotlib.pyplot as plt
    fig1 = plt.figure(figsize=(8,8))
    ax1_1 = fig1.add_subplot(1,1,1)
    ax1_1.set_title(label="Geometry")
    ax1_1.set_xlabel(xlabel='X (mm)', fontsize='large')
    ax1_1.set_ylabel(ylabel='Y (mm)', fontsize='large')
    ax1_1.scatter(X[:,0],X[:,1],color='grey',label="Reference Configuration")

    for i,txt in enumerate(NODE_ID_N):
       ax1_1.annotate(int(txt),(X_n[i,0]+0.1,X_n[i,1]+0.1))
    ax1_1.scatter(X_n[:,0],X_n[:,1],color='green',label="Neumann Boundary")
    for i in range(len(N_n)):
        ax1_1.arrow(X_n[i,0],X_n[i,1],5*N_n[i,0],5*N_n[i,1],head_width=0.5, head_length=0.5)

    ax1_1.scatter(X_d[:,0],X_d[:,1],color='orange',label="Dirichlet Boundary")
    
    for i,txt in enumerate(NODE_ID_D):
       ax1_1.annotate(int(txt),(X_d[i,0]+0.1,X_d[i,1]+0.1))
    for i in range(len(N_d)):
        ax1_1.arrow(X_d[i,0],X_d[i,1],5*N_d[i,0],5*N_d[i,1],head_width=0.5, head_length=0.5)
    plt.legend()
    #plt.show()
    plt.savefig(fname = save_loc+"\Geomtery_Reference_config_normalized_cordinates")


##Calculate
##To create cliping boundary for the plots
def boundary():     
    b_mesh =  np.unique(np.concatenate((X_d,X_n),axis = 0),axis=0)
    x = np.unique(b_mesh[:,0])
    #print(x)
    m = []
    for i in range(x.shape[0]):
        count = 0
        y = []
        for j in range(b_mesh.shape[0]):
            if(x[i]==b_mesh[j,0] and b_mesh[j,1]>=0):
                count += 1
                y.append(b_mesh[j,1])

        if(len(y)>1):
            y.sort()
            
            if(i!=0):
                amin = y[0]
                amax = y[-1]
                last = m[-1][1]
                if(np.abs(last-amax)<=np.abs(last-amin)):
                    y.sort(reverse=True)

                   
        for k in range(len(y)):
            m.append([x[i],y[k]])
            
    for i in range(x.shape[0]):
        index = x.shape[0]-i-1
        count = 0
        y = []
        for j in range(b_mesh.shape[0]):
            if(x[index]==b_mesh[j,0] and b_mesh[j,1]<=0):
                count += 1
                y.append(b_mesh[j,1])
        
        if(len(y)>1):
            y.sort()
            amin = y[0]
            amax = y[-1]
            last = m[-1][1]
            if(np.abs(last-amax)<=np.abs(last-amin)):
                y.sort(reverse=True)

        for k in range(len(y)):
            m.append([x[index],y[k]])    
        
    m.append(m[0])
    m = np.array(m,dtype=np.float32)
    #print(m)
    return m

##1st Piola-Kirchoff stress
@tf.function
def First_Piola_Kirchhoff_Stress(Xi):               
    F = cm.Deformation_Gradient(Xi)
    invF =  tf.linalg.inv(F)
    invF_T = tf.transpose(invF,perm=[0,2,1])
    detF =  tf.linalg.det(F)
    P = []
    for i in range(len(detF)):
        P.append(2*(F[i] - pow(detF[i],-2) * invF_T[i]))             
    P = tf.convert_to_tensor(P, tf.float32)
    #print(P)
    return P

##Stress divergence
@tf.function
def divP(Xii):            
    with tf.GradientTape(persistent = True) as g:
        g.watch(Xii)
        P = First_Piola_Kirchhoff_Stress(Xii)
        P11 = P[:,0,0]
        P12 = P[:,0,1]
        P21 = P[:,1,0]
        P22 = P[:,1,1]
    P11_x = g.gradient(P11,Xii)      #dP11/dx  &  dP11/dy
    P12_x = g.gradient(P12,Xii)      #dP12/dx  &  dP12/dy
    P21_x = g.gradient(P21,Xii)      #dP21/dx  &  dP21/dy
    P22_x = g.gradient(P22,Xii)      #dP22/dx  &  dP22/dy
        
    div_P =  tf.concat([tf.reshape(P11_x[:,0] + P21_x[:,1],shape=[X.shape[0],1]),
                       tf.reshape(P12_x[:,0] + P22_x[:,1],shape=[X.shape[0],1])],axis = 1)
    #tf.print(div_P.shape)
    return div_P
 

##Plotting function
def plotter():

    ##Boundary in current state    
    X_b = boundary()
    u_b = cm.Deformation(X_b)
    x_b = X_b + u_b

    u = cm.Deformation(X)
    x = X + u
    
    ##Error in balance equation
    balance_error = tf.math.reduce_euclidean_norm(divP(X),axis=1)
    error2 = plot.Contour_Plotter(x_b)
    error2.set_grid_size(xlim=[-110,110], ylim=[-110,110])
    error2.set_title(title="Error in balance equation")
    error2.set_labels(x_label="X (mm)", y_label="Y (mm)")
    error2.plot(x[:,0], x[:,1], balance_error, levels=[0,0.5,1,2,4,6], save=True, loc= save_loc+'\loss_4')
    
tf.keras.backend.set_floatx("float32")  
np.random.seed(25)
tf.random.set_seed(25)
random.seed(25)
global save_loc, step, cm

step = 20
save_loc = "D:\Mini Thesis Codes\Weak Formulation\Plot"

os.chdir("D:\Mini Thesis Codes\Weak Formulation")

read_geometry()
training_data()

#Defining NN Architecture for Deformation
inputs = tf.keras.Input(shape=(2,),dtype=tf.float32)
d1 = tf.keras.layers.Dense(400,activation='tanh',use_bias=True,
                           kernel_initializer='glorot_normal',
                           bias_initializer='zeros')(inputs)
d2 = tf.keras.layers.Dense(400,activation='tanh',use_bias=True,
                           kernel_initializer='glorot_normal',
                           bias_initializer='zeros')(d1)
outputs = tf.keras.layers.Dense( 2 ,activation='linear',use_bias=True,
                                kernel_initializer='glorot_normal',
                                bias_initializer='zeros')(d2)
model = tf.keras.Model(inputs=inputs,outputs=outputs)
print("NN for Deformation")
model.summary()


model.load_weights('D:\Task 5\Task5_b\Images\Step 20\model.h5')
print("Weights have been loaded.")

cm = Kinematics.Kinematics(model)

plotter()
