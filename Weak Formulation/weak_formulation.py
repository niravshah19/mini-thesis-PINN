##Solve Weak form without using Shape function

'''
Global notations
mesh     : Contains mesh information i.e mesh[i,j,k] = mesh[nodes,x,y]
bc       : Boundary deisplacement bc[i,j] = bc[step,displacment,force]
step     : Define the step number from displacemnt table 
save_loc : Location where all the graphs will be saved
X        : Position vector in Reference Configuration
X_d      : Position vector for dirichlet boundary
N_d      : Unit Outward Normal vector for dirichlet boundary 
u_d      : Dirichlet vector for dirichlet boundary
X_n      : Position vector for Neumann boundary
N_n      : Unit Outward Normal vector for Neumann boundary 
NODE_ID_D: Array containing node ID's of dirichlet boundary
NODE_ID_N: Array containing node ID's of neumann boundary
UD_List  : List containing displacment of dirichlet boundary points
D_List   : List containing reference position of the dirichlet boundary points
ND_List  : List containing outward unit normal vectoer for dirichlet boundary points
'''

import pandas as pd
import numpy as np
import sys
sys.path.insert(0,'D:\Mini Thesis Codes')    
import Plotting as plot
import os
import random
import tensorflow as tf
import time
import Kinametics

global cm
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

##Read and calculate area of elements
def Triangle_area():
    area = []
    global elements

    ##Reading element and nodes information
    df_element = pd.read_csv(filepath_or_buffer = "Element.txt",skiprows=1, header=None) 
    elements = df_element.to_numpy(dtype=np.float32)         
    #print(elements)

    for i in range(elements.shape[0]):
        nodes = elements[i,1:].astype(np.int32)
        x = []
        y = []
        for j,n in enumerate(nodes):
            x.append(mesh[n-1,1])
            y.append(mesh[n-1,2])
            #print(f"Node ID:{n}     meshID:{mesh[n-1,0]}")
        A = np.array([[1, x[0], y[0]],
                      [1, x[1], y[1]],
                      [1, x[2], y[2]]])
        detA = np.linalg.det(A)
        if(detA<0):
            nodes = np.flip(nodes)
            print(i)
            elements[i,1:] = nodes
            detA = np.abs(detA)
        area.append(0.5*detA)
    area = np.array(area,dtype=np.float32)
    elements = np.hstack((elements,np.reshape(area,newshape=(area.shape[0],1))))

##Strain Energy
@tf.function
def Strain_Energy():
    energy = 0
    C = cm.Right_Cauchy_Green(X)
    detC = tf.linalg.det(C)
    C33 = tf.math.reciprocal(detC)
    trC = tf.linalg.trace(C)
    Ic = tf.math.add(trC,C33)
    W = Ic - 3
    #tf.print(W)
    for i in range(elements.shape[0]):
        nodes = elements[i,1:4]
        A = elements[i,4]
        el_energy = 0
        for j in range(nodes.shape[0]):
            n = int(nodes[j])
            el_energy += W[n-1]
        energy += (A/3) * el_energy
    #tf.print(energy)

    return energy

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

##External Work 
@tf.function
def External_Work():
    external_work = 0
    for i in range(len(D_List)):
        #ub = tf.convert_to_tensor(UD_List[i], tf.float32)
        Xb = D_List[i]
        ub = cm.Deformation(Xb)
        ub = tf.reshape(ub,shape=[ub.shape[0],ub.shape[1],1])
        Nb = tf.convert_to_tensor(ND_List[i], tf.float32)
        Nb = tf.reshape(Nb,shape=[Nb.shape[0],Nb.shape[1],1])
        P = First_Piola_Kirchhoff_Stress(Xb)
        T = tf.matmul(P,Nb)
        We = tf.matmul(T,ub, transpose_a=True)
        X1 = Xb[:,0]
        X2 = Xb[:,1]
        for j in range(len(We)-1):
            dX1 = X1[j+1] - X1[j]
            dX2 = X2[j+1] - X2[j]
            w = np.sqrt(pow(dX1,2) + pow(dX2,2))
            external_work += (w/2) * (We[j]+We[j+1])
    #tf.print(external_work)
    return external_work 

##Loss function
@tf.function
def loss_fn():

    ##Energy loss
    loss_1 = tf.math.reduce_mean(tf.square(Strain_Energy() - External_Work()))

    ##Dirichlet boundary loss term
    u_pred = cm.Deformation(X_d)
    loss_2 = tf.math.reduce_mean(tf.square(u_d[:,0]-u_pred[:,0])) + tf.math.reduce_mean(tf.square(u_d[:,1]-u_pred[:,1]))

    ###Neumann boundary loss
    #N = tf.reshape(N_n,shape=[N_n.shape[0],N_n.shape[1],1])
    #P_n = First_Piola_Kirchhoff_Stress(X_n)
    #f_n = tf.matmul(P_n,N)
    #loss_3 = tf.math.reduce_mean(tf.square(tf.math.reduce_euclidean_norm(f_n,axis=1)))
    ##print(loss_3)

    loss = loss_1 + 5000*loss_2 #+ 100*loss_3

    return loss

##function factory
@tf.function
def function_factory(model, optimizer):

    # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
    with tf.GradientTape() as g:
        loss_value = loss_fn()
    grads = g.gradient(loss_value,model.trainable_variables)
    optimizer.apply_gradients(zip(grads,model.trainable_variables))
    return loss_value

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


##Plotting function
def plotter(step,Training_loss,external_work,SE,save_loc):

    ##Boundary in current state    
    X_b = boundary()
    u_b = cm.Deformation(X_b)
    x_b = X_b + u_b

    # Plotting current configuration

    u = cm.Deformation(X)
    x = X + u
    sc = plot.Scatter_Plotter()
    #sc.set_figure_size((10,8))
    sc.set_grid_size(xlim=[-110,110], ylim=[-110,110])
    sc.set_title(title="Geometry")
    sc.set_labels(x_label='X (mm)', y_label='Y (mm)')
    sc.plot(X=[X[:,0],X_d[:,0],X_n[:,0],x[:,0]],Y=[X[:,1],X_d[:,1],X_n[:,1],x[:,1]],
            labels=["Reference Configuration","Dirichlet Boundary","Neumann Boundary","Current Configuration"],
            color=['grey','orange','green','blue'],tr=[1,1,1,0.3],save=True,loc=save_loc+'\Geometry')

    ##Dirichlet error
    u_d_pred = cm.Deformation(X_d)
    a = np.linalg.norm(u_d,axis=1)
    b = np.linalg.norm(u_d-u_d_pred,axis=1)
    dirichlet_error = np.divide(b,a)
    #print(error_1)
    
    f1 = open(save_loc+'\Displacement_value_at_Boundary.txt', 'w')
    for i in range(len(X_d)):
        print("Displacement predicted using trained neural network")
        print(f"X:{X_d[i]}    u_d:{u_d_pred[i]}")
        f1.write(f"X:{X_d[i]}    u_d:{u_d_pred[i]} ")
        f1.write("\n")
    f1.write(f"External work: {external_work}") 
    f1.write("\n")
    f1.write(f"Strain Energy: {SE}")
    f1.close()
    # print(u_d)
    # print(u_d_pred)

    ##Neumann Error
    N = tf.reshape(N_n,shape=[N_n.shape[0],N_n.shape[1],1])
    P_n = First_Piola_Kirchhoff_Stress(X_n)
    f_n = tf.matmul(P_n,N)
    neumann_error = np.squeeze(np.linalg.norm(f_n,axis=1))
    print(neumann_error)

    ##Plotting Dirichlet and Neumann errors
    sc2 = plot.Scatter_Plotter(projection=True)
    sc2.set_grid_size(xlim=[-60,60], ylim=[-60,60], zlim=[0,5])
    sc2.set_title(title="Error")
    sc2.set_labels(x_label='X (mm)', y_label='Y (mm)', z_label='Normalized Error')   
    sc2.plot(X=[X_d[:,0],X_n[:,0]],Y=[X_d[:,1],X_n[:,1]],Z=[dirichlet_error,neumann_error],
            labels=[f'Dirichlet Boundary(increment={step})',f'Neumann Boundary(increment={step})'],
            color=['orange','green'],size=[20,20],save=True,loc=save_loc+'\Error')
    
    ##Plotting Green-Lagrange Strain
    E = cm.Green_Lagrange_Strain(X)
    ##Exx
    Exx = plot.Contour_Plotter(x_b)
    Exx.set_grid_size(xlim=[-110,110], ylim=[-110,110])
    Exx.set_title(title="Green-Lagrange Strain(Exx)")
    Exx.set_labels(x_label="X (mm)", y_label="Y (mm)")
    Exx.plot(x[:,0], x[:,1], E[:,0,0], levels=[-2.5,0,2,4,6,8,10], save=True, loc= save_loc+'\Exx')
    
    ##Eyy
    Eyy = plot.Contour_Plotter(x_b)
    Eyy.set_grid_size(xlim=[-110,110], ylim=[-110,110])
    Eyy.set_title(title="Green-Lagrange Strain(Eyy)")
    Eyy.set_labels(x_label="X (mm)", y_label="Y (mm)")
    Eyy.plot(x[:,0], x[:,1], E[:,1,1],levels=[-2.5,0,2,4,6,8,10], save=True, loc= save_loc+'\Eyy')
    
    ##Exy
    Exy = plot.Contour_Plotter(x_b)
    Exy.set_grid_size(xlim=[-110,110], ylim=[-110,110])
    Exy.set_title(title="Green-Lagrange Strain(Exy)")
    Exy.set_labels(x_label="X (mm)", y_label="Y (mm)")
    Exy.plot(x[:,0], x[:,1], E[:,0,1], levels=[-5,-2.5,-1,0,1,2.5,5],save=True, loc= save_loc+'\Exy')
    
    ##Deformation_x
    deformation_x = plot.Contour_Plotter(x_b)
    deformation_x.set_grid_size(xlim=[-110,110], ylim=[-110,110])
    deformation_x.set_title(title="ux")
    deformation_x.set_labels(x_label="X (mm)", y_label="Y (mm)")
    deformation_x.plot(x[:,0], x[:,1], u[:,0],levels=[-50,-33.33,-25,-16.67,-8.33,8.33,16.67,25,33.33,50],
                       save=True, loc= save_loc+'\Deformation_ux')
    #[-50,-33.33,-25,-16.67,-8.33,8.33,16.67,25,33.33,50]

    ##Deformation_y
    deformation_y = plot.Contour_Plotter(x_b)
    deformation_y.set_grid_size(xlim=[-110,110], ylim=[-110,110])
    deformation_y.set_title(title="uy")
    deformation_y.set_labels(x_label="X (mm)", y_label="Y (mm)")
    deformation_y.plot(x[:,0], x[:,1], u[:,1],levels=[-50,-33.33,-25,-16.67,-8.33,8.33,16.67,25,33.33,50],
                       save=True, loc= save_loc+'\Deformation_uy')

    ##Convergence_Plot
    iterations = int(Training_loss.shape[0])
    Epochs = np.linspace(start = 1, stop = iterations, num = iterations, dtype = np.int32)
    Log_training_loss = np.log10(Training_loss)
    convergence = plot.Line_Graph()
    convergence.set_grid_size(xlim=[0,iterations], ylim=[np.amin(Log_training_loss),np.amax(Log_training_loss)])
    convergence.set_title(title="Convergence")
    convergence.set_labels(x_label="Epochs", y_label="Loss (Log_scale)")
    convergence.plot(X=Epochs,Y=Log_training_loss,labels="Traiaing loss", 
                    color='orange', save=True, loc= save_loc+'\Convergence_plot')
    

tf.keras.backend.set_floatx("float32")  
np.random.seed(25)
tf.random.set_seed(25)
random.seed(25)
global save_loc, step
save_loc = "D:\Mini Thesis Codes\Weak Formulation\Plot"

os.chdir("D:\Mini Thesis Codes\Weak Formulation")

read_geometry()

step = 20                  #Increment

training_data()
Triangle_area()

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

##Optimizer

lr = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001,
                                                    decay_steps=50000,
                                                    decay_rate=0.001)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

cm = Kinametics.Kinametics(model)

#Strain_Energy()
#print(External_Work())

##Training loop
Training_loss = []
start_time = time.time()
relative_norm = 1e4
patience = 500
wait = 0

epochs = 50000

for epoch in range(epochs):
    loss_value = function_factory(model, optimizer)
    if(epoch!=0):
        relative_norm = abs(Training_loss[-1]-loss_value)
        if(relative_norm<1.0e-5):
            wait += 1
        else:
            wait = 0
    Training_loss.append(loss_value)
    if(epoch%100==0):
        tf.print(f"Training loss:{loss_value}    Epochs:{epoch}     wait:{wait}")
    if(wait == patience):
        tf.print("Convergence is reached")
        break

end_time = time.time()
Training_loss = np.array(Training_loss,dtype=np.float32)

tf.print(f"Training time:{round((end_time-start_time)/3600,2)} hour.")
external_work = External_Work()
SE = Strain_Energy()
tf.print(f"External work: {external_work}")
tf.print(f"Strain Energy: {SE}")

tf.print(Training_loss.shape)

model.save_weights(filepath=save_loc+"\model.h5",overwrite=True,save_format='h5')

plotter(step,Training_loss,external_work,SE,save_loc)