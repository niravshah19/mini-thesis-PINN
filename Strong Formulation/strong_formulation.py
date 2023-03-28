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
def read_geometry(filename):    
    
    ##Reading the mesh information
    df_mesh = pd.read_csv(filepath_or_buffer = filename,skiprows=1, header=None) 
    mesh = df_mesh.to_numpy(dtype=np.float32)
    #print(mesh)
    
        ##Reading the B.C
    df_bc = pd.read_csv(filepath_or_buffer = "Nirav_Boundary conditions.txt",
                    header=0,squeeze=True)#,usecols=['X','N: 464','Total force'])
    #print(df_bc) 

    df_bc = df_bc.values.tolist()
    bc=[]
    for l in range(len(df_bc)):
        x=[]
        s = df_bc[l].split()
        for i in range(len(s)):
            x.append(float(s[i]))
        bc.append(x)
    bc = np.array(bc,dtype=np.float32) 
    
    return mesh, bc

##Return Dirichlet and Neumann boundary mesh
def training_data(mesh,bc,step):     
    
    ##Dirichlet Boundary Condition    
    df_DBottom = pd.read_csv(filepath_or_buffer="DBottom.txt",skiprows=1,header=None)
    DBottom = df_DBottom.to_numpy(dtype=np.float32)
    DBottom = DBottom.reshape(DBottom.shape[0]*DBottom.shape[1])

    df_DLeft = pd.read_csv(filepath_or_buffer="DLeft.txt",skiprows=1,header=None)
    DLeft = df_DLeft.to_numpy(dtype=np.float32)
    DLeft = DLeft.reshape(DLeft.shape[0]*DLeft.shape[1])

    df_DTop = pd.read_csv(filepath_or_buffer="DTop.txt",skiprows=1,header=None)
    DTop = df_DTop.to_numpy(dtype=np.float32)
    DTop = DTop.reshape((DTop.shape[0]*DTop.shape[1]))

    df_DRight = pd.read_csv(filepath_or_buffer="DRight.txt",skiprows=1,header=None)
    DRight = df_DRight.to_numpy(dtype=np.float32)
    DRight = DRight.reshape(DRight.shape[0]*DRight.shape[1])
    
    temp = np.concatenate((DBottom,DLeft,DTop,DRight))
    bc_d = temp[np.logical_not(np.isnan(temp))]
    #print(bc_d.shape[0])
    
    X_d = []
    for i in range(bc_d.shape[0]):
        for j in range(mesh.shape[0]):
            if(mesh[j,0]==bc_d[i]):
                X_d.append(mesh[j,1:])
                break     
    X_d = np.array(X_d,dtype=np.float32)   
    #print(X_d.shape)
    
    n_d = []
    for i in range(len(X_d)):
        if(X_d[i,1]==52.5):
            n_d.append(np.array([0,1]))
            
        elif (X_d[i,1]==-52.5):
            n_d.append(np.array([0,-1]))
            
        elif (X_d[i,0]==52.5):
            n_d.append(np.array([1,0]))
 
        elif (X_d[i,0]==-52.5):
            n_d.append(np.array([-1,0]))

    n_d = np.array(n_d,dtype = np.float32)
    #print(n_d.shape)
    
    ##Calculate displacement at the boundary
    t = bc[step,1]
    #print(t)
    u = []
    for i in range(len(X_d)):
        if(X_d[i,0]==-52.5):
            u.append([-t,0])
        elif(X_d[i,0]==52.5):
            u.append([t,0])
        elif(X_d[i,1]==52.5):
            u.append([0,t])
        elif(X_d[i,1]==-52.5):
            u.append([0,-t])
            
           

    u = np.array(u,dtype=np.float32)
    
    train_d = np.hstack((X_d,n_d,u)) 

    ##Neuman Boundary Condition
    
    df_NBLeft = pd.read_csv(filepath_or_buffer="NBLeft.txt",skiprows=1,header=None)
    NBLeft = df_NBLeft.to_numpy(dtype=np.float32)
    NBLeft = NBLeft.reshape(NBLeft.shape[0]*NBLeft.shape[1])
    
    df_NBRight = pd.read_csv(filepath_or_buffer="NBRight.txt",skiprows=1,header=None)
    NBRight = df_NBRight.to_numpy(dtype=np.float32)
    NBRight = NBRight.reshape(NBRight.shape[0]*NBRight.shape[1])
    
    df_NTLeft = pd.read_csv(filepath_or_buffer="NTLeft.txt",skiprows=1,header=None)
    NTLeft = df_NTLeft.to_numpy(dtype=np.float32)
    NTLeft = NTLeft.reshape(NTLeft.shape[0]*NTLeft.shape[1])
    
    df_NTRight = pd.read_csv(filepath_or_buffer="NTRight.txt",skiprows=1,header=None)
    NTRight = df_NTRight.to_numpy(dtype=np.float32)
    NTRight = NTRight.reshape(NTRight.shape[0]*NTRight.shape[1])
    
    temp2 = np.concatenate((NBLeft,NBRight,NTLeft,NTRight))
    bc_n = temp2[np.logical_not(np.isnan(temp2))]
    # print(bc_n.shape)
    
    X_n = []
    for i in range(bc_n.shape[0]):
        for j in range(mesh.shape[0]):
            if(mesh[j,0]==bc_n[i]):
                X_n.append(mesh[j,1:])
                break
    X_n = np.array(X_n,dtype =np.float32)        
    #print(X_n.shape)
    
    n_n = []
    for i in range(len(X_n)):
        x =X_n[i,0]
        y =X_n[i,1]
        if (x>0 and y>0):
            if(x==17.5):
                n_n.append(np.array([1,0])) 
            elif(y==17.5):
                n_n.append(np.array([0,1]))
            else:
                x_c = 29.5
                y_c = 29.5
                n = np.array([x_c-x,y_c-y])
                n_n.append(n/(np.linalg.norm(n)))
    
        elif (x<0 and y>0):
            if(x==-17.5):
                n_n.append(np.array([-1,0])) 
            elif(y==17.5):
                n_n.append(np.array([0,1]))
            else:
                x_c = -29.5
                y_c = 29.5
                n = np.array([x_c-x,y_c-y])
                n_n.append(n/(np.linalg.norm(n)))
    
        elif (x<0 and y<0):
            if(x==-17.5):
                n_n.append(np.array([-1,0])) 
            elif(y==-17.5):
                n_n.append(np.array([0,-1]))
            else:
                x_c = -29.5
                y_c = -29.5
                n = np.array([x_c-x,y_c-y])
                n_n.append(n/(np.linalg.norm(n)))
        elif (x>0 and y<0):
            if(x==17.5):
                n_n.append(np.array([1,0])) 
            elif(y==-17.5):
                n_n.append(np.array([0,-1]))
            else:
                x_c = 29.5
                y_c = -29.5
                n = np.array([x_c-x,y_c-y])
                n_n.append(n/(np.linalg.norm(n)))
    
    n_n = np.array(n_n,dtype = np.float32)
    train_n = np.hstack((X_n,n_n))
    #print(X_n)
    
    # ##For checking normals 
    # fig1 = plt.figure(figsize=(8,8))
    # ax1_1 = fig1.add_subplot(1,1,1)
    # ax1_1.set_title(label="Geometry")
    # ax1_1.set_xlabel(xlabel='X (mm)', fontsize='large')
    # ax1_1.set_ylabel(ylabel='Y (mm)', fontsize='large')
    # ax1_1.scatter(X[:,0],X[:,1],color='grey',label="Reference Configuration")
    # ax1_1.scatter(X_d[:,0],X_d[:,1],color='orange',label="Dirichlet Boundary")
    # ax1_1.scatter(X_n[:,0],X_n[:,1],color='green',label="Neumann Boundary")
    # for i in range(len(train_d)):
    #     ax1_1.arrow(train_d[i,0],train_d[i,1],5*train_d[i,2],5*train_d[i,3],
    #              head_width=0.5, head_length=0.5)
    # for i in range(len(train_n)):
    #     ax1_1.arrow(train_n[i,0],train_n[i,1],4*train_n[i,2],4*train_n[i,3],
    #               head_width=0.5, head_length=0.5)
    
    return train_d,train_n

##function factory
@tf.function
def function_factory(model_1, model_2, train_d, train_n, X_train, optimizer):

    ##Get coordinates and normals for training data
    X_d = train_d[:,0:2]
    u_d = train_d[:,4:6]

    X_n = train_n[:,0:2]
    n_n = train_n[:,2:4]


    ##Training varaibles
    training_variables = []
    for i in range(len(model_1.trainable_variables)):
        training_variables.append(model_1.trainable_variables[i])
    for i in range(len(model_2.trainable_variables)):
        training_variables.append(model_2.trainable_variables[i])

    # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
    with tf.GradientTape() as g:
        loss_value = loss_fn(model_2,X_d,u_d,X_n,n_n,X_train)
    grads = g.gradient(loss_value,training_variables)
    optimizer.apply_gradients(zip(grads,training_variables))
    return loss_value

##1st Piola-Kirchoff stress
@tf.function
def First_Piola_Kirschoff_Stress(X):               
    F = cm.Deformation_Gradient(X)
    invF =  tf.linalg.inv(F)
    invF_T = tf.transpose(invF,perm=[0,2,1])
    detF =  tf.linalg.det(F)
    P = []
    for i in range(len(detF)):
        P.append(2*(F[i] - pow(detF[i],-2) * invF_T[i]))             
    P = tf.convert_to_tensor(P, tf.float32)
    #print(P)
    return P
 
##Stress divergence using model 2
@tf.function
def divP(model_2,X):            
    with tf.GradientTape(persistent = True) as g:
        g.watch(X)
        P = model_2(X)
        P11 = P[:,0]
        P12 = P[:,1]
        P21 = P[:,2]
        P22 = P[:,3]
    P11_x = g.gradient(P11,X)      #dP11/dx  &  dP11/dy
    P12_x = g.gradient(P12,X)      #dP12/dx  &  dP12/dy
    P21_x = g.gradient(P21,X)      #dP21/dx  &  dP21/dy
    P22_x = g.gradient(P22,X)      #dP22/dx  &  dP22/dy
        
    divP =  tf.concat([tf.reshape(P11_x[:,0] + P12_x[:,1],shape=[X.shape[0],1]),
                       tf.reshape(P21_x[:,0] + P22_x[:,1],shape=[X.shape[0],1])],axis = 1)
            
    return divP

##Loss function
@tf.function
def loss_fn(model_2,X_d,u_d,X_n,n_n,X):        

    ##Stress loss for model_2 at all material points
    P = First_Piola_Kirschoff_Stress(X)
    P_pred = model_2(X)
    loss_1 = tf.math.reduce_mean(tf.square(P[:,0,0]-P_pred[:,0])+\
                                 tf.square(P[:,0,1]-P_pred[:,1])+\
                                 tf.square(P[:,1,0]-P_pred[:,2])+\
                                 tf.square(P[:,1,1]-P_pred[:,3]))
        
    ##Dirichlet boundary loss term
    u_pred = cm.Deformation(X_d)
    loss_2 = tf.math.reduce_mean(tf.square(u_d-u_pred))

    ##Neumann boundary loss
    P_n = model_2(X_n)
    #print(P_n)
    fx = tf.reshape(n_n[:,0]*P_n[:,0] + n_n[:,1]*P_n[:,1],shape=[X_n.shape[0],1])
    fy = tf.reshape(n_n[:,0]*P_n[:,2] + n_n[:,1]*P_n[:,3],shape=[X_n.shape[0],1]) 
    f_n = tf.concat([fx,fy],axis = 1)
    loss_3 = tf.math.reduce_mean(tf.square(tf.math.reduce_euclidean_norm(f_n,axis=1)))
    #print(loss_3)
        
    #Balance equation loss
    loss_4 = tf.math.reduce_mean(tf.square(tf.math.reduce_euclidean_norm(divP(model_2,X),axis=1)))
        
    loss = 100*loss_1 + 10*loss_2 + 500*loss_3 + 200*loss_4 
     
    return loss

##To create cliping boundary for the plots
def boundary(b_mesh):               
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
def plotter(model_2,train_d,train_n,X,step,Training_loss,file_loc):

    # Plotting current configuration
    X_d = train_d[:,0:2]
    u_d = train_d[:,4:6]
        
    X_n = train_n[:,0:2]
    n_n = train_n[:,2:4]

    u = cm.Deformation(X)
    x = X + u
    sc = plot.Scatter_Plotter()
    sc.set_grid_size(xlim=[-110,110], ylim=[-110,110])
    sc.set_title(title="Geometry")
    sc.set_labels(x_label='X (mm)', y_label='Y (mm)')
    sc.plot(X=[X[:,0],X_d[:,0],X_n[:,0],x[:,0]],Y=[X[:,1],X_d[:,1],X_n[:,1],x[:,1]],
            labels=["Reference Configuration","Dirichlet Boundary","Neumann Boundary","Current Configuration"],
            color=['grey','orange','green','blue'],tr=[1,1,1,0.3],save=True,loc=file_loc+'\Geometry')
    
    ##Dirichlet error
    u_d_pred = cm.Deformation(X_d)
    a = np.linalg.norm(u_d,axis=1)
    b = np.linalg.norm(u_d-u_d_pred,axis=1)
    dirichlet_error = np.divide(b,a)
    #print(error_1)
    
    f1 = open(file_loc+'\Displacement_value_at_Boundary.txt', 'w')
    for i in range(len(X_d)):
        print("Displacement predicted using trained neural network")
        print(f"X:{X_d[i]}    u_d:{u_d_pred[i]}")
        f1.write(f"X:{X_d[i]}    u_d:{u_d_pred[i]} ")
        f1.write("\n")
    f1.close()
    # print(u_d)
    # print(u_d_pred)

    ##Neumann Error
    P_n = model_2(X_n)
    fx = tf.reshape(n_n[:,0]*P_n[:,0] + n_n[:,1]*P_n[:,1],shape=[X_n.shape[0],1])
    fy = tf.reshape(n_n[:,0]*P_n[:,2] + n_n[:,1]*P_n[:,3],shape=[X_n.shape[0],1]) 
    f_n = tf.concat([fx,fy],axis = 1)
    neumann_error = np.linalg.norm(f_n,axis=1)
    #print(error_2)

    ##Plotting Dirichlet and Neumann errors
    sc2 = plot.Scatter_Plotter(projection=True)
    sc2.set_grid_size(xlim=[-60,60], ylim=[-60,60], zlim=[0,0.1])
    sc2.set_title(title="Error")
    sc2.set_labels(x_label='X (mm)', y_label='Y (mm)', z_label='Normalized Error')   
    sc2.plot(X=[X_d[:,0],X_n[:,0]],Y=[X_d[:,1],X_n[:,1]],Z=[dirichlet_error,neumann_error],
            labels=[f'Dirichlet Boundary(increment={step})',f'Neumann Boundary(increment={step})'],
            color=['orange','green'],size=[20,20],save=True,loc=file_loc+'\Error')
    
    #Boundary in current state
    boundary_points = np.unique(np.concatenate((X_d,X_n),axis = 0),axis=0)
    X_b = boundary(boundary_points)
    u_b = cm.Deformation(X_b)
    x_b = X_b + u_b

    ##Plotting Green-Lagrange Strain
    E = cm.Green_Lagrange_Strain(X)
    ##Exx
    Exx = plot.Contour_Plotter(x_b)
    Exx.set_grid_size(xlim=[-110,110], ylim=[-110,110])
    Exx.set_title(title="Green-Lagrange Strain(Exx)")
    Exx.set_labels(x_label="X (mm)", y_label="Y (mm)")
    Exx.plot(x[:,0], x[:,1], E[:,0,0], levels=12, save=True, loc= file_loc+'\Exx')
    
    ##Eyy
    Eyy = plot.Contour_Plotter(x_b)
    Eyy.set_grid_size(xlim=[-110,110], ylim=[-110,110])
    Eyy.set_title(title="Green-Lagrange Strain(Eyy)")
    Eyy.set_labels(x_label="X (mm)", y_label="Y (mm)")
    Eyy.plot(x[:,0], x[:,1], E[:,1,1],levels=12, save=True, loc= file_loc+'\Eyy')
    
    ##Exy
    Exy = plot.Contour_Plotter(x_b)
    Exy.set_grid_size(xlim=[-110,110], ylim=[-110,110])
    Exy.set_title(title="Green-Lagrange Strain(Exy)")
    Exy.set_labels(x_label="X (mm)", y_label="Y (mm)")
    Exy.plot(x[:,0], x[:,1], E[:,0,1], levels=12,save=True, loc= file_loc+'\Exy')
    
    ##Error in 1st Piola-Kirschoff Stress Tensor
    P_x = First_Piola_Kirschoff_Stress(X)
    P_pred = model_2(X)
    Error_P = P_x[:,0,0]-P_pred[:,0] + P_x[:,0,1]-P_pred[:,1] +P_x[:,1,0]-P_pred[:,2] + P_x[:,1,1]-P_pred[:,3]
    error1 = plot.Contour_Plotter(x_b)
    error1.set_grid_size(xlim=[-110,110], ylim=[-110,110])
    error1.set_title(title="Error in prediction of P")
    error1.set_labels(x_label="X (mm)", y_label="Y (mm)")
    error1.plot(x[:,0], x[:,1], Error_P,levels=6, save=True, loc= file_loc+'\stress_error')
    
    ##Error in balance equation
    balance_error = tf.math.reduce_euclidean_norm(divP(model_2,X),axis=1)
    error2 = plot.Contour_Plotter(x_b)
    error2.set_grid_size(xlim=[-110,110], ylim=[-110,110])
    error2.set_title(title="Error in balance equation")
    error2.set_labels(x_label="X (mm)", y_label="Y (mm)")
    error2.plot(x[:,0], x[:,1], balance_error, levels=6,save=True, loc= file_loc+'\loss_4')
    
    ##Deformation_x
    deformation_x = plot.Contour_Plotter(x_b)
    deformation_x.set_grid_size(xlim=[-110,110], ylim=[-110,110])
    deformation_x.set_title(title="ux")
    deformation_x.set_labels(x_label="X (mm)", y_label="Y (mm)")
    deformation_x.plot(x[:,0], x[:,1], u[:,0],levels=[-50,-33.33,-25,-16.67,-8.33,8.33,16.67,25,33.33,50],save=True, loc= file_loc+'\Deformation_ux')
    
    ##Deformation_y
    deformation_y = plot.Contour_Plotter(x_b)
    deformation_y.set_grid_size(xlim=[-110,110], ylim=[-110,110])
    deformation_y.set_title(title="uy")
    deformation_y.set_labels(x_label="X (mm)", y_label="Y (mm)")
    deformation_y.plot(x[:,0], x[:,1], u[:,1],levels=[-50,-33.33,-25,-16.67,-8.33,8.33,16.67,25,33.33,50],save=True, loc= file_loc+'\Deformation_uy')
     
    ##Convergence_Plot
    iterations = int(Training_loss.shape[0])
    Epochs = np.linspace(start = 1, stop = iterations, num = iterations, dtype = np.int32)
    Log_training_loss = np.log10(Training_loss)
    convergence = plot.Line_Graph()
    convergence.set_grid_size(xlim=[0,iterations], ylim=[np.amin(Log_training_loss),np.amax(Log_training_loss)])
    convergence.set_title(title="Convergence")
    convergence.set_labels(x_label="Epochs", y_label="Loss (Log_scale)")
    convergence.plot(X=Epochs,Y=Log_training_loss,labels="Traiaing loss",  color='orange', save=True, loc= file_loc+'\Convergence_plot')

tf.keras.backend.set_floatx("float32")  
np.random.seed(25)
tf.random.set_seed(25)
random.seed(25)

os.chdir("D:\Mini Thesis Codes\Strong Formulation\Mesh_3")
filename =  "Nirav3.txt"
mesh,bc = read_geometry(filename)

X = mesh[:,1:]                   #Reference Configuration
step = 20                      #Increment

input_d,input_n = training_data(mesh, bc, step)

#Defining NN Architecture for Deformation
inputs = tf.keras.Input(shape=(2,),dtype=tf.float32)
d1 = tf.keras.layers.Dense(200,activation='tanh',use_bias=True,
                           kernel_initializer='glorot_normal',
                           bias_initializer='zeros')(inputs)
d2 = tf.keras.layers.Dense(200,activation='tanh',use_bias=True,
                           kernel_initializer='glorot_normal',
                           bias_initializer='zeros')(d1)                              
outputs = tf.keras.layers.Dense( 2 ,activation='linear',use_bias=True,
                                kernel_initializer='glorot_normal',
                                bias_initializer='zeros')(d2)
model_1 = tf.keras.Model(inputs=inputs,outputs=outputs)
print("NN for Deformation")
model_1.summary()
print("NN for stress")

#Defining NN Architecture for Stress
inputs_2 = tf.keras.Input(shape=(2,),dtype=tf.float32)
c1 = tf.keras.layers.Dense(200,activation='tanh',use_bias=True,
                           kernel_initializer='glorot_normal',
                           bias_initializer='zeros')(inputs_2)
c2 = tf.keras.layers.Dense(200,activation='tanh',use_bias=True,
                           kernel_initializer='glorot_normal',
                           bias_initializer='zeros')(c1)                              
outputs_2 = tf.keras.layers.Dense( 4 ,activation='linear',use_bias=True,
                                kernel_initializer='glorot_normal',
                                 bias_initializer='zeros')(c2)
model_2 = tf.keras.Model(inputs=inputs_2,outputs=outputs_2)
model_2.summary()


##Optimizer

lr = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001,
                                                    decay_steps=60000,
                                                    decay_rate=0.001)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

cm = Kinametics.Kinametics(model_1)
##Training loop
Training_loss = []
start_time = time.time()

epochs = 60000

##Define parameters for early stopping based on relative norm of loss during training
relative_norm = 1e4
patience = 50
wait = 0

for epoch in range(epochs):
    loss_value = function_factory(model_1, model_2, input_d, input_n, X, optimizer)
    if(epoch!=0):
        relative_norm = abs(Training_loss[-1]-loss_value)
        if(relative_norm<1e-4):
            wait += 1
        else:
            wait = 0
    Training_loss.append(loss_value)
    tf.print(f"Training loss:{loss_value}    Epochs:{epoch}")
    if(wait == patience):
        tf.print("Convergence is reached")
        break

end_time = time.time()
Training_loss = np.array(Training_loss,dtype=np.float32)

tf.print(f"Training time:{round((end_time-start_time)/3600,2)} hour.")

file_loc = "D:\Mini Thesis Codes\Strong Formulation\Plot"     ##Define saving location

model_1.save_weights(filepath=file_loc+"\model_1.h5",overwrite=True,save_format='h5')
model_2.save_weights(filepath=file_loc+"\model_2.h5",overwrite=True,save_format='h5')

plotter(model_2,input_d, input_n,X,step,Training_loss,file_loc)
