import pandas as pd
import numpy as np
import sys
sys.path.insert(0,'D:\Mini Thesis Codes')      ##Plotting & Kinematics file 
import Plotting as plot
import os
import tensorflow as tf
import Kinematics

###Get array of corodinates from array containig node id
def coordinates_from_nodeID(nodeID):

    X_i = []
    u_i = []
    for i,ID in enumerate(nodeID):
        ID = int(ID)-1
        X_i.append(mesh[ID,1:]) 
        u_i.append(u[ID,:])
    X_i = np.array(X_i,dtype=np.float32) 
    u_i = np.array(u_i,dtype=np.float32)
    return X_i,u_i 

##Return mesh and bc
def read_geometry():    
    
    ##define global variable for mesh and bounary_condition(bc)
    global mesh,u,X, X_d, X_n, u_d, u_n, N_n
    
    ##Reading the mesh information
    df_mesh = pd.read_csv(filepath_or_buffer = "Mesh.txt",skiprows=1, header=None) 
    mesh = df_mesh.to_numpy(dtype=np.float32)
    X = mesh[:,1:]
    #print(mesh.shape)

    ##reading displacements from Abaqus simulation
    df_u = pd.read_csv(filepath_or_buffer = "U.txt",skiprows=3, header=None,sep='\s+') 
    u = df_u.to_numpy(dtype=np.float32)
    u = u[:,2:]
    #print(u.shape)

    ##Boundary nodes    
    df_Dirichlet = pd.read_csv(filepath_or_buffer="Dirichlet.txt",skiprows=1,header=None)
    Dirichlet = df_Dirichlet.to_numpy(dtype=np.float32)
    Dirichlet = Dirichlet.reshape(Dirichlet.shape[0]*Dirichlet.shape[1])
    Dirichlet = Dirichlet[np.logical_not(np.isnan(Dirichlet))]
    X_d,u_d = coordinates_from_nodeID(Dirichlet)

    df_Neumann = pd.read_csv(filepath_or_buffer="Neumann.txt",skiprows=1,header=None)
    Neumann = df_Neumann.to_numpy(dtype=np.float32)
    Neumann = Neumann.reshape(Neumann.shape[0]*Neumann.shape[1])
    Neumann = Neumann[np.logical_not(np.isnan(Neumann))]
    X_n,u_n = coordinates_from_nodeID(Neumann)

    N_n = []
    for i in range(len(X_n)):
        x =X_n[i,0]
        y =X_n[i,1]
        if (x>0 and y>0):
            if(x==17.5):
                N_n.append(np.array([1,0])) 
            elif(y==17.5):
                N_n.append(np.array([0,1]))
            else:
                x_c = 29.5
                y_c = 29.5
                n = np.array([x_c-x,y_c-y])
                N_n.append(n/(np.linalg.norm(n)))
    
        elif (x<0 and y>0):
            if(x==-17.5):
                N_n.append(np.array([-1,0])) 
            elif(y==17.5):
                N_n.append(np.array([0,1]))
            else:
                x_c = -29.5
                y_c = 29.5
                n = np.array([x_c-x,y_c-y])
                N_n.append(n/(np.linalg.norm(n)))
    
        elif (x<0 and y<0):
            if(x==-17.5):
                N_n.append(np.array([-1,0])) 
            elif(y==-17.5):
                N_n.append(np.array([0,-1]))
            else:
                x_c = -29.5
                y_c = -29.5
                n = np.array([x_c-x,y_c-y])
                N_n.append(n/(np.linalg.norm(n)))
        elif (x>0 and y<0):
            if(x==17.5):
                N_n.append(np.array([1,0])) 
            elif(y==-17.5):
                N_n.append(np.array([0,-1]))
            else:
                x_c = 29.5
                y_c = -29.5
                n = np.array([x_c-x,y_c-y])
                N_n.append(n/(np.linalg.norm(n)))
    
    N_n = np.array(N_n,dtype = np.float32)
    #print(N_n.shape)

    #For checking normals 
    import matplotlib.pyplot as plt
    fig1 = plt.figure(figsize=(8,8))
    ax1_1 = fig1.add_subplot(1,1,1)
    ax1_1.set_title(label="Geometry")
    ax1_1.set_xlabel(xlabel='X (mm)', fontsize='large')
    ax1_1.set_ylabel(ylabel='Y (mm)', fontsize='large')
    ax1_1.scatter(X[:,0],X[:,1],color='grey',label="Reference Configuration")

    for i,txt in enumerate(Neumann):
       ax1_1.annotate(int(txt),(X_n[i,0]+0.1,X_n[i,1]+0.1))
    ax1_1.scatter(X_n[:,0],X_n[:,1],color='green',label="Neumann Boundary")
    for i in range(len(N_n)):
        ax1_1.arrow(X_n[i,0],X_n[i,1],5*N_n[i,0],5*N_n[i,1],head_width=0.5, head_length=0.5)

    plt.legend()
    #plt.show()
    plt.savefig(fname = save_loc+"\Geomtery_Reference_config_normalized_cordinates")
    

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
    #tf.print(P)
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
        
    divP =  tf.concat([tf.reshape(P11_x[:,0] + P12_x[:,1],shape=[Xii.shape[0],1]),
                       tf.reshape(P21_x[:,0] + P22_x[:,1],shape=[Xii.shape[0],1])],axis = 1)
            
    return divP

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
def plotter():

    x = X + model(X)
    ##Dirichlet error
    u_d_pred = cm.Deformation(X_d)
    a = np.linalg.norm(u_d,axis=1)
    b = np.linalg.norm(u_d-u_d_pred,axis=1)
    dirichlet_error = np.divide(b,a)
    #print(error_1)
    
    # print(u_d)
    # print(u_d_pred)

    ##Neumann Error
    N = tf.reshape(N_n,shape=[N_n.shape[0],N_n.shape[1],1])
    P_n = First_Piola_Kirchhoff_Stress(X_n)
    f_n = tf.matmul(P_n,N)
    #print(f_n.shape)
    neumann_error = np.squeeze(np.linalg.norm(f_n,axis=1))
    print(neumann_error)

    ##Plotting Dirichlet and Neumann errors
    sc2 = plot.Scatter_Plotter(projection=True)
    sc2.set_grid_size(xlim=[-60,60], ylim=[-60,60], zlim=[0,3])
    sc2.set_title(title="Error")
    sc2.set_labels(x_label='X (mm)', y_label='Y (mm)', z_label='Normalized Error')   
    sc2.plot(X=[X_d[:,0],X_n[:,0]],Y=[X_d[:,1],X_n[:,1]],Z=[dirichlet_error,neumann_error],
            labels=['Dirichlet Boundary','Neumann Boundary'],
            color=['orange','green'],size=[20,20],save=True,loc=save_loc+'\Error')
    

    ## Plotting current configuration
    #x = X + u
    #x_d = X_d + u_d
    #x_n = X_n + u_n
    #sc = plot.Scatter_Plotter()
    ##sc.set_figure_size((10,8))
    #sc.set_grid_size(xlim=[-110,110], ylim=[-110,110])
    #sc.set_title(title="Geometry")
    #sc.set_labels(x_label='X (mm)', y_label='Y (mm)')
    #sc.plot(X=[X[:,0],X_d[:,0],X_n[:,0],x[:,0],x_d[:,0],x_n[:,0]],Y=[X[:,1],X_d[:,1],X_n[:,1],x[:,1],x_d[:,1],x_n[:,1]],
    #        labels=["Reference Configuration","Dirichlet Boundary in reference state","Neumann Boundary in reference state",
    #                "Current Configuration","Dirichlet Boundary in current state","Neumann Boundary in current state"],
    #        color=['blue','orange','green','blue','orange','green'],tr=[1,1,1,0.3,0.3,0.3],save=True,loc=save_loc+'\Geometry')

    
    #Boundary in current state
    boundary_points = np.unique(np.concatenate((X_d,X_n),axis = 0),axis=0)
    X_b = boundary(boundary_points)
    u_b = cm.Deformation(X_b)
    x_b = X_b + u_b
    
    ##Deformation_x
    deformation_x = plot.Contour_Plotter(x_b)
    deformation_x.set_grid_size(xlim=[-110,110], ylim=[-110,110])
    deformation_x.set_title(title="ux")
    deformation_x.set_labels(x_label="X (mm)", y_label="Y (mm)")
    deformation_x.plot(x[:,0], x[:,1], u[:,0],levels=[-50,-33.33,-25,-16.67,-8.33,8.33,16.67,25,33.33,50],
                       save=True, loc= save_loc+'\Deformation_ux')

    ##Deformation_y
    deformation_y = plot.Contour_Plotter(x_b)
    deformation_y.set_grid_size(xlim=[-110,110], ylim=[-110,110])
    deformation_y.set_title(title="uy")
    deformation_y.set_labels(x_label="X (mm)", y_label="Y (mm)")
    deformation_y.plot(x[:,0], x[:,1], u[:,1],levels=[-50,-33.33,-25,-16.67,-8.33,8.33,16.67,25,33.33,50],
                       save=True, loc= save_loc+'\Deformation_uy')

    ##Error in balance equation
    balance_error = tf.math.reduce_euclidean_norm(divP(X),axis=1)
    error2 = plot.Contour_Plotter(x_b)
    error2.set_grid_size(xlim=[-110,110], ylim=[-110,110])
    error2.set_title(title="Error in balance equation")
    error2.set_labels(x_label="X (mm)", y_label="Y (mm)")
    error2.plot(x[:,0], x[:,1], balance_error, levels=[0,0.5,1,2,4,6], save=True, loc= save_loc+'\loss_4')
    #[0.1,0.4,0.8,1.5,2,2.5,3.5]

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


global save_loc, Training_loss,cm
save_loc = "D:\Mini Thesis Codes\Abaqus Results\Plot"

os.chdir("D:\Mini Thesis Codes\Abaqus Results")

read_geometry()

#Defining NN Architecture for Deformation
inputs = tf.keras.Input(shape=(2,),dtype=tf.float32)
d1 = tf.keras.layers.Dense(100,activation='tanh',use_bias=True,
                           kernel_initializer='glorot_normal',
                           bias_initializer='zeros')(inputs)
d2 = tf.keras.layers.Dense(100,activation='tanh',use_bias=True,
                           kernel_initializer='glorot_normal',
                           bias_initializer='zeros')(d1)
d3 = tf.keras.layers.Dense(100,activation='tanh',use_bias=True,
                           kernel_initializer='glorot_normal',
                           bias_initializer='zeros')(d2)
d4 = tf.keras.layers.Dense(100,activation='tanh',use_bias=True,
                           kernel_initializer='glorot_normal',
                           bias_initializer='zeros')(d3)
outputs = tf.keras.layers.Dense( 2 ,activation='linear',use_bias=True,
                                kernel_initializer='glorot_normal',
                                bias_initializer='zeros')(d4)
model = tf.keras.Model(inputs=inputs,outputs=outputs)
print("NN for Deformation")
model.summary()

lr = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001,
                                                    decay_steps=10000,
                                                    decay_rate=0.1)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')

early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='loss',
                    min_delta=1.0e-6,
                    patience=500,
                    verbose=0,
                    mode='min')

history = model.fit(X,u,shuffle=False,callbacks=early_stopping,epochs=10000,verbose=1)
cm = Kinematics.Kinematics(model)
Training_loss = np.array(history.history['loss'],dtype=np.float32)
plotter()