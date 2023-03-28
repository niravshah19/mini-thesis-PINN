import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, InputLayer
import random
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
    
def function_factory(model, x_0, x_l, x_f, x_valid):

    # obtain the shapes of all trainable parameters in the model
    shapes = tf.shape_n(model.trainable_variables)
    n_tensors = len(shapes)

    # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
    # prepare required information first
    count = 0
    idx = [] # stitch indices
    part = [] # partition indices

    for i, shape in enumerate(shapes):
        n = np.product(shape)
        idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
        part.extend([i]*n)
        count += n

    part = tf.constant(part)


    @tf.function
    def assign_new_model_parameters(params_1d):
        #A function updating the model's parameters with a 1D tf.Tensor.
    
        params = tf.dynamic_partition(params_1d, part, n_tensors)
        for i, (shape, param) in enumerate(zip(shapes, params)):
            model.trainable_variables[i].assign(tf.reshape(param, shape))

    # now create a function that will be returned by this factory
    @tf.function
    def f( params_1d):
        ##A function that can be used by tfp.optimizer.lbfgs_minimize.
        ##This function is created by function_factory.

        x_tf = tf.convert_to_tensor(x_f,dtype = tf.float32)
        # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
        with tf.GradientTape() as g:
            assign_new_model_parameters(params_1d)
            fx = residual(x_tf)
            u_0,ux_l = bc(x_0,x_l)
            loss_value = loss_fn(u_0,ux_l,fx) 
        grads = g.gradient(loss_value, model.trainable_variables)
        grads = tf.dynamic_stitch(idx, grads)
        validation_loss_val = validation_loss(x_valid)
        
        # print out iteration & loss
        f.iter.assign_add(1)
        tf.print("Iter:", f.iter, "loss:", loss_value,"validation_error:",validation_loss_val)

        # store loss value so we can retrieve later
        tf.py_function(f.training_loss.append, inp=[loss_value], Tout=[])

        return loss_value, grads
    
    ##Validation error
    @tf.function
    def validation_loss(x_valid):
        fx = residual(x_valid)
        validation_loss = tf.math.reduce_mean(tf.square(fx))
        tf.py_function(f.validation_loss.append, inp=[validation_loss], Tout=[])
        return validation_loss
        
    ##Fx calculation
    @tf.function
    def residual(x_f):
        with tf.GradientTape() as g:
            g.watch(x_f)
            with tf.GradientTape() as gg:
                gg.watch(x_f)
                u = model(x_f,training=True)
            u_x = gg.gradient(u,x_f)
        u_xx = g.gradient(u_x,x_f)
        fx = 15*( u_xx + (1.1 * 10**-5)*x_f)
        return fx
    
    ##Loss function
    @tf.function
    def loss_fn(u_0,ux_l,f): 
        loss = tf.math.reduce_mean(tf.square(u_0)) + \
             tf.math.reduce_mean(tf.square(ux_l)) + \
            tf.math.reduce_mean(tf.square(f))

        return loss
    
    #Boundary condition
    @tf.function
    def bc(x_0,x_l):

        with tf.GradientTape() as gc:
            gc.watch(x_l)
            u_l = model(x_l)
        ux_l = gc.gradient(u_l, x_l)
        u_0 = model(x_0)

        return u_0,ux_l

    # store these information as members so we can use them outside the scope
    f.iter = tf.Variable(0)
    f.idx = idx
    f.part = part
    f.shapes = shapes
    f.assign_new_model_parameters = assign_new_model_parameters
    f.training_loss = []
    f.validation_loss = []

    return f

@tf.function
def analytical_fn(x):
    u = ((-0.0011e-2)/6)*x**3+0.055*x
    return u

# use float32 by default
tf.keras.backend.set_floatx("float32")

np.random.seed(25)
tf.random.set_seed(25)
os.chdir("D:/Mini Thesis Codes/1D bar in Tension")
np.random.seed(25)
tf.random.set_seed(25)
length = 100 #Length of a rod
x_f = np.random.uniform(0,length,200).astype(np.float32)  #Collocation points
x_0 = np.array([0]).astype(np.float32)
x_l = np.array([length]).astype(np.float32)
x_valid = np.random.rand(100)*length
x_test = np.linspace(0,length,100).reshape((100,1)).astype(np.float32)

layers = [ 1, 200,  1] #Neurons in each layer

 ##Defining NN Architecture 
inputs = tf.keras.Input(shape=(1,),dtype=tf.float32)
d1 = tf.keras.layers.Dense(layers[1],activation='tanh',use_bias=True,
                           kernel_initializer='glorot_normal',
                           bias_initializer='zeros')(inputs)                                                                                                                                                                                                                                                                             
outputs = tf.keras.layers.Dense(1,activation='linear',use_bias=True,
                                kernel_initializer='glorot_normal',
                                bias_initializer='zeros')(d1)
model = tf.keras.Model(inputs=inputs,outputs=outputs)
model.summary()

func = function_factory(model, x_0,x_l, x_f, x_valid)

# convert initial model parameters to a 1D tf.Tensor
init_params = tf.dynamic_stitch(func.idx, model.trainable_variables)

    # train the model with L-BFGS solver
results = tfp.optimizer.lbfgs_minimize(
    value_and_gradients_function=func, initial_position=init_params,
     max_iterations=10000, f_absolute_tolerance = 1e-12, 
     tolerance = 1.0e-12, x_tolerance = 1e-12)

# after training, the final optimized parameters are still in results.position
# so we have to manually put them back to the model
func.assign_new_model_parameters(results.position)
u_test = analytical_fn(x_test)

u_pred = model(x_test)

test_loss = tf.math.reduce_mean(tf.square(u_pred-u_test))
print("Calculate the loss of neural network w.r.t analytical solution")
print(f"Test Loss= {test_loss}")  

##Result Plotting
fig1 = plt.figure(figsize=(8,6))
ax1 = fig1.add_subplot(1,1,1)
#ax1.set_title(label ='Comparison of Result', fontsize=16)
ax1.plot(x_test,u_test,color='red',label='Analytical solution')
ax1.plot(x_test,u_pred,color='blue',ls=':',label='Neural network solution')
ax1.set_xlabel(xlabel="x (mm)", fontsize=16)
ax1.set_ylabel(ylabel="u (mm)", fontsize=16)
ax1.legend()
ax1.grid()

iter = int(func.iter)
Epochs = np.linspace(start = 1, stop = iter, num = iter, dtype = np.int32)
#print(Epochs)
fig2 = plt.figure(figsize=(8,6))
ax2 = fig2.add_subplot(1,1,1)
#ax2.set_title(label ='Convergence', fontsize=16)
ax2.plot(Epochs,func.training_loss,color='red',label='Training Error')
ax2.plot(Epochs,func.validation_loss,color='blue',label='Validation Error')
ax2.set_xlabel(xlabel="Epochs", fontsize=16)
ax2.set_ylabel(ylabel="MSE", fontsize=16)
ax2.set_ylim(0,1e-4)
ax2.legend()
ax2.grid()
plt.show()
model.reset_states()