import tensorflow as tf

class Kinametics():
    def __init__(self,model):
        self.model = model
    
    @tf.function
    def Deformation(self,X):
        u = self.model(X)
        return u

    @tf.function
    def Deformation_Gradient(self,X):               ##Deformation Gradient
        with tf.GradientTape(persistent=True) as g:
            g.watch(X)
            u = self.Deformation(X)
            ux = u[:,0]
            uy = u[:,1]
        ux_x = g.gradient(ux,X)             #(dux/dx) and (dux/dy)
        uy_x = g.gradient(uy,X)             #(duy/dx) and (duy/dy)    
    
        Grad_u = tf.concat([tf.reshape(ux_x,shape=(X.shape[0],1,2)),
                            tf.reshape(uy_x,shape=(X.shape[0],1,2))], axis=1)

        I = tf.eye(2,batch_shape=[X.shape[0]])  #IDENTITY TENSOR
        F = Grad_u + I 
        #print(F)
        return F

    @tf.function
    def Right_Cauchy_Green(self,X):
        F = self.Deformation_Gradient(X)
        F_T = tf.transpose(F,perm=[0,2,1])
        C = tf.matmul(F_T,F)   
        return C

    @tf.function
    def Green_Lagrange_Strain(self,X):
        C = self.Right_Cauchy_Green(X)
        I = tf.eye(2,batch_shape=[X.shape[0]])      #Identity Tensor
        E = 0.5*(C-I)  
        return E
    
