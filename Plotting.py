import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib import cm

np.random.seed(25)


class Plotter():
    
    def __init__(self):
        self.title = " "
        self.xlim = [-1,1]
        self.ylim = [-1,1]
        self.zlim=[0,1]
        self.x_label = " "  
        self.y_label = " " 
        self.z_label = " "
        
    def set_grid_size(self,xlim,ylim,zlim=[0,1]):
        self.xlim = xlim
        self.ylim = ylim 
        self.zlim = zlim
        
    def set_title(self,title):
        self.title = title
        
    def set_labels(self,x_label,y_label,z_label=" "):
        self.x_label = x_label
        self.y_label = y_label
        self.z_label = z_label
         
class Contour_Plotter(Plotter):
    def __init__(self,ver,codes=[]):
        self.ver = ver
        self.codes = codes
        if(len(codes)==0):
            self.path = Path(ver) 
        else:
            self.path = Path(ver, codes)       ##Create clip path

    def get_vertices(self):
        return self.ver
    
    def get_codes(self):
        return self.codes

    def plot_grid(self):
        patch = patches.PathPatch(self.path, facecolor='none', lw=2)
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(1,1,1)
        ax.set_xlim(self.xlim[0], self.xlim[0])
        ax.set_ylim(self.ylim[1], self.ylim[1])
        ax.add_patch(patch)
        plt.show()
        plt.close(fig)
        
    def check_clip_path(self,points):   ##To check clip path with the given boundary
        patch = patches.PathPatch(self.path, facecolor='none', lw=2)
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(1,1,1)
        ax.set_xlim(self.xlim[0], self.xlim[1])
        ax.set_ylim(self.ylim[0], self.ylim[1])
        ax.add_patch(patch)
        ax.scatter(points[:,0],points[:,1])
        plt.show()
        plt.close(fig)

    def plot(self,X,Y,Z,levels,save=False,loc=""): ##plot contour figure
        clip_patch = patches.PathPatch(self.path,facecolor='none', lw=2)
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(1,1,1)
        ax.set_xlim(self.xlim[0], self.xlim[1])
        ax.set_ylim(self.ylim[0], self.ylim[1])
        ax.set_title(label=self.title)
        ax.set_xlabel(xlabel=self.x_label, fontsize='large')
        ax.set_ylabel(ylabel=self.y_label, fontsize='large')
        ax.add_patch(clip_patch)
        dntr = ax.tricontour(X,Y,Z, levels=levels, linewidths=0.5, colors='k')
        cntr = ax.tricontourf(X,Y,Z,levels=levels,cmap=cm.jet)  
        for d in dntr.collections:
            d.set_clip_path(clip_patch)
        for c in cntr.collections:
            c.set_clip_path(clip_patch)  
        fig.colorbar(cntr,ax=ax)
        ax.grid(visible = True,clip_on = True, clip_path = clip_patch)
        if(save):
            fig.savefig(fname = loc)
        return fig

class Scatter_Plotter(Plotter):

    def __init__(self,projection=False):
        self.projection = projection

    def plot(self,X,Y,labels,color,Z=[],size=[],style=[],tr=[],save=False,loc=""):
        if(len(size)==0):
            for i in range(len(X)):
                size.append(12)
                
        if(len(style)==0):
                for i in range(len(X)):
                    style.append("o")
                    
        if(len(tr)==0):
                for i in range(len(X)):
                    tr.append(1)
                    
        fig = plt.figure(figsize=(10,8))       
                    
        if(self.projection):
            ax = fig.add_subplot(1,1,1,projection='3d')
            ax.set_title(label=self.title)
            ax.set_xlabel(xlabel=self.x_label, fontsize='large')
            ax.set_ylabel(ylabel=self.y_label, fontsize='large')
            ax.set_zlabel(zlabel=self.z_label, fontsize='large')
            ax.set_xlim(self.xlim[0], self.xlim[1])
            ax.set_ylim(self.ylim[0], self.ylim[1])
            ax.set_zlim3d(self.zlim[0],self.zlim[1])
             
            for i in range(len(X)):
                ax.scatter(X[i],Y[i],Z[i],label=labels[i],c = color[i], s = size[i], 
                           marker = style[i],alpha=tr[i] )
                
            ax.legend(loc = 'lower right', fontsize='small')
            ax.grid()
            
        else:
            ax = fig.add_subplot(1,1,1)
            
            ax.set_title(label=self.title)
            ax.set_xlabel(xlabel=self.x_label, fontsize='large')
            ax.set_ylabel(ylabel=self.y_label, fontsize='large')
            ax.set_xlim(self.xlim[0], self.xlim[1])
            ax.set_ylim(self.ylim[0], self.ylim[1])
            
            for i in range(len(X)):
                ax.scatter(X[i],Y[i],label=labels[i],c = color[i], s = size[i], 
                           marker = style[i],alpha=tr[i] )
                
            ax.legend(loc = 'lower right', fontsize='small')
            ax.grid()
            
        if(save):
            fig.savefig(fname = loc)
            
        return fig
    
class Line_Graph(Plotter):

    def plot(self,X,Y,labels,color='orange',line_style=None,save=False,loc=""):
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(1,1,1)
        ax.set_title(label =self.title, fontsize=16)
        if type(X) is list:
            for i in range(len(X)):
                ax.plot(X[i],Y[i],color=color[i],label=labels[i],ls=line_style[i])
        else:
            ax.plot(X,Y,color=color,label=labels)
            
        ax.set_xlabel(xlabel=self.x_label, fontsize=16)
        ax.set_ylabel(ylabel=self.y_label, fontsize=16)
        ax.set_xlim(self.xlim[0],self.xlim[1])
        ax.set_ylim(self.ylim[0],self.ylim[1])
        ax.legend()
        ax.grid()
        
        if(save):
            fig.savefig(fname = loc)
            
        return fig
