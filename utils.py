from typing import Callable, Tuple
import warnings
import lasso.dyna as ld
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import numpy as np
import sklearn.decomposition as skld
import tensorflow as tf

class Example2D(object):
    """Read and parse a LS-DYNA binary output file to easily accessible full and reduced order data."""
    def __init__(self, fprefix):
        self.name = fprefix
        self.load_binout(fprefix)
        self.load_massout(fprefix)
        self.stack_components()
        self.pod = None

    def __str__(self):
        return f"Example2D({self.name})"

    def load_massout(self, fprefix):
        self.m = np.genfromtxt(fprefix+'.massout',skip_header=5,skip_footer=1,usecols=(1))


    def load_binout(self, fprefix):
        try:            
            bo = ld.Binout(fprefix+".binout")
            self.x =  bo.read('nodout','x_coordinate')
            self.y =  bo.read('nodout','y_coordinate')
            self.dx = bo.read('nodout','x_displacement')
            self.dy = bo.read('nodout','y_displacement')
            self.vx = bo.read('nodout','x_velocity')
            self.vy = bo.read('nodout','y_velocity')
            self.ax = bo.read('nodout','x_acceleration')
            self.ay = bo.read('nodout','y_acceleration')
            self.fx = bo.read('nodfor','x_force')
            self.fy = bo.read('nodfor','y_force')
            self.ids= bo.read('nodout','ids')
            self.t  = bo.read('nodout','time')

            self.peps = bo.read('eloutdet','nodavg','lower_yield')
            epsxx = bo.read('eloutdet','nodavg', 'lower_eps_xx')
            epsyy = bo.read('eloutdet','nodavg', 'lower_eps_yy')
            epszz = bo.read('eloutdet','nodavg', 'lower_eps_zz')
            epsxy = bo.read('eloutdet','nodavg', 'lower_eps_xy')
            epsyz = bo.read('eloutdet','nodavg', 'lower_eps_yz')
            epszx = bo.read('eloutdet','nodavg', 'lower_eps_zx')
            self.epsbar = np.sqrt(2/3*(epsxx**2+epsyy**2+epszz**2+2*(epsxy**2+epsyz**2+epszx**2)))

        except Exception as e:
            raise RuntimeError(f"There was an error loading binary data from {fprefix}", e)

    def stack_components(self):
        self.xx = np.hstack([self.x, self.y])
        self.x = self.xx[:,:self.xx.shape[1]//2]
        self.y = self.xx[:,self.xx.shape[1]//2:]
        self.dd = np.hstack([self.dx,self.dy])
        self.dx = self.dd[:,:self.dd.shape[1]//2]
        self.dy = self.dd[:,self.dd.shape[1]//2:]
        self.vv = np.hstack([self.vx,self.vy])
        self.vx = self.vv[:,:self.vv.shape[1]//2]
        self.vy = self.vv[:,self.vv.shape[1]//2:]
        self.aa = np.hstack([self.ax,self.ay])
        self.ax = self.aa[:,:self.aa.shape[1]//2]
        self.ay = self.aa[:,self.aa.shape[1]//2:]
        self.ff = np.hstack([self.fx,self.fy])
        self.fx = self.ff[:,:self.ff.shape[1]//2]
        self.fy = self.ff[:,self.ff.shape[1]//2:]
        self.fa = self.split_applied_force()
        
        self.mm = np.hstack([self.m, self.m])
        self.M  = np.diag(self.mm)

        self.fint = self.fa-self.mm*self.aa
        self.pepspeps = np.hstack([self.peps, self.peps])
        self.epsbarepsbar = np.hstack([self.epsbar, self.epsbar])

    def split_applied_force(self):
        applied_list_ids = [1405, 1443, 1481, 1519, 1557, 1595, 1633, 1671, 1709, 1747,
                            1785, 1823, 1861, 1899, 1936, 1972, 2007, 2041, 2074, 2106,
                            2137, 2167, 2196, 2224, 2251, 2277, 2302, 2326, 2349, 2371,
                            2392, 2412, 2431, 2449, 2466, 2482, 2497, 2511, 2524, 2536,
                            2547, 2557, 2566, 2574, 2581, 2587, 2592, 2596, 2599, 2601,
                            2602]
        applied_mask = np.isin(self.ids, applied_list_ids)
        fax = np.zeros(self.fx.shape)
        fax[:,applied_mask] = self.fx[:,applied_mask]
        fay = np.zeros(self.fy.shape)
        fay[:,applied_mask] = self.fy[:,applied_mask]

        return np.hstack([fax,fay])

    def estimate_SVD_fit(self, n_components, V,  A):
        ratios = []
        fnorm = np.linalg.norm(A,'fro')
        for i in range(1,n_components+1):
            Vi = V[:,:i]
            Ar = (Vi @ Vi.T @ A.T).T
            fnormi = np.linalg.norm(A - Ar, 'fro')
            ratios.append(fnormi/fnorm)

        return ratios

    def reduce_components(
            self,
            n_components,
            random_state=42,
            to_reduce=['x','eps'],
            decomposable="self.dd"
        ):
        self.pod = skld.PCA(n_components=n_components,svd_solver='randomized',random_state=random_state)
        self.pod.fit(eval(decomposable))
        self.V = self.pod.components_.T
        
        if 'x' in to_reduce:
            self.Mr = self.V.T @ self.M @ self.V
            self.xr = self.transform(self.xx)
            self.vr = self.transform(self.vv)
            self.ar = self.transform(self.aa)
            self.fr = self.transform(self.ff)
            self.far = self.transform(self.fa)
            self.fint_r = self.transform(self.fint)
        if 'eps' in to_reduce:
            self.peps_r = self.transform(self.pepspeps)
            self.epsbar_r = self.transform(self.epsbarepsbar)


    def transform(self, x):
        # PCA transofrm centers data and POD doesn't, so wrapper.
        if self.pod is None:
            print("POD has not been initialized. Use `example.reduce_components(n_components)` first.")
            return None
        
        return self.pod.transform(x+self.pod.mean_)

    def inverse_transform(self, x):
        
        return self.pod.inverse_transform(x)
    


class NamedFunc:
    """Decorate a function with description"""
    def __init__(self, func, description):
        self.description = description
        self.func = func
    
    def __str__(self):
        return self.description

    def __call__(self,*args,**kwargs):
        return self.func(*args,**kwargs)

class EmptyStruct:
    """Handy store anything container"""
    pass

def roughplot2D(f: np.array, x: np.array, y: np.array,*, width=20, height=4.5, start=0, skip=100, panes=6, unit='mm'):
    fig, axs = plt.subplots(figsize=(width,height),nrows=1,ncols=panes,sharey=True,squeeze=True)
    for i in range(panes):
        ax = axs[i]
        mappable = ax.scatter(x[start+i*skip],y[start+i*skip],c=f[start+i*skip],s=10)
        ax.set_aspect('equal')
        ax.set_xlim([-15, 20])
        ax.set_ylim([-13, 13])
        ax.set_title(rf"$t$ = {i} ms")
        cb = plt.colorbar(mappable,location='bottom',ax=ax);
        if i == 0:
            ax.set_ylabel('mm')
            ax.set_xlabel('mm',labelpad=-20,x=-0.2)
            cb.set_label(unit,labelpad=-40,x=-0.2)

def animatedplot2D(f: np.array, x:np.array, y:np.array, path: str, title: str, total_frames = 50):
    DURATION = 10
    W, H = 10,8
    fps = total_frames // DURATION
    skip_steps = len(f)//total_frames
    fig = plt.figure(figsize=(W, H))

    plot = plt.scatter(x[0],y[0],c=f[0],s=50)
    plot.set_clim(f.min(),f.max())

    ax = plt.gca()
    cb = plt.colorbar(plot,ax=ax);
    ax.time_text = ax.text(-0.05, -0.15, '', 
                                    transform=ax.transAxes)
    plt.title(title)
    ax.set_aspect('equal')
    ax.set_xlim([-15, 20])
    ax.set_ylim([-13, 13])
    ax.set_ylabel('mm')
    ax.set_xlabel('mm')
    cb.set_label('mm')
    plt.tight_layout()

    def draw(frame):
        i = frame * skip_steps
        plot.set_offsets(np.vstack([x[i], y[i]]).T)
        plot.set_array(f[i])
        ax.time_text.set_text(rf"$t$ = {i*0.01:.1f} ms" )
        return plot, ax.time_text


    ani = FuncAnimation(fig, 
                        draw,
                        init_func = lambda: (plot,),
                        frames= total_frames,
                        blit = True)
    ani.save(f"{path}.gif", writer = PillowWriter(fps=fps))  
    return f"Saved to {path}.gif"

def step(x0: np.ndarray, v0: np.ndarray, a: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    # Central difference assuming constant step size
    v = v0 + a*dt
    x = x0 + v*dt
    return x, v

def evolve(x0: np.ndarray, v0: np.ndarray, aa: Callable[[int,np.ndarray,np.ndarray],np.ndarray], dt: float, N: int)  -> Tuple[np.ndarray, np.ndarray]:
    xout = np.zeros((N+1,x0.shape[1]))
    vout = np.zeros((N+1,v0.shape[1]))
    x = x0
    v = v0
    xout[:1] = x
    vout[:1] = v
    for i in range(N):
        a = aa(i,x,v)
        x, v = step(x,v,a,dt)
        xout[i+1:i+2] = x
        vout[i+1:i+2] = v
    return xout, vout

class EpochProgbar(tf.keras.callbacks.ProgbarLogger):
    """Tensorflow progbar extension to count only whole epochs"""
    def __init__(self,skip_epochs=1,**kwargs):
        self.skip_epochs = skip_epochs
        super(EpochProgbar, self).__init__(**kwargs)

    def on_epoch_begin(self,epoch,logs=None):
        if (epoch+1) % self.skip_epochs == 0:
            super(EpochProgbar, self).on_epoch_begin(epoch,logs)
    
    def on_epoch_end(self,epoch,logs=None):
        if (epoch+1) % self.skip_epochs == 0:
            super(EpochProgbar, self).on_epoch_end(epoch,logs)

class SaveBest(tf.keras.callbacks.Callback):
    """Tensorflow callback to save the weights of the model with the least validation loss"""
    best_weights=None   
    def __init__(self, monitor='val_loss'):
        super().__init__()
        self.best = np.Inf
        self.monitor = monitor
    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get(self.monitor)
        if np.less(current_loss, self.best):
            self.best = current_loss            
            self.best_weights= self.model.get_weights()

from sklearn.base import TransformerMixin, BaseEstimator
class MonomialFeatures(TransformerMixin, BaseEstimator):
    def __init__(self,  degree=2, include_bias=False):
        self.degree = degree
        self.include_bias = include_bias
    
    def fit(self, X, y=None):
        n_samples, n_features = self._validate_data(
            X, accept_sparse=False).shape
        self.n_input_features_ = n_features
        self.n_output_features_ = n_features * self.degree + self.include_bias
        return self
    
    def transform(self, X):
        n_samples, n_features = X.shape

        if n_features != self.n_input_features_:
            raise ValueError("X shape does not match training shape")

        XP = np.empty((n_samples, self.n_output_features_),
                        dtype=X.dtype)

        if self.include_bias:
            XP[:, 0] = 1
            current_col = 1
        else:
            current_col = 0
            
        for i in range(1,self.degree+1):
            XP[:,current_col:current_col+n_features] = X ** i
            current_col += n_features

        return XP

