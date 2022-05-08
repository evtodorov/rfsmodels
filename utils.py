from matplotlib import transforms
import lasso.dyna as ld
import numpy as np
import sklearn.decomposition as skld

class Example2D(object):

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

    def reduce_components(self, n_components, random_state=42, mass_correction=False):
        self.pod = skld.PCA(n_components=n_components,svd_solver='randomized',random_state=random_state)
        self.pod.fit(self.dd)
        self.V = self.pod.components_.T
        self.Mr = self.V.T @ self.M @ self.V
        
        if mass_correction:
            self.Vm = np.linalg.inv(self.Mr) @ self.V.T @ self.M
            transform = self.transform_Morth
        else:
            transform = self.transform

        self.xr = transform(self.xx)
        self.vr = transform(self.vv)
        self.ar = transform(self.aa)
        self.fr = transform(self.ff)
        self.far = transform(self.fa)

        #self.fint_r = transform(self.ff-self.mm*self.aa)
        self.fint_r = transform(self.fa-self.mm*self.aa)


    def transform(self, x):
        # PCA transofrm centers data and POD doesn't, so wrapper.
        if self.pod is None:
            print("POD has not been initialized. Use `example.reduce_components(n_components)` first.")
            return None
        
        return self.pod.transform(x+self.pod.mean_)

    def inverse_transform(self, x):
        
        return self.pod.inverse_transform(x)
    
    def transform_Morth(self, x, Vm=None):
        if Vm is None:
            try:
                Vm = self.Vm
            except AttributeError:
                self.Vm = np.linalg.inv(self.Mr) @ self.V.T @ self.M

        return (Vm @ x.T).T
    
    def inverse_transform_Morth(self, x, Vm=None):

        if Vm is None:
            try:
                Vm = self.Vm
            except AttributeError:
                self.Vm = np.linalg.inv(self.Mr) @ self.V.T @ self.M

        return (Vm.T @ x.T).T

class NamedFunc:
    def __init__(self, func, description):
        self.description = description
        self.func = func
    
    def __str__(self):
        return self.description

    def __call__(self,*args,**kwargs):
        return self.func(*args,**kwargs)