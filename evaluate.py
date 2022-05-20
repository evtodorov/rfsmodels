import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pltp
import matplotlib.lines as pltl
import matplotlib.ticker as pltt

from helperFuncs import *

default_dt = 0.01
t_tot = 5
def evaluate(mat, model, trainfunc, testfunc, ar_predfunc, pod_basis_size, training_stop=None,dt=default_dt,model_type='sklearn',pod_ratio='variance'):
    '''
    mat - data
    model - fit/predict object
    trainfunc - model.fit(fitfunc(mat,training_stop),mat.fint_r[:training_stop])
    testfunc - model.predict(testfunc(mat))
    ar_predfunc - evolve(lambda i,x,v: ar_predfunc(i,x,v,mat,model,Mrinv))
    pod_basis_size - number of POD components
    '''
    training_stop = training_stop if training_stop is not None else mat.t.shape[0]
    timesteps = mat.t.shape[0]

    print(f"Evaluating:\n"
    f"\tTraining Data:\t{mat}\n"
    f"\tPrediction Model:\t{model}({trainfunc}, Reduced Internal force)\n"
    f"\tAcceleration function:\t{ar_predfunc}\n"
    f"\tPOD components:\t{pod_basis_size}\n"
    f"\tTraining range:\t0:{training_stop}/{timesteps}")

    mat.reduce_components(pod_basis_size)

    if pod_ratio.lower()=="variance":
        plt.figure(figsize=(16,5))
        plt.subplot(131)
        plt.semilogy(mat.pod.explained_variance_ratio_)
        plt.title(f"POD components energy - remainder {1-sum(mat.pod.explained_variance_ratio_):.3}")
    elif pod_ratio.lower()=='fronorm':
        plt.figure(figsize=(11,5))
        plt.subplot(121)
        pod_ratios = mat.estimate_SVD_fit(pod_basis_size, mat.V, mat.dd)
        plt.plot(range(1,pod_basis_size+1), pod_ratios)
        plt.xlim([0.5,pod_basis_size+0.5])
        plt.gca().xaxis.set_major_locator(pltt.MultipleLocator(1))
        plt.ylabel(r'$\epsilon$')
        plt.xlabel('Number of  components in POD')
        plt.title(f"POD fit ratios")

    x_train,y_train = trainfunc(mat,stop=training_stop)
    x_test, y_test = testfunc(mat,stop=timesteps)

    if model_type.lower()=='sklearn':
        model.fit(x_train,y_train)
    elif model_type.lower()=='keras':
        model.fit(x_train,y_train,shuffle=False)
    else:
        raise ValueError(f"Unknown model type: {model_type}, should be one of ['sklearn','keras']")


    pred_fint_r = model.predict(x_test)

    plt.subplot(132)
    plt.plot(((y_test-pred_fint_r)**2).mean(axis=1)/(y_test.max()-y_test.min()))
    plt.title(f"Internal force prediction - NMSE");

    plt.subplot(133)
    l1 = plt.plot(mat.t,y_test)[0]
    plt.gca().set_prop_cycle(None)
    l2 = plt.plot(mat.t[::10], pred_fint_r[::10],'x')[0]
    rectmin, rectmax = plt.gca().get_ylim()
    plt.gca().add_patch(pltp.Rectangle((0,rectmin),mat.t[training_stop-1],rectmax-rectmin,alpha=0.1))
    plt.gca().annotate("Training data", (mat.t[training_stop-1]/2,0.8*rectmax),ha='center', va='center')
    plt.title("Internal force prediction - comparison")
    plt.gca().set_prop_cycle(None)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    legend1 = plt.legend(handles=[pltl.Line2D([0], [0],
                                                marker='x',
                                                color=colors[i % len(colors)],
                                                label=f"POD{i}") for i in range(min((20,pod_basis_size)))],
                            loc="lower left")
    plt.legend([l1,l2],['reference','prediction'])
    plt.gca().add_artist(legend1)

    plt.suptitle("Performance of fit without evolution")

    Mrinv = np.linalg.inv(mat.Mr)
    hist = lambda: None # empty struct
    ar_predfunc_wrapper = lambda i,x,v: ar_predfunc(i,x,v,mat,model,Mrinv,hist)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xrout, vrout = evolve(mat.xr[0],mat.vr[0],ar_predfunc_wrapper,dt,int(t_tot/dt))
    dt_ratio = round(default_dt/dt)      

    xrout = xrout[::dt_ratio]
    vrout = vrout[::dt_ratio]
    xroutinv = mat.pod.inverse_transform(xrout)
    dx_xroutinv = (xroutinv - xroutinv[0])[:,:xroutinv.shape[1]//2]
    xx_xroutinv = mat.xx[0] + xroutinv - xroutinv[0]

    plt.figure(figsize=(16,5))
    plt.subplot(131)
    l1 = plt.plot(mat.t[::10],mat.xr[::10])[0]
    plt.gca().set_prop_cycle(None)
    l2 = plt.plot(mat.t[::10],xrout[::10],'x')[0]
    plt.legend([l1,l2],['reference', 'prediction'],loc='upper left')
    plt.title("Reduced components - comparison")
    plt.subplot(132)
    plt.plot(((mat.xr[:]-xrout)**2).mean(axis=1)/(mat.xr.max()-mat.xr.min()))
    plt.title("Predicted state in the reduced space - NMSE")
    plt.subplot(133)
    plt.plot(((mat.xx-xx_xroutinv)**2).mean(axis=1)/(mat.xx.max()-mat.xx.min()))
    plt.title("NMSE after transformation to full space");
    plt.suptitle("Performance of fit during evolution")


    roughplot(dx_xroutinv, xx_xroutinv[:,:xx_xroutinv.shape[1]//2],xx_xroutinv[:,xx_xroutinv.shape[1]//2:])
    plt.suptitle("X-displacement: prediction")

    dx_xx = (mat.xx - mat.xx[0])[:,:mat.xx.shape[1]//2]
    roughplot(dx_xx, mat.xx[:,:mat.xx.shape[1]//2],mat.xx[:,mat.xx.shape[1]//2:])
    plt.suptitle("X-dispalcement: reference")

    return {"xrout": xrout, "vrout":vrout, "xroutinv":xroutinv, "hist":hist, "xr": np.copy(mat.xr), 'model':model}