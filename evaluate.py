from random import random
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pltp
import matplotlib.lines as pltl
import matplotlib.ticker as pltt

from scipy.interpolate import interp1d

from utils import evolve, roughplot2D, EmptyStruct

default_dt = 0.01
t_tot = 5
def evaluate2D(mat, model, trainfunc, testfunc, ar_predfunc, pod_basis_size,
                training_stop=None, dt=default_dt, fit_params=dict(), save_best = None, eps_basis_size = None, interrupt=False,
                **kwargs):
    '''
    mat - data
    model - fit/predict object
    trainfunc - model.fit(*fitfunc(mat,training_stop),**fit_params)
    testfunc - model.predict(*testfunc(mat))
    ar_predfunc - evolve(lambda i,x,v: ar_predfunc(i,x,v,mat,model,Mrinv,hist))
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

    if eps_basis_size is None:
        mat.reduce_components(pod_basis_size)
    else:
        mat.reduce_components(eps_basis_size,to_reduce=['eps'],decomposable="self.pepspeps")
        mat.reduce_components(pod_basis_size,to_reduce='x')

    
    plt.figure(figsize=(20,6))

    x_train,y_train = trainfunc(mat,stop=training_stop)
    x_test, y_test = testfunc(mat,stop=timesteps)

    if 'validtion_split' in fit_params.keys():
        from sklearn.model_selection import train_test_split
        validation_split = fit_params.pop('validation_split')
        x_train2, x_val, y_train2, y_val = train_test_split(x_train, y_train, validation_split, random_state=42)
        fit_history = model.fit(x_train2,y_train2,validation_data=(x_val,y_val),**fit_params)
    else:
        fit_history = model.fit(x_train,y_train,**fit_params)

    if save_best is not None:
        model.set_weights(save_best.best_weights)

    pred_fint_r = model.predict(x_test)

    plt.subplot(121)
    plt.plot(mat.t, ((y_test-pred_fint_r)**2)[:,:pod_basis_size].mean(axis=1)/(y_test.max()-y_test.min()))
    plt.xlabel("ms")
    plt.ylabel(r"N$^2$")
    plt.title(f"NMSE (reduced space)");

    plt.subplot(122)
    l1 = plt.plot(mat.t,y_test[:,:pod_basis_size])[0]
    plt.gca().set_prop_cycle(None)
    l2 = plt.plot(mat.t[::10], pred_fint_r[::10,:pod_basis_size],'x')[0]
    rectmin, rectmax = plt.gca().get_ylim()
    plt.gca().add_patch(pltp.Rectangle((0,rectmin),mat.t[training_stop-1],rectmax-rectmin,alpha=0.1))
    plt.gca().annotate("Training data", (mat.t[training_stop-1]/2,0.8*rectmax),ha='center', va='center')
    plt.title("Predicted components - comparison")
    plt.xlabel("ms")
    plt.ylabel("N")
    plt.gca().set_prop_cycle(None)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    legend1 = plt.legend(handles=[pltl.Line2D([0], [0],
                                                marker='x',
                                                color=colors[i % len(colors)],
                                                label=rf"$\hat{{f}}_{{{i}}}$") for i in range(min((20,pod_basis_size)))],
                            loc="upper left",
                            bbox_to_anchor=(1.01,1.2))
    plt.legend([l1,l2],['reference','prediction'],framealpha=0.6,edgecolor="gray")
    plt.gca().add_artist(legend1)

    plt.suptitle("Performance of internal force fit without evolution (single prediction step from ground truth)")
    plt.tight_layout()

    if interrupt:
        plt.figure()
        plt.semilogy(fit_history.history['loss'],label='training loss')
        try:
            plt.semilogy(fit_history.history['val_loss'],label='validation loss')
        except:
            pass
        plt.xlabel('epoch')
        plt.ylabel("MSE")
        plt.legend()
        plt.title("Training progression")
        return { "xr": np.copy(mat.xr), 'model':model, 'fit_history':fit_history, 'pred_fint_r':pred_fint_r}

    Mrinv = np.linalg.inv(mat.Mr)
    hist = EmptyStruct()
    mat.far_interp = interp1d(np.arange(0,t_tot+default_dt, default_dt), mat.far, axis=0)
    hist.dt = dt
    ar_predfunc_wrapper = lambda i,x,v: ar_predfunc(i,x,v,mat,model,Mrinv,hist)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xrout, vrout = evolve(mat.xr[:1],mat.vr[:1],ar_predfunc_wrapper,dt,int(t_tot/dt))
    dt_ratio = round(default_dt/dt)      

    xrout = xrout[::dt_ratio]
    vrout = vrout[::dt_ratio]
    xroutinv = mat.pod.inverse_transform(xrout)
    dx_xroutinv = (xroutinv - xroutinv[0])[:,:xroutinv.shape[1]//2]
    xx_xroutinv = mat.xx[0] + xroutinv - xroutinv[0]

    plt.figure(figsize=(20,6))
    plt.subplot(122)
    l1 = plt.plot(mat.t[::10],mat.xr[::10])[0]
    plt.gca().set_prop_cycle(None)
    l2 = plt.plot(mat.t[::10],xrout[::10],'x')[0]
    legend1 = plt.legend(handles=[pltl.Line2D([0], [0],
                                        marker='x',
                                        color=colors[i % len(colors)],
                                        label=rf"$\hat{{u}}_{{{i}}}$") for i in range(min((20,pod_basis_size)))],
                    loc="upper left",
                    bbox_to_anchor=(1.01,1.2))
    plt.legend([l1,l2],['reference', 'prediction'],framealpha=0.6,edgecolor="gray")

    plt.gca().add_artist(legend1)
    plt.title("Reduced components - comparison",wrap=True)
    plt.xlabel("ms")
    plt.ylabel("mm")
    #plt.subplot(121)
    #plt.plot(mat.t,((mat.xr[:]-xrout)**2).mean(axis=1)/(mat.xr.max()-mat.xr.min()))
    #plt.title("NMSE (reduced space)",wrap=True)
    #plt.xlabel("ms")
    #plt.ylabel(r"mm$^2$",labelpad=-40,y=1.2)
    plt.subplot(121)
    nmse = ((mat.xx-xx_xroutinv)**2).mean(axis=1)/(mat.xx.max()-mat.xx.min())
    plt.plot(mat.t,nmse)
    plt.title("NMSE (full space)",wrap=True);
    plt.xlabel("ms")
    plt.ylabel(r"mm$^2$")
    plt.suptitle("Performance of displacement fit during evolution")
    plt.tight_layout()

    roughplot2D(dx_xroutinv, xx_xroutinv[:,:xx_xroutinv.shape[1]//2],xx_xroutinv[:,xx_xroutinv.shape[1]//2:])
    plt.suptitle(rf"2D example, $u_x$ - prediction")

    dx_xx = (mat.xx - mat.xx[0])[:,:mat.xx.shape[1]//2]
    roughplot2D(dx_xx, mat.xx[:,:mat.xx.shape[1]//2],mat.xx[:,mat.xx.shape[1]//2:])
    plt.suptitle(rf"2D example, $u_x$ - reference")

    return {"xrout": xrout, "vrout":vrout, "xroutinv":xroutinv, "hist":hist, "xr": np.copy(mat.xr),
            'model':model, 'fit_history':fit_history, 'NMSE':nmse, 'pred_fint_r':pred_fint_r}