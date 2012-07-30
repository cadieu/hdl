import hdl
reload(hdl)

from hdl.models import SparseSlowModel
from hdl.learners import SGD

whitenpatches = 150000

#l = SGD(model=SparseSlowModel(patch_sz=48,N=2048,T=48,sparse_cost='subspacel1',slow_cost='dist'),datasource='PLoS09_Cars_Planes',batchsize=48,display_every=20000,save_every=20000,eta_target_maxupdate=.05)
#l = SGD(model=SparseSlowModel(patch_sz=32,N=1028,T=64,sparse_cost='subspacel1',slow_cost='dist'),datasource='PLoS09_Cars_Planes',batchsize=64,display_every=20000,save_every=20000,eta_target_maxupdate=.05)
l = SGD(model=SparseSlowModel(patch_sz=20,N=400,T=48,sparse_cost='subspacel1',slow_cost='dist',perc_var=99.9),datasource='PLoS09_Cars_Planes',batchsize=48,display_every=1000,save_every=20000,eta_target_maxupdate=.05)

databatch = l.get_databatch(whitenpatches)
l.model.learn_whitening(databatch)
l.model.setup()

from kairos.data import PLoS09
#from kairos.metrics.kanalysis import kanalysis

from devthor.kanalysis import kanalysis_linear as kanalysis

X, Y = PLoS09.get_testing_selection(l.videos,l.chunk_labels)

def evaluate_kernel(model,X,Y,output_function='infer_abs'):

    offset_y = (X.shape[1] - model.patch_sz)/2
    offset_x = (X.shape[2] - model.patch_sz)/2
    X = X[:,offset_y:offset_y+model.patch_sz,offset_x:offset_x+model.patch_sz]

    batch = X.reshape(X.shape[0],X.shape[1]*X.shape[2]).T

    new_X = model.output(model.preprocess(batch),output_function=output_function).T

    k_curve, k_auc = kanalysis(new_X,Y)

    return k_curve, k_auc

def plot_kanalysis(l,X,Y):

    from matplotlib import pyplot as plt
    import os
    from hdl.config import state_dir

    output_functions = ['infer', 'infer_abs', 'proj', 'proj_abs', 'infer_loga', 'proj_loga']

    all_k_auc = []
    for output_function in output_functions:
        k_curve, k_auc = evaluate_kernel(l.model,X,Y,output_function)

        all_k_auc.append(k_auc)

        plt.figure(10)
        plt.clf()
        plt.plot(range(len(k_curve)),k_curve)
        plt.xlabel('Kernel dim')
        plt.ylabel('loss')
        plt.title('Kernel Analysis: %.4f'%k_auc)

        savepath = os.path.join(state_dir,l.model.model_name + '_' + l.model.tstring)
        if not os.path.exists(savepath): os.makedirs(savepath)

        fname = os.path.join(savepath,'KAnalysis_%s_%d.png'%(output_function,l.iter))
        plt.savefig(fname)

    return l.iter, min(all_k_auc)

def plot_k_results(k_results):

    from matplotlib import pyplot as plt
    import os
    from hdl.config import state_dir
    import numpy as np

    trials = np.array([k[0] for k in k_results])
    values = np.array([k[1] for k in k_results])

    plt.figure(10)
    plt.clf()
    plt.plot(trials,values)
    plt.xlabel('Save trial')
    plt.ylabel('KAnalysis AUC')
    plt.title('Kernel Analysis AUC')

    savepath = os.path.join(state_dir,l.model.model_name + '_' + l.model.tstring)
    if not os.path.exists(savepath): os.makedirs(savepath)

    fname = os.path.join(savepath,'KAnalysis_AUC_progress.png')
    plt.savefig(fname)


k_results = []

k_results.append(plot_kanalysis(l,X,Y))
plot_k_results(k_results)

epochs = 20
for epoch in range(epochs):
    l.learn(iterations=10000)
    k_results.append(plot_kanalysis(l,X,Y))
    plot_k_results(k_results)

l.change_target(.1)
epochs = 10
for epoch in range(epochs):
    l.learn(iterations=10000)
    k_results.append(plot_kanalysis(l,X,Y))
    plot_k_results(k_results)

l.change_target(.1)
epochs = 10
for epoch in range(epochs):
    l.learn(iterations=10000)
    k_results.append(plot_kanalysis(l,X,Y))
    plot_k_results(k_results)

from hdl.display import display_final
display_final(l.model)

