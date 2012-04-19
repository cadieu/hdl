import hdl
reload(hdl)

from hdl.models import BinocColorModel
from hdl.learners import SGD

whitenpatches = 160000

#l = SGD(model=BinocColorModel(patch_sz=32,N=1024,T=48,sparse_cost='subspacel1',slow_cost='dist'),datasource='3Dvideo_color',batchsize=48,display_every=20000)
#l = SGD(model=BinocColorModel(patch_sz=16,N=512,T=48,sparse_cost='subspacel1',slow_cost='dist',lam_sparse=.3,perc_var=99.9),datasource='3Dvideo_color',batchsize=48,display_every=10000)
#l = SGD(model=BinocColorModel(patch_sz=16,N=2048,T=48,sparse_cost='subspacel1',slow_cost='dist',lam_sparse=1.2,perc_var=99),datasource='3Dvideo_color',batchsize=48,display_every=10000)
#l = SGD(model=BinocColorModel(patch_sz=20,N=3200,T=48,sparse_cost='subspacel1',slow_cost='dist',lam_sparse=1.2,perc_var=99),datasource='3Dvideo_color',batchsize=48,display_every=10000)
#l = SGD(model=BinocColorModel(patch_sz=20,N=3200,T=48,sparse_cost='subspacel1',slow_cost='dist',lam_sparse=1.2,lam_slow=1.0,perc_var=99),datasource='3Dvideo_color',batchsize=48,display_every=10000)
l = SGD(model=BinocColorModel(patch_sz=20,N=3200,T=48,
    sparse_cost='subspacel1',slow_cost='dist',
    lam_sparse=1.2,lam_slow=.1,perc_var=99,
    binoc_movie_mse_reject = 300.),datasource='3Dvideo_color',batchsize=48,display_every=10000)

databatch = l.get_databatch(whitenpatches)
l.model.learn_whitening(databatch)
l.model.setup()

l.learn(iterations=360000)
l.change_target(.5)
l.learn(iterations=20000)
l.change_target(.5)
l.learn(iterations=20000)

from display_results import display_final
display_final(l.model)
