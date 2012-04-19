import hdl
reload(hdl)

from hdl.models import SparseSlowModel
from hdl.learners import SGD

whitenpatches = 160000

# Control:
#l = SGD(model=SparseSlowModel(patch_sz=32,N=1024,sparse_cost='l1',slow_cost=None))
#l = SGD(model=SparseSlowModel(patch_sz=8,N=64,sparse_cost='l1',slow_cost=None))
#l = SGD(model=SparseSlowModel(patch_sz=64,N=2048,sparse_cost='l1',slow_cost=None))

# Experiment with datasource:
#l = SGD(model=SparseSlowModel(patch_sz=8,N=64,T=64,sparse_cost='l1',slow_cost=None),datasource='vid075-chunks')

# Experiment with subspacel1 prior:
#l = SGD(model=SparseSlowModel(patch_sz=8,N=64,T=64,sparse_cost='subspacel1',slow_cost=None),datasource='vid075-chunks')
#l = SGD(model=SparseSlowModel(patch_sz=32,N=1024,T=64,sparse_cost='subspacel1',slow_cost=None),datasource='vid075-chunks')

# Experiment with subspacel1slow prior:
#l = SGD(model=SparseSlowModel(patch_sz=8,N=64,T=64,sparse_cost='subspacel1',slow_cost='dist'),datasource='vid075-chunks')
#l = SGD(model=SparseSlowModel(patch_sz=8,N=64,T=64,sparse_cost='subspacel1',slow_cost='dist',lam_slow=1.),datasource='vid075-chunks')
#l = SGD(model=SparseSlowModel(patch_sz=32,N=1024,T=64,sparse_cost='subspacel1',slow_cost='dist',perc_var=99.8),datasource='vid075-chunks')
#l = SGD(model=SparseSlowModel(patch_sz=64,N=2048,T=64,sparse_cost='subspacel1',slow_cost='dist'),datasource='vid075-chunks')
#l = SGD(model=SparseSlowModel(patch_sz=8,N=64,T=64,sparse_cost='subspacel1',slow_cost='dist',perc_var=99.8),datasource='vid075-chunks',save_every=100)

# Experiment with Ksubspacel1 prior:
#l = SGD(model=SparseSlowModel(patch_sz=16,N=1024,T=64,sparse_cost='Ksubspacel1',slow_cost=None,perc_var=99.8),datasource='vid075-chunks')

# Experiment with fista params
#l = SGD(model=SparseSlowModel(patch_sz=32,N=1028,T=64,sparse_cost='l1',slow_cost=None,fista_maxiter=1,u_init_method='proj'),datasource='vid075-chunks')
#self.get_databatch() time 0.000483
#self.model.gradient time 0.034545
#self.model.update_model time 0.009720
#self.get_databatch() time 0.000477
#self.model.gradient time 0.033651
#self.model.update_model time 0.011094
#self.get_databatch() time 0.000486
#self.model.gradient time 0.034500
#self.model.update_model time 0.009688

#l = SGD(model=SparseSlowModel(patch_sz=32,N=1028,T=64,sparse_cost='l1',slow_cost=None,fista_maxiter=1),datasource='vid075-chunks')
#self.get_databatch() time 0.000671
#self.model.gradient time 0.038676
#self.model.update_model time 0.010409
#self.get_databatch() time 0.000458
#self.model.gradient time 0.038264
#self.model.update_model time 0.010783
#self.get_databatch() time 0.000666
#self.model.gradient time 0.038689
#self.model.update_model time 0.010434
#self.get_databatch() time 0.000498
#self.model.gradient time 0.038317
#self.model.update_model time 0.010779

#l = SGD(model=SparseSlowModel(patch_sz=32,N=1028,T=64,sparse_cost='l1',slow_cost=None),datasource='vid075-chunks')
#self.get_databatch() time 0.000453
#self.model.gradient time 0.118604
#self.model.update_model time 0.011153
#self.get_databatch() time 0.000623
#self.model.gradient time 0.119030
#self.model.update_model time 0.010784
#self.get_databatch() time 0.000476
#self.model.gradient time 0.118870
#self.model.update_model time 0.009735
#self.get_databatch() time 0.000473
#self.model.gradient time 0.118527
#self.model.update_model time 0.011175

# Experiment with projection and maxiter=1
#l = SGD(model=SparseSlowModel(patch_sz=32,N=1024,T=64,sparse_cost='l1',slow_cost=None,fista_maxiter=1,u_init_method='proj'),datasource='vid075-chunks')

#Default
l = SGD(model=SparseSlowModel(patch_sz=16,N=256,T=64,sparse_cost='subspacel1',slow_cost='dist'),datasource='vid075-chunks')

databatch = l.get_databatch(whitenpatches)
l.model.learn_whitening(databatch)
l.model.setup()

l.learn(iterations=160000)
l.change_target(.5)
l.learn(iterations=20000)
l.change_target(.5)
l.learn(iterations=20000)

from hdl.display import display_final
display_final(l.model)
