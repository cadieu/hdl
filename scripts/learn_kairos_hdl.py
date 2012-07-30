import os
import hdl
reload(hdl)

from hdl.models import SparseSlowModel
from hdl.hdl import HDL

from hdl.config import tstring, state_dir

timestring = tstring()
model_base_name = 'HDL_loga_' + timestring + '/layer_%s'

m1 = SparseSlowModel()
layer1_name = 'SparseSlowModel_patchsz020_N512_NN512_l2_subspacel1_dist_2012-05-24_17-33-06/model.model'
fname = os.path.join(state_dir,layer1_name)
m1.load(fname)
m1.model_name = model_base_name % '1'

model_sequence = [
    m1,
    SparseSlowModel(patch_sz=None, N=512,  T=48, sparse_cost='subspacel1', slow_cost='dist', perc_var=99.9, tstring=timestring, model_name=model_base_name % '2'),
    SparseSlowModel(patch_sz=None, N=256,  T=48, sparse_cost='l1', slow_cost=None, perc_var=99.9, tstring=timestring, model_name=model_base_name % '3')]

hdl_learner  = HDL(model_sequence=model_sequence,datasource='PLoS09_Cars_Planes',output_function='proj_loga')

hdl_learner.learn(layer_start=1)
