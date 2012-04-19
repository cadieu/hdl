import hdl
reload(hdl)

from hdl.models import SparseSlowModel
from hdl.hdl import HDL

from hdl.config import tstring

model_base_name = 'HDL_test1_infer/layer_%s'
timestring = tstring()

model_sequence = [
    SparseSlowModel(patch_sz=16, N=512, sparse_cost='subspacel1', slow_cost=None, tstring=timestring, model_name=model_base_name % '1'),
    SparseSlowModel(patch_sz=16, N=512, sparse_cost='subspacel1', slow_cost=None, perc_var=100., tstring=timestring, model_name=model_base_name % '2'),
    SparseSlowModel(patch_sz=16, N=512, sparse_cost='l1', slow_cost=None, perc_var=100., tstring=timestring, model_name=model_base_name % '3')]

hdl_learner  = HDL(model_sequence=model_sequence,datasource='berkeleysegmentation')

hdl_learner.learn()
