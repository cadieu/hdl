import hdl
reload(hdl)

from hdl.models import SparseSlowModel
from hdl.hdl import HDL

from hdl.config import tstring

timestring = tstring()
model_base_name = 'HDL_loga_' + timestring + '/layer_%s'

#model_sequence = [
#    SparseSlowModel(patch_sz=16,   N=200, sparse_cost='subspacel1', slow_cost='dist', tstring=timestring, model_name=model_base_name % '1'),
#    SparseSlowModel(patch_sz=None, N=300, sparse_cost='subspacel1', slow_cost='dist', perc_var=99., tstring=timestring, model_name=model_base_name % '2'),
#    SparseSlowModel(patch_sz=None, N=100, sparse_cost='l1', slow_cost=None, perc_var=99., tstring=timestring, model_name=model_base_name % '3')]

#model_sequence = [
#    SparseSlowModel(patch_sz=32, N=2048, sparse_cost='subspacel1', slow_cost='dist', tstring=timestring, model_name=model_base_name % '1'),
#    SparseSlowModel(patch_sz=32, N=2048, sparse_cost='subspacel1', slow_cost='dist', perc_var=99., tstring=timestring, model_name=model_base_name % '2'),
#    SparseSlowModel(patch_sz=32, N=2048, sparse_cost='l1', slow_cost=None, perc_var=99., tstring=timestring, model_name=model_base_name % '3')]
#
#hdl_learner  = HDL(model_sequence=model_sequence,datasource='vid075-chunks',output_function='infer_loga')

model_sequence = [
    SparseSlowModel(patch_sz=48,   N=512, T=48, sparse_cost='subspacel1', slow_cost=None, tstring=timestring, model_name=model_base_name % '1'),
    SparseSlowModel(patch_sz=None, N=256,  T=48, sparse_cost='subspacel1', slow_cost=None, perc_var=99., tstring=timestring, model_name=model_base_name % '2'),
    SparseSlowModel(patch_sz=None, N=128,  T=48, sparse_cost='l1', slow_cost=None, perc_var=99., tstring=timestring, model_name=model_base_name % '3')]

hdl_learner  = HDL(model_sequence=model_sequence,datasource='YouTubeFaces_aligned',output_function='proj_loga')

hdl_learner.learn()
