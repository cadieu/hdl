import os
import hdl
reload(hdl)

from hdl.models import SparseSlowModel, BinocColorModel
from hdl.config import state_dir
from hdl.display import display_final

if __name__ == '__main__':

    #model_name = 'SparseSlowModel_patchsz064_N2048_NN2048_l2_l1_None_2012-02-05_15-29-08/SparseSlowModel_patchsz064_N2048_NN2048_l2_l1_None.model'

    # faces
    #model_name = 'SparseSlowModel_patchsz032_N512_NN512_l2_l1_None_2012-02-09_11-40-37/SparseSlowModel_patchsz032_N512_NN512_l2_l1_None.model'
    #model_name = 'SparseSlowModel_patchsz064_N1024_NN1024_l2_l1_None_2012-02-09_11-47-43/SparseSlowModel_patchsz064_N1024_NN1024_l2_l1_None.model'
    #model_name = 'SparseSlowModel_patchsz032_N512_NN512_l2_l1_None_2012-02-09_16-38-31/SparseSlowModel_patchsz032_N512_NN512_l2_l1_None.model'
    #model_name = 'SparseSlowModel_patchsz032_N512_NN512_l2_l1_None_2012-02-09_19-05-45/SparseSlowModel_patchsz032_N512_NN512_l2_l1_None.model'
    model_name = 'SparseSlowModel_patchsz048_N768_NN768_l2_l1_None_2012-02-09_19-07-07/SparseSlowModel_patchsz048_N768_NN768_l2_l1_None.model'
    #model_name = 'SparseSlowModel_patchsz064_N1024_NN1024_l2_l1_None_2012-02-09_19-08-50/SparseSlowModel_patchsz064_N1024_NN1024_l2_l1_None.model'
    model_name = 'SparseSlowModel_patchsz048_N1024_NN1024_l2_l1_None_2012-02-10_16-23-20/SparseSlowModel_patchsz048_N1024_NN1024_l2_l1_None.model'
    model_name = 'SparseSlowModel_patchsz048_N1024_NN1024_l2_subspacel1_None_2012-02-10_17-22-55/SparseSlowModel_patchsz048_N1024_NN1024_l2_subspacel1_None.model'
    model_name = 'SparseSlowModel_patchsz048_N4096_NN4096_l2_l1_None_2012-03-01_19-15-33/SparseSlowModel_patchsz048_N4096_NN4096_l2_l1_None.model'
    model_name = 'SparseSlowModel_patchsz048_N512_NN512_l2_subspacel1_None_2012-03-05_11-42-48/SparseSlowModel_patchsz048_N512_NN512_l2_subspacel1_None.model'

    #fname = os.path.join(state_dir,model_name)
    #m = SparseSlowModel()
    #m.load(fname,reset_theano=False)

    model_name = 'BinocColorModel_patchsz016_N512_NN512_l2_subspacel1_dist_2012-03-13_15-24-27/BinocColorModel_patchsz016_N512_NN512_l2_subspacel1_dist.model'
    model_name = 'BinocColorModel_patchsz016_N512_NN512_l2_subspacel1_dist_2012-03-15_15-56-39/BinocColorModel_patchsz016_N512_NN512_l2_subspacel1_dist.model'
    model_name = 'BinocColorModel_patchsz020_N3200_NN3200_l2_subspacel1_dist_2012-03-18_17-35-13/BinocColorModel_patchsz020_N3200_NN3200_l2_subspacel1_dist.model'

    fname = os.path.join(state_dir,model_name)
    m = BinocColorModel()
    m.load(fname,reset_theano=False)

    display_final(m)

    print m

