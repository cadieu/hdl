import os
from copy import copy
import numpy as np
from fista import Fista
import theano

from config import state_dir, verbose, tstring, verbose_timing
from time import time

from matplotlib import pyplot as plt

# Models
class BaseModel(object):

    _params = ['patch_sz','D','tstring','model_name']

    def __init__(self,**kargs):
        self.patch_sz = kargs.get('patch_sz',4)
        self.tstring = kargs.get('tstring', tstring()) # used for saving
        self.model_name = kargs.get('model_name','BaseModel')
        if self.patch_sz:
            self.D = kargs.get('D',self.patch_sz*self.patch_sz)
        else:
            self.D = kargs.get('D',None)

    def __repr__(self):
        s = '='*30 + '\n'
        s += 'Model type: %s'%type(self) + '\n'
        for param in self._params:
            s += '-'*10  + '\n'
            s += '-- ' + param + '\n'
            p = getattr(self,param)
            if hasattr(p,'get_value'):
                p = p.get_value()
            s += str(p) + '\n'
        s += '-'*10 + '\n'

        return s

    def save(self,fname=None,save_name=None,ext='.model',save_txt=False,extra_info=None):
        """save model to disk
        """
        import cPickle
        if save_name is None: save_name = self.model_name
        if fname is None:
            savepath = os.path.join(state_dir,save_name + '_' + self.tstring)
            if not os.path.isdir(savepath): os.makedirs(savepath)
            fname = os.path.join(savepath, 'model'+ext)
        else:
            savepath = os.path.split(fname)[0]

        sdict = {}
        for sname in self._params:
            save_prop = getattr(self,sname)
            if hasattr(save_prop,'get_value'): save_prop = save_prop.get_value() # in case we have CudaNdarrayType -> NP
            sdict[sname] = save_prop
        fh = open(fname,'wb')
        cPickle.dump(sdict,fh)
        fh.close()
        if verbose: print "saving model to", fname

        if save_txt:
            repr_string = self.__repr__()
            if not extra_info is None:
                repr_string += '\n'
                if isinstance(extra_info,str):
                    repr_string += extra_info
                elif isinstance(extra_info,list):
                    for extra_item in extra_info:
                        repr_string += str(extra_item)

            model_details_fname = os.path.join(savepath,'model_details.txt')
            with open(model_details_fname,'w') as fh:
                fh.write(repr_string)

        return fname

    def load(self,fname,replace_name=True,reset_theano=True):
        """
        load model from disk

        this load method should be the default load method for the Model class
        (once all models are updated to use the new config objects).
        """
        import cPickle

        noreplace_list = ['model_name','tstring']

        with open(fname,'rb') as fh:
            ldict = cPickle.load(fh)
            for sname in self._params:
                if ldict.has_key(sname):
                    if replace_name:
                        setattr(self,sname,ldict[sname])
                    else:
                        if sname not in noreplace_list:
                            setattr(self,sname,ldict[sname])
                else:
                    print 'WARNING: key %s missing in file %s'%(sname,fname)

        if reset_theano: self._reset_on_load()

    def display(self,save_string=None,save=True,normalize_A=True,max_factors=100,zerophasewhiten=True):
        print 'display not implemented'

    def _reset_on_load(self):
        pass

class WhitenInputModel(BaseModel):
    _params = copy(BaseModel._params)
    _params.extend(['M','whiten','inputmean','whitenmatrix', 'dewhitenmatrix','zerophasewhitenmatrix','num_eigs','perc_var'])

    def __init__(self, **kargs):
        super(WhitenInputModel, self).__init__(**kargs)

        self.whiten = kargs.get('whiten',True)

        if self.whiten:
            self.inputmean = None
            self.whitenmatrix = None
            self.dewhitenmatrix = None
            self.zerophasewhitenmatrix = None
            self.num_eigs = kargs.get('num_eigs',None)
            self.perc_var = kargs.get('perc_var', 99.)
            self.M = None
        else:
            self.M = self.D

    def learn_whitening(self,patches):
        from utils import whiten_var

        wpatches, inputmean, whitenmatrix, dewhitenmatrix, zerophasewhitenmatrix, zerophasedewhitenmatrix = whiten_var(patches,num_eigs=self.num_eigs,perc_var=self.perc_var)

        self.inputmean = inputmean
        self.whitenmatrix = whitenmatrix
        self.dewhitenmatrix = dewhitenmatrix
        self.zerophasewhitenmatrix = zerophasewhitenmatrix

        self.M = self.whitenmatrix.shape[0]
        if self.D is None:
            self.D = patches.shape[0]

    def preprocess(self,batch):
        batch -= self.inputmean
        return np.dot(self.whitenmatrix,batch).astype(theano.config.floatX)

    def display_whitening(self,save_string=None,save=True,normalize_A=True,max_factors=256,zerophasewhiten=True):
        output = {}
        if zerophasewhiten:
            A = self.zerophasewhitenmatrix
        else:
            A = self.whitenmatrix
        if hasattr(A,'get_value'):
            A = A.get_value()

        A = A.T # make the columns the filters

        if self.patch_sz:
            psz = self.patch_sz
        else:
            psz = int(np.ceil(np.sqrt(A.shape[0])))
            if not psz**2 == A.shape[0]:
                A = np.vstack((A,np.zeros((psz**2 - A.shape[0],A.shape[1]))))

        # plot the vectors in A
        NN = min(A.shape[1],max_factors)
        buf = 1
        sz = int(np.sqrt(NN))
        hval = np.max(np.abs(A))
        array = -np.ones(((psz+buf)*sz+buf,(psz+buf)*sz+buf))
        Aind = 0
        for r in range(sz):
            for c in range(sz):
                if normalize_A:
                    hval = np.max(np.abs(A[:,Aind]))
                Avalues = A[:,Aind].reshape(psz,psz)/hval
                array[buf+(psz+buf)*c:buf+(psz+buf)*c+psz,buf+(psz+buf)*r:buf+(psz+buf)*r+psz] = Avalues
                Aind += 1
        hval = 1.
        plt.figure(1)
        plt.clf()
        plt.imshow(array,vmin=-hval,vmax=hval,interpolation='nearest',cmap=plt.cm.gray)
        plt.colorbar()

        output['whitenmatrix'] = array
        plt.draw()

        if save:
            from config import state_dir, tstring
            savepath = os.path.join(state_dir,self.model_name + '_' + self.tstring)
            if not os.path.isdir(savepath): os.makedirs(savepath)

            if save_string is None:
                save_string = tstring()
            plt.figure(1)
            fname = os.path.join(savepath, 'whitenmatrix_' + save_string + '.png')
            plt.savefig(fname)

        return output

class SparseSlowModel(WhitenInputModel):
    """
    Model class for a sparse and slow model
    m = SparseSlowModel(patch_sz, M, N, L, K, rec_cost, sparse_cost, slow_cost)

        patch_sz - image patch side length
        M - size of whitened space
        N - number of subspaces
        T - default temporal length

        rec_cost - reconstruction cost (l2)
        sparse_cost - sparse cost on subspace amplitude (l1)
        slow_cost - cost on time derivative of amplitude (dist)

    """
    _params = copy(WhitenInputModel._params)
    _params.extend(['N','NN','T','A','inference_method','inference_params','lam_sparse','lam_slow','lam_l2','rec_cost','sparse_cost','slow_cost'])

    def __init__(self, **kargs):

        super(SparseSlowModel, self).__init__(**kargs)

        self.N = kargs.get('N',100)
        self.NN = self.N
        self.T = kargs.get('T',128)
        self.rec_cost = kargs.get('rec_cost','l2')
        self.sparse_cost = kargs.get('sparse_cost','l1')
        self.slow_cost = kargs.get('slow_cost','dist')

        self.lam_sparse = theano.shared(getattr(np,theano.config.floatX)(kargs.get('lam_sparse',.1)))
        self.lam_slow = theano.shared(getattr(np,theano.config.floatX)(kargs.get('lam_slow',.1)))
        self.lam_l2 = theano.shared(getattr(np,theano.config.floatX)(kargs.get('lam_l2',.1)))

        self.K = kargs.get('K',8)

        self.whiten = True

        self.inference_method = 'FISTA'
        self.inference_params = {'u_init_method': kargs.get('u_init_method','rand'),
                                 'FISTAargs': {'maxiter': kargs.get('fista_maxiter', 20),
                                               'maxline': kargs.get('fista_maxline', 10),
                                               'errthres': 1e-5,
                                               #'L':1., # default .1
                                               'verbose': False}}

        self.setup_complete = False

        if self.patch_sz:
            self.model_name = kargs.get('model_name','SparseSlowModel_patchsz%03d_N%03d_NN%03d_%s_%s_%s'%(self.patch_sz, self.N, self.NN, self.rec_cost, self.sparse_cost, self.slow_cost))
        else:
            self.model_name = kargs.get('model_name','SparseSlowModel_N%03d_NN%03d_%s_%s_%s'%(self.N, self.NN, self.rec_cost, self.sparse_cost, self.slow_cost))


        if self.slow_cost == 'dist' and self.T < 2:
            raise ValueError, 'self.T must be greater than or equal to 2 with slow_cost = dist'
        #self.setup()

    def _reset_on_load(self):
        self.lam_sparse = theano.shared(getattr(np,theano.config.floatX)(self.lam_sparse))
        self.lam_slow = theano.shared(getattr(np,theano.config.floatX)(self.lam_slow))
        self.lam_l2 = theano.shared(getattr(np,theano.config.floatX)(self.lam_l2))
        self.A = theano.shared(self.A.astype(theano.config.floatX))

        self.setup(init=False)

    def reset_functions(self):
        self.setup(init=False)

    def setup(self,init=True):
        """Setup model functions"""

        if init:
            A = self.normalize_A(np.random.randn(self.M,self.NN))
            self.A = theano.shared(A.astype(theano.config.floatX))

        # Inference
        if self.rec_cost == 'l2' and self.sparse_cost == 'subspacel1' and self.slow_cost == 'dist':
            print 'Use l2subspacel1slow problem for Fista'
            #setup theano inputs to pass to setup:
            self.x = theano.shared(np.random.randn(self.M, self.T).astype(theano.config.floatX))
            self.u = np.random.randn(self.NN,self.T).astype(theano.config.floatX)

            self._fista = Fista(xinit=self.u,A=self.A,lam_sparse=self.lam_sparse,lam_slow=self.lam_slow,x=self.x,problem_type='l2subspacel1slow')

        elif self.rec_cost == 'l2' and self.sparse_cost == 'subspacel1':
            print 'Use l2subspacel1 problem for Fista'
            #setup theano inputs to pass to setup:
            self.x = theano.shared(np.random.randn(self.M, self.T).astype(theano.config.floatX))
            self.u = np.random.randn(self.NN,self.T).astype(theano.config.floatX)

            self._fista = Fista(xinit=self.u,A=self.A,lam_sparse=self.lam_sparse,x=self.x,problem_type='l2subspacel1')

        elif self.rec_cost == 'l2' and self.sparse_cost == 'Ksubspacel1':
            print 'Use l2Ksubspacel1 problem for Fista'
            #setup theano inputs to pass to setup:
            self.x = theano.shared(np.random.randn(self.M, self.T).astype(theano.config.floatX))
            self.u = np.random.randn(self.NN,self.T).astype(theano.config.floatX)

            self._fista = Fista(xinit=self.u,A=self.A,lam_sparse=self.lam_sparse,x=self.x,K=self.K,problem_type='l2Ksubspacel1')

        elif self.rec_cost == 'l2' and self.sparse_cost == 'elastic':
            print 'Use l2l1 problem for Fista'
            #setup theano inputs to pass to setup:
            self.x = theano.shared(np.random.randn(self.M, self.T).astype(theano.config.floatX))
            self.u = np.random.randn(self.NN,self.T).astype(theano.config.floatX)

            self._fista = Fista(xinit=self.u,A=self.A,lam_sparse=self.lam_sparse,lam_l2=self.lam_l2,x=self.x,problem_type='l2elastic')

        elif self.rec_cost == 'l2' and self.sparse_cost == 'l1':
            print 'Use l2l1 problem for Fista'
            #setup theano inputs to pass to setup:
            self.x = theano.shared(np.random.randn(self.M, self.T).astype(theano.config.floatX))
            self.u = np.random.randn(self.NN,self.T).astype(theano.config.floatX)

            self._fista = Fista(xinit=self.u,A=self.A,lam=self.lam_sparse,x=self.x)

        else:
            raise NotImplementedError, 'Inference for rec_cost %s, sparse_cost %s, slow_cost %s'%(self.rec_cost, self.sparse_cost, self.slow_cost)

        # Update gradients:
        if self.rec_cost == 'l2':
            from theano_methods import T_l2_cost, T_l2_cost_norm
            x = theano.tensor.matrix('x')
            u = theano.tensor.matrix('u')
            grad_A = theano.tensor.grad(T_l2_cost_norm(x,u,self.A),self.A)
            self._df_dA = theano.function([x,u],grad_A)
            #grad_A = grad_A.flatten()
            #grad2, _ = theano.scan(fn=lambda i, A, grad_A: theano.tensor.grad(grad_A[i], A).flatten()[i],
            #    sequences=theano.tensor.arange(grad_A.shape[0]), non_sequences=[self.A, grad_A])
            #self._df_dA2 = theano.function([x,u],grad2)
        else:
            raise NotImplementedError, 'rec_cost unknown %s'%self.rec_cost

        self.setup_complete = True

    def gradient(self,batch):

        if not self.setup_complete: raise AssertionError, 'Please call model.setup() before processing'

        t0 = time()
        batch = self.preprocess(batch)
        if verbose_timing: print 'self.preprocess time %f'%(time() - t0)

        t0 = time()
        u = self.inferlatent(batch)
        if verbose_timing: print 'self.inferlatent time %f'%(time() - t0)

        t0 = time()
        modelgradient = self.calc_modelgradient(batch,u)
        if verbose_timing: print 'self.calc_modelgradient time %f'%(time() - t0)

        return modelgradient

    def calc_modelgradient(self,batch,u):

        grad_dict = dict(dA=self._df_dA(batch,u))

        return grad_dict

    def update_model(self,update_dict,eta=None):
        """ update model parameters with update_dict['dA']
        returns the maximum update percentage max(dA/A)
        """

        A = self.A.get_value()
        param_max = np.max(np.abs(A),axis=0)

        if eta is None:
            update_max = np.max(np.abs(update_dict['dA']),axis=0)
            A -= update_dict['dA']
        else:
            update_max = eta*np.max(np.abs(update_dict['dA']),axis=0)
            A -= eta*update_dict['dA']

        update_max = np.max(update_max/param_max)

        A = self.normalize_A(A)
        self.A.set_value(A)

        return update_max

    def normalize_A(self,A):

        Anorm = np.sqrt((A**2).sum(axis=0)).reshape(1,self.NN)
        return A/Anorm

    def __call__(self,batch,output_function='infer_abs',chunk_size=None):

        return self.output(self.preprocess(batch),
            output_function=output_function,chunk_size=chunk_size)

    def output(self,batch,output_function='infer_abs',chunk_size=None):

        bsz = batch.shape[1]
        if chunk_size is None:
            chunk_size = self.T

        if bsz > chunk_size:
            u_out = None
            done = False
            t0 = 0
            while not done:
                u_minibatch = self._output_minibatch(batch[:,t0:t0+chunk_size],output_function)
                if u_out is None:
                    u_out = np.zeros((u_minibatch.shape[0],bsz))
                u_out[:,t0:t0+chunk_size] = u_minibatch
                t0 += chunk_size
                if t0 >= bsz: done = True
            return u_out
        else:
            return self._output_minibatch(batch,output_function)

    def _output_minibatch(self,batch,output_function):
        if output_function == 'infer':
            return self.inferlatent(batch)
        elif output_function == 'proj':
            return np.dot(self.A.get_value().T,batch)
        elif output_function == 'infer_abs':
            u = self.inferlatent(batch)
            return np.sqrt(u[::2,:]**2 + u[1::2,:]**2)
        elif output_function == 'proj_abs':
            u = np.dot(self.A.get_value().T,batch)
            return np.sqrt(u[::2,:]**2 + u[1::2,:]**2)
        elif output_function == 'infer_loga':
            u = self.inferlatent(batch)
            amp = np.sqrt(u[::2,:]**2 + u[1::2,:]**2)
            return np.log(amp + .01)
        elif output_function == 'proj_loga':
            u = np.dot(self.A.get_value().T,batch)
            amp = np.sqrt(u[::2,:]**2 + u[1::2,:]**2)
            return np.log(amp + .01)
        else:
            assert NotImplemented, 'Unknown output_function %s'%output_function

    def __initu(self,batch):

        if self.inference_params['u_init_method'] == 'proj':
            return np.dot(self.A.get_value().T,batch)
        elif self.inference_params['u_init_method'] == 'rand':
            return .1*np.random.randn(self.NN, batch.shape[1]).astype(theano.config.floatX)
        else:
            raise NotImplementedError

    def inferlatent(self,batch):
        """

        """
        if not self.inference_method == 'FISTA':
            raise NotImplementedError

        # set shared variables
        t0 = time()
        u0 = self.__initu(batch)
        if verbose_timing: print 'self.__initu time %f'%(time() - t0)

        # run inference
        uhat, fista_history = self._fista(u0, batch, **self.inference_params['FISTAargs'])

        # return
        return uhat

    def display(self,save_string=None,save=True,normalize_A=True,max_factors=256,zerophasewhiten=True):
        output = {}
        if hasattr(self.A,'get_value'):
            A = self.A.get_value()
        else:
            A = self.A

        if self.whiten:
            A = np.dot(self.dewhitenmatrix,A)
            if zerophasewhiten:
                A = np.dot(self.zerophasewhitenmatrix,A)

        if self.patch_sz:
            psz = self.patch_sz
        else:
            psz = int(np.ceil(np.sqrt(A.shape[0])))
            if not psz**2 == A.shape[0]:
                A = np.vstack((A,np.zeros((psz**2 - A.shape[0],A.shape[1]))))

        # plot the vectors in A
        NN = min(self.NN,max_factors)
        buf = 1
        sz = int(np.sqrt(NN))
        hval = np.max(np.abs(A))
        array = -np.ones(((psz+buf)*sz+buf,(psz+buf)*sz+buf))
        Aind = 0
        for r in range(sz):
            for c in range(sz):
                if normalize_A:
                    hval = np.max(np.abs(A[:,Aind]))
                Avalues = A[:,Aind].reshape(psz,psz)/hval
                array[buf+(psz+buf)*c:buf+(psz+buf)*c+psz,buf+(psz+buf)*r:buf+(psz+buf)*r+psz] = Avalues
                Aind += 1
        hval = 1.
        plt.figure(1)
        plt.clf()
        plt.imshow(array,vmin=-hval,vmax=hval,interpolation='nearest',cmap=plt.cm.gray)
        plt.colorbar()

        output['A'] = array

        plt.draw()

        if save:
            from config import state_dir, tstring
            savepath = os.path.join(state_dir,self.model_name + '_' + self.tstring)
            if not os.path.isdir(savepath): os.makedirs(savepath)

            if save_string is None:
                save_string = tstring()
            plt.figure(1)
            fname = os.path.join(savepath, 'A_' + save_string + '.png')
            plt.savefig(fname)

        return output

class BinocColorModel(SparseSlowModel):

    _params = copy(SparseSlowModel._params)
    _params.extend(['binoc_movie_mse_reject'])

    def __init__(self, **kargs):

        patch_sz = kargs.get('patch_sz',8)
        kargs['D'] = 6*patch_sz*patch_sz

        super(BinocColorModel, self).__init__(**kargs)

        self.binoc_movie_mse_reject = kargs.get('binoc_movie_mse_reject',None)

        self.model_name = kargs.get('model_name','BinocColorModel_patchsz%03d_N%03d_NN%03d_%s_%s_%s'%(self.patch_sz, self.N, self.NN, self.rec_cost, self.sparse_cost, self.slow_cost))

        #self.setup()

    def display_whitening(self,save_string=None,save=True,normalize_A=True,max_factors=64,zerophasewhiten=True):

        if self.patch_sz is None:
            raise NotImplemented

        from display import display_color_patches

        output = {}
        if zerophasewhiten:
            A = self.zerophasewhitenmatrix
        else:
            A = self.whitenmatrix
        if hasattr(A,'get_value'):
            A = A.get_value()

        A = A.T # make the columns the filters

        # plot the vectors in A
        NN = min(A.shape[1],2*max_factors)
        patches = np.zeros((self.patch_sz,self.patch_sz,3,2*NN))
        count = 0
        for nn in range(NN):
            patches[...,count  ] = A[...,nn].reshape(6,self.patch_sz,self.patch_sz)[:3,...].T
            patches[...,count+1] = A[...,nn].reshape(6,self.patch_sz,self.patch_sz)[3:,...].T
            count += 2

        patches = patches.reshape((self.patch_sz*self.patch_sz*3,2*NN))
        array = display_color_patches(patches,self.patch_sz,fig_num=1,normalize=normalize_A)

        output['whitenmatrix'] = array
        plt.draw()

        if save:
            from config import state_dir, tstring
            savepath = os.path.join(state_dir,self.model_name + '_' + self.tstring)
            if not os.path.isdir(savepath): os.makedirs(savepath)

            if save_string is None:
                save_string = tstring()
            plt.figure(1)
            fname = os.path.join(savepath, 'whitenmatrix_' + save_string + '.png')
            plt.savefig(fname)

        return output

    def display(self,save_string=None,save=True,normalize_A=True,max_factors=64,zerophasewhiten=True):

        if self.patch_sz is None:
            raise NotImplemented

        from display import display_color_patches

        output = {}
        if hasattr(self.A,'get_value'):
            A = self.A.get_value()
        else:
            A = self.A

        if self.whiten:
            A = np.dot(self.dewhitenmatrix,A)
            if zerophasewhiten:
                A = np.dot(self.zerophasewhitenmatrix,A)

        # plot the vectors in A
        NN = min(self.NN,2*max_factors)

        patches = np.zeros((self.patch_sz,self.patch_sz,3,2*NN))
        count = 0
        for nn in range(NN):
            if normalize_A:
                Ann = A[...,nn]
                Ann /= np.abs(Ann).max() / 127.5
                Ann += 127.5
            else:
                Ann = A[...,nn]
            patches[...,count  ] = Ann.reshape(6,self.patch_sz,self.patch_sz)[:3,...].T
            patches[...,count+1] = Ann.reshape(6,self.patch_sz,self.patch_sz)[3:,...].T
            count += 2

        patches = patches.reshape((self.patch_sz*self.patch_sz*3,2*NN))

        array = display_color_patches(patches,self.patch_sz,fig_num=1,normalize=False)

        output['A'] = array

        plt.draw()

        if save:
            from config import state_dir, tstring
            savepath = os.path.join(state_dir,self.model_name + '_' + self.tstring)
            if not os.path.isdir(savepath): os.makedirs(savepath)

            if save_string is None:
                save_string = tstring()
            plt.figure(1)
            fname = os.path.join(savepath, 'A_' + save_string + '.png')
            plt.savefig(fname)

        return output

class HierarchicalModel(object):

    def __init__(self,model_sequence,layer_params):

        self.model_sequence = model_sequence
        self.patch_sz = self.model_sequence[0].patch_sz

        self.layer_params = layer_params

        self.model_name = self.model_sequence[-1].model_name
        self.tstring = self.model_sequence[-1].tstring

    def __call__(self, batch, output_function=None, chunk_size=None):

        for mind, m in enumerate(self.model_sequence):
            if (mind + 1 == len(self.model_sequence)) and (not output_function is None):
                batch = m.output(m.preprocess(batch),output_function,chunk_size=chunk_size)
            else:
                batch = m.output(m.preprocess(batch),self.layer_params[mind]['output_function'],chunk_size=chunk_size)

        return batch

    def __repr__(self):
        return self.model_sequence[-1].__repr__()