import numpy as np
import os
from config import verbose, verbose_timing
import time

import theano

from collections import defaultdict

from IPython import parallel

import socket
import zmq

import learners

class SGD(learners.SGD):

    def __init__(self,**kargs):
        super(SGD, self).__init__(**kargs)

        self.ipython_profile = kargs.get('ipython_profile','nodb')

        self.parallel_initialized = False

    def init_parallel(self):

        # create client & view
        rc = parallel.Client(profile=self.ipython_profile)
        dv = rc[:]
        ids = rc.ids
        # scatter 'id', so id=0,1,2 on engines 0,1,2
        dv.scatter('id', rc.ids, flatten=True)
        print("Engine IDs: ", dv['id'])

        with dv.sync_imports():
            import os

        def check_theano_environ():
            global os
            return os.environ['THEANO_FLAGS']

        os.environ['THEANO_FLAGS']='mode=FAST_RUN,floatX=float32'
        print 'local env:'
        print check_theano_environ()

        import theano
        print theano.config.mode, theano.config.device, theano.config.floatX

        def check_pbs_environ():
            import os
            return os.environ['PBS_O_HOST'], os.environ['PBS_TASKNUM'], os.environ['PBS_NODENUM'], os.environ['PBS_VNODENUM']

        pbs = dv.apply(check_pbs_environ)
        pbs_info = pbs.get()

        if verbose: print(pbs_info)

        self.num_engines = len(ids)
        if verbose: print 'num_engines', self.num_engines

        with dv.sync_imports():
            import theano

        def check_gpu():
            import theano
            return theano.config.mode, theano.config.device, theano.config.floatX

        rs = dv.apply(check_gpu)
        theano_gpu_info = rs.get()
        if verbose: print(theano_gpu_info)

        node_gpu = defaultdict(list)
        for pbs_ind, pbs_item in enumerate(pbs_info):
            node_gpu[pbs_item[2]].append(theano_gpu_info[pbs_ind][1])

        if verbose:
            for node_ind in sorted(node_gpu.keys()):
                print 'Node:', node_ind
                print 'GPUS:', sorted(node_gpu[node_ind])


        # Setup zmq ports:
        local_interface = socket.gethostbyname(socket.gethostname())
        self.zmq_vent_port = 62000 + int(np.random.rand()*1000)
        self.zmq_vent_address = 'tcp://%s:%s'%(local_interface,str(self.zmq_vent_port))
        self.zmq_sink_port = self.zmq_vent_port + 1
        self.zmq_sink_address = 'tcp://%s:%s'%(local_interface,str(self.zmq_sink_port))
        context = zmq.Context(io_threads=4)

        # Socket to send messages on
        sender = context.socket(zmq.PUSH)
        sender.bind("tcp://*:%s"%str(self.zmq_vent_port))
        receiver = context.socket(zmq.PULL)
        receiver.bind("tcp://*:%s"%str(self.zmq_sink_port))

        self.context = context
        self.sender = sender
        self.receiver = receiver

        self.ipython_dv = dv
        self.ipython_rc = rc
        self.ipython_ids = ids

    def init_remote_models(self):

        # Import the modules we need:
        with self.ipython_dv.sync_imports():
            import numpy as np
            import time
            import theano

        self._msg_sz = self.model.A.get_value().shape
        self._msg_dtype = str(self.model.A.get_value().dtype)

        #self.ipython_dv.execute("from hdl.models import SparseSlowModel")
        #self.ipython_dv.execute("from hdl.learners import SGD")
        #r = self.ipython_dv.execute("l = SGD(model=SparseSlowModel())")
        #self.ipython_dv.wait()
        #print r.get()

        def setup_model(patch_sz,M,N,NN,D,T,
                        sparse_cost,slow_cost,
                        lam_sparse,lam_slow,lam_l2,
                        inputmean,whitenmatrix,dewhitenmatrix,zerophasewhitenmatrix,A,
                        datasource,batchsize):

            import numpy as np
            import theano
            from hdl.models import SparseSlowModel
            from hdl.learners import SGD

            global l

            l = SGD(model=SparseSlowModel())

            l.model.patch_sz = patch_sz
            l.model.M = M
            l.model.D = D
            l.model.N = N
            l.model.NN = NN
            l.model.T = T
            l.model.sparse_cost = sparse_cost
            l.model.slow_cost = slow_cost
            l.model.inputmean = inputmean
            l.model.whitenmatrix = whitenmatrix
            l.model.dewhitenmatrix = dewhitenmatrix
            l.model.zerophasewhitenmatrix = zerophasewhitenmatrix

            l.datasource = datasource
            l.batchsize = batchsize

            old_type = type(A)
            l.model.A = theano.shared(A.astype(theano.config.floatX))
            l.model.lam_sparse = theano.shared(getattr(np,theano.config.floatX)(lam_sparse))
            l.model.lam_slow = theano.shared(getattr(np,theano.config.floatX)(lam_slow))
            l.model.lam_l2 = theano.shared(getattr(np,theano.config.floatX)(lam_l2))
            #l.model._reset_on_load()
            new_type = type(l.model.A)
            l.model.setup(init=False)
            return old_type, new_type, type(l.model.lam_sparse), l.model.lam_sparse.get_value()

        def check_model():
            return type(l.model.A)

        if verbose: print("setup_model(...)")
        r = self.ipython_dv.apply(setup_model,
            self.model.patch_sz,self.model.M,self.model.N,self.model.NN,self.model.D,self.model.T,
            self.model.sparse_cost,self.model.slow_cost,
            self.model.lam_sparse.get_value(),self.model.lam_slow.get_value(),self.model.lam_l2.get_value(),
            self.model.inputmean,self.model.whitenmatrix,self.model.dewhitenmatrix,self.model.zerophasewhitenmatrix,self.model.A.get_value(),
            self.datasource,self.batchsize)
        r.wait()
        print(r.get())
        if verbose: print("check_model")
        r = self.ipython_dv.apply(check_model)
        r.wait()
        print(r.get())

        if verbose:
            print 'msg_sz', self._msg_sz
            print 'msg_dtype', self._msg_dtype


    def __del__(self):

        def cleanup_worker():
            global sender, receiver, receiver_stream

            # IPython Engines will die if we call iolooper.stop() !
            #from zmq.eventloop import ioloop
            #iolooper = ioloop.IOLoop.instance()
            #iolooper.stop()

            receiver_stream.stop_on_recv()
            receiver.close()
            sender.close()

            return True

        if hasattr(self,'context'):
            #while self.receiver.poll(.01):
            #    self.receiver.recv()

            if verbose: print 'Cleanup:'
            self.ipython_dv.apply(cleanup_worker)

            self.sender.close()
            self.receiver.close()


    def init_remote_zmq_workers(self):

        def worker(vent_address,sink_address,sz,dtype):

            import zmq
            import theano
            from zmq.eventloop import ioloop
            ioloop.install()
            from zmq.eventloop.zmqstream import ZMQStream

            # Context
            context = zmq.Context()

            # Socket to receive messages on
            receiver = context.socket(zmq.PULL)
            receiver.connect(vent_address)
            receiver_stream = ZMQStream(receiver)

            # Socket to send messages to
            sender = context.socket(zmq.PUSH)
            sender.connect(sink_address)

            def _worker(msg_list,sz=sz,dtype=dtype,sender=sender):
                import theano
                import numpy as np

                msg = msg_list[0]

                new_A = np.frombuffer(buffer(msg), dtype=dtype).reshape(sz)#.copy()
                new_A = l.model.normalize_A(new_A)

                l.model.A.set_value(new_A.astype(theano.config.floatX))

                x = l.get_databatch()
                dA = l.model.gradient(x)['dA']
                dA *= l.eta

                param_max = np.max(np.abs(l.model.A.get_value()),axis=0)
                update_max = np.max(np.abs(dA),axis=0)
                update_max = np.max(update_max/param_max)

                l._adapt_eta(update_max)

                # no subset selection:
                #sender.send(dA,copy=False)

                # subset selection:
                inds = np.argwhere(dA.sum(0) != 0.).ravel()
                subset_dA = dA[:,inds]
                sender.send_pyobj(dict(inds=inds,subset_dA=subset_dA))

            receiver_stream.on_recv(_worker,copy=False)
            iolooper = ioloop.IOLoop.instance()
            iolooper.start()

            return

        self.amr = {}
        for engine_id in range(self.num_engines):
            if verbose: print("Start worker on engine", engine_id)
            self.amr[engine_id] = self.ipython_rc[engine_id].apply_async(worker,
                self.zmq_vent_address,self.zmq_sink_address,self._msg_sz,self._msg_dtype)

        if verbose: print('Waiting for engines to get ready...')
        time.sleep(5.)

        new_A = self.model.A.get_value()
        for engine_id in range(self.num_engines):
            print("Send msg to engine", engine_id)
            self.sender.send(new_A)

        # DEBUG:
        #for engine_id in range(self.num_engines):
        #    print("Check worker on engine", engine_id)
        #    print(self.amr[engine_id].get()) # this will block, but check for errors

    def initialization_sequence(self):
        self.init_parallel()
        self.init_remote_models()
        self.init_remote_zmq_workers()
        self.parallel_initialized = True

    def learn(self,iterations=1000):

        if not self.parallel_initialized:
            self.initialization_sequence()

        normalize_every = 100
        i = 0
        updates = 0
        not_done = True
        update_counter = defaultdict(int)

        LOCAL_A = self.model.A.get_value()

        def local_update(dA,local_A,normalize=False):
            local_A -= dA
            if normalize:
                Anorm = np.sqrt((local_A**2).sum(axis=0)).reshape(1,dA.shape[1])
                local_A /= Anorm
            return local_A

        def local_update_inds(dA,local_A,inds,normalize=False):
            local_A[:,inds] -= dA
            if normalize:
                Anorm = np.sqrt((local_A**2).sum(axis=0)).reshape(1,local_A.shape[1])
                local_A /= Anorm
            return local_A

        time_last_update_stamp = time.time()

        import cPickle as pickle
        print 'start listen loop...'
        while not_done:

            time_waiting_stamp = time.time()

            #new_message = self.receiver.recv(copy=False)
            new_message = self.receiver.recv()
            time_recv = time.time() - time_waiting_stamp
            time_pickle_stamp = time.time()
            new_message = pickle.loads(new_message)
            time_pickle = time.time() - time_pickle_stamp

            tic = time.time()
            time_last_update = tic - time_last_update_stamp
            time_waiting = tic - time_waiting_stamp
            time_last_update_stamp = tic

            tic = time.time()

            # pickle and active selection:
            LOCAL_A = local_update_inds(new_message['subset_dA'],
                                        LOCAL_A,
                                        new_message['inds'],
                                        not updates%normalize_every)
            # no pickle / active selection:
            #dA = np.frombuffer(buffer(new_message),dtype=self._msg_dtype).reshape(self._msg_sz)
            #LOCAL_A = local_update(dA,LOCAL_A,not updates%normalize_every)

            time_update_model = time.time() - tic

            tic = time.time()
            self.sender.send(LOCAL_A,copy=False)
            time_apply_async = time.time() - tic

            if verbose:
                print("|last update", '%2.2e'%time_last_update,
                      "|waiting", '%2.2e'%time_waiting,
                      "|recv", '%2.2e'%time_recv,
                      "|pickle", '%2.2e'%time_pickle,
                      "|len(inds)",'%05d'%len(new_message['inds']),
                      "|update_model",'%2.2e'%time_update_model,
                      "|apply_async", '%2.2e'%time_apply_async)

            updates += 1
            self.iter += 1
            update_counter[i] += 1

            self.save_state(updates,iterations,LOCAL_A)

            if updates == iterations:
                print("Done!")
                for engine in range(self.num_engines):
                    print("Engine %d Updates: %d"%(engine,update_counter[engine]))
                not_done = False

            i = (i + 1)%self.num_engines

        LOCAL_A = self.model.normalize_A(LOCAL_A)
        self.model.A.set_value(LOCAL_A)


    def change_target(self,fraction):

        if not self.parallel_initialized:
            self.initialization_sequence()

        def change_model_target(change_factor):
            global l
            l.change_target(change_factor)
            return l.eta

        self.eta_target_maxupdate *= fraction
        r = self.ipython_dv.apply(change_model_target,fraction)
        worker_etas = r.get()
        if verbose: print 'Latest etas', worker_etas
        self.eta = np.mean(worker_etas)

    def save_state(self,it,iterations,local_A):
        if not self.iter%self.save_every or self.iter == 1 or not self.iter%self.display_every:
            self.model.A.set_value(local_A)

        if not self.iter%self.save_every or self.iter == 1:
            print('Saving model at iteration %d'%self.iter)
            if self.iter == 1:
                self.model.save(save_txt=True)
            else:
                self.model.save()
            self.save()

        if not self.iter%self.display_every or self.iter == 1:
            print('Displaying model at iteration %d'%self.iter)
            self.model.display(save_string='learning_update=%07d'%self.iter)

        if not it%100: print('Update %d of %d, total updates %d'%(it, iterations, self.iter))


class SGD_layer(SGD):

    def __init__(self,**kargs):

        super(SGD_layer,self).__init__(**kargs)

        self.model_sequence = kargs['model_sequence']
        self.layer_params = kargs['layer_params']
        self.first_layer_learner = kargs['first_layer_learner']

    def get_databatch(self,batchsize=None,testing=False):
        batch = self.first_layer_learner.get_databatch(batchsize=batchsize,testing=testing)

        for mind, m in enumerate(self.model_sequence):
            if 'output_function' in self.layer_params[mind]:
                batch = m.output(m.preprocess(batch),self.layer_params[mind]['output_function'])
            else:
                batch = m.output(m.preprocess(batch))

        return batch

    def init_remote_models(self):

        # Import the modules we need:
        with self.ipython_dv.sync_imports():
            import numpy as np
            import time
            import theano

        self._msg_sz = self.model.A.get_value().shape
        self._msg_dtype = str(self.model.A.get_value().dtype)

        #self.ipython_dv.execute("from hdl.models import SparseSlowModel")
        #self.ipython_dv.execute("from hdl.learners import SGD")
        #r = self.ipython_dv.execute("l = SGD(model=SparseSlowModel())")
        #self.ipython_dv.wait()
        #print r.get()

        def reset_model_sequence():
            global global_model_sequence
            global_model_sequence = []
            return True

        def setup_model_seq(patch_sz,M,N,NN,D,T,
                        sparse_cost,slow_cost,
                        lam_sparse,lam_slow,lam_l2,
                        inputmean,whitenmatrix,dewhitenmatrix,zerophasewhitenmatrix,A
                        ):

            import numpy as np
            import theano
            from hdl.models import SparseSlowModel

            global global_model_sequence

            model=SparseSlowModel()

            model.patch_sz = patch_sz
            model.M = M
            model.D = D
            model.N = N
            model.NN = NN
            model.T = T
            model.sparse_cost = sparse_cost
            model.slow_cost = slow_cost
            model.inputmean = inputmean
            model.whitenmatrix = whitenmatrix
            model.dewhitenmatrix = dewhitenmatrix
            model.zerophasewhitenmatrix = zerophasewhitenmatrix

            old_type = type(A)
            model.A = theano.shared(A.astype(theano.config.floatX))
            model.lam_sparse = theano.shared(getattr(np,theano.config.floatX)(lam_sparse))
            model.lam_slow = theano.shared(getattr(np,theano.config.floatX)(lam_slow))
            model.lam_l2 = theano.shared(getattr(np,theano.config.floatX)(lam_l2))
            #model._reset_on_load()
            new_type = type(model.A)
            model.setup(init=False)

            global_model_sequence.append(model)

            return old_type, new_type, type(model.lam_sparse), model.lam_sparse.get_value()

        def setup_multilayer_model(layer_params,datasource,batchsize):

            global global_model_sequence, l
            from hdl.learners import SGD_layer, SGD

            first_layer_learner = SGD(model=global_model_sequence[0],datasource=datasource,batchsize=batchsize)
            m = global_model_sequence[-1]
            model_sequence = global_model_sequence[:-1]
            l = SGD_layer(first_layer_learner=first_layer_learner,model=m,
                datasource=datasource,
                batchsize=batchsize,
                model_sequence=model_sequence,
                layer_params=layer_params)
            return len(model_sequence)

        def check_multilayer_model():
            if hasattr(l,'model_sequence'):
                return type(l), len(l.model_sequence), l.model.A.get_value().shape, l.model.M, l.model.D, l.model.NN, l.model.N
            return type(l)

        if verbose: print("setup_model(...)")
        r = self.ipython_dv.apply(reset_model_sequence)
        r.wait()
        print(r.get())

        for model in self.model_sequence + [self.model,]:
            print("setup layer model")
            r = self.ipython_dv.apply(setup_model_seq,
                model.patch_sz,model.M,model.N,model.NN,model.D,model.T,
                model.sparse_cost,model.slow_cost,
                model.lam_sparse.get_value(),model.lam_slow.get_value(),model.lam_l2.get_value(),
                model.inputmean,model.whitenmatrix,model.dewhitenmatrix,model.zerophasewhitenmatrix,model.A.get_value())
            r.wait()
            print(r.get())

        r = self.ipython_dv.apply(setup_multilayer_model,self.layer_params,self.datasource,self.batchsize)
        r.wait()
        print(r.get())
        if verbose: print("check_model")
        r = self.ipython_dv.apply(check_multilayer_model)
        r.wait()
        print(r.get())

        if verbose:
            print 'expected array msg_sz:', self._msg_sz, 'and dtype msg_dtype:', self._msg_dtype

    def init_remote_zmq_workers(self):

        def worker(vent_address,sink_address,sz,dtype):

            import zmq
            import theano
            from zmq.eventloop import ioloop
            ioloop.install()
            from zmq.eventloop.zmqstream import ZMQStream

            # Context
            context = zmq.Context()

            # Socket to receive messages on
            receiver = context.socket(zmq.PULL)
            receiver.connect(vent_address)
            receiver_stream = ZMQStream(receiver)

            # Socket to send messages to
            sender = context.socket(zmq.PUSH)
            sender.connect(sink_address)

            def _worker(msg_list,sz=sz,dtype=dtype,sender=sender):
                import theano
                import numpy as np

                msg = msg_list[0]

                if True:
                    new_A = np.frombuffer(buffer(msg), dtype=dtype).reshape(sz)#.copy()
                    new_A = l.model.normalize_A(new_A)

                    l.model.A.set_value(new_A.astype(theano.config.floatX))

                    x = l.get_databatch()
                    dA = l.model.gradient(x)['dA']
                    dA *= l.eta

                    param_max = np.max(np.abs(l.model.A.get_value()),axis=0)
                    update_max = np.max(np.abs(dA),axis=0)
                    update_max = np.max(update_max/param_max)

                    l._adapt_eta(update_max)

                    # no subset selection:
                    #sender.send(dA,copy=False)

                    # subset selection:
                    inds = np.argwhere(dA.sum(0) != 0.).ravel()
                    subset_dA = dA[:,inds]
                    sender.send_pyobj(dict(inds=inds,subset_dA=subset_dA))

                else:
                    new_A = np.frombuffer(buffer(msg), dtype=dtype).reshape(sz)#.copy()
                    new_A = l.model.normalize_A(new_A)

                    inds = np.arange(10)
                    subset_dA = np.zeros((sz[0],len(inds)),dtype=dtype)
                    sender.send_pyobj(dict(inds=inds,subset_dA=subset_dA))

            receiver_stream.on_recv(_worker,copy=False)
            iolooper = ioloop.IOLoop.instance()
            iolooper.start()

            return

        self.amr = {}
        for engine_id in range(self.num_engines):
            if verbose: print("Start worker on engine", engine_id)
            self.amr[engine_id] = self.ipython_rc[engine_id].apply_async(worker,
                self.zmq_vent_address,self.zmq_sink_address,self._msg_sz,self._msg_dtype)

        if verbose: print('Waiting for engines to get ready...')
        time.sleep(5.)


        new_A = self.model.A.get_value()
        print 'Sending new_A with shape:', new_A.shape, 'and dtype:', new_A.dtype
        for engine_id in range(self.num_engines):
            print("Send msg to engine", engine_id)
            self.sender.send(new_A)

        #time.sleep(10.)
        ## DEBUG:
        #for engine_id in range(self.num_engines):
        #    print("Check worker on engine", engine_id)
        #    print(self.amr[engine_id].get()) # this will block, but check for errors

    def dummy_learn(self,iterations=1000):

        if not self.parallel_initialized:
            self.initialization_sequence()

        normalize_every = 100
        i = 0
        updates = 0
        not_done = True
        update_counter = defaultdict(int)

        LOCAL_A = self.model.A.get_value()

        def local_update(dA,local_A,normalize=False):
            local_A -= dA
            if normalize:
                Anorm = np.sqrt((local_A**2).sum(axis=0)).reshape(1,dA.shape[1])
                local_A /= Anorm
            return local_A

        def local_update_inds(dA,local_A,inds,normalize=False):
            local_A[:,inds] -= dA
            if normalize:
                Anorm = np.sqrt((local_A**2).sum(axis=0)).reshape(1,local_A.shape[1])
                local_A /= Anorm
            return local_A

        time_last_update_stamp = time.time()

        import cPickle as pickle
        print 'start listen loop...'
        while not_done:

            time_waiting_stamp = time.time()

            #new_message = self.receiver.recv(copy=False)
            #print self.receiver.poll(1000)


            #time.sleep(10.)
            ## DEBUG:
            #for engine_id in range(self.num_engines):
            #    print("Check worker on engine", engine_id)
            #    print(self.amr[engine_id].get()) # this will block, but check for errors

            new_message = self.receiver.recv()

            time_recv = time.time() - time_waiting_stamp
            time_pickle_stamp = time.time()
            new_message = pickle.loads(new_message)
            time_pickle = time.time() - time_pickle_stamp

            tic = time.time()
            time_last_update = tic - time_last_update_stamp
            time_waiting = tic - time_waiting_stamp
            time_last_update_stamp = tic

            tic = time.time()

            # pickle and active selection:
            LOCAL_A = local_update_inds(new_message['subset_dA'],
                                        LOCAL_A,
                                        new_message['inds'],
                                        not updates%normalize_every)
            # no pickle / active selection:
            #dA = np.frombuffer(buffer(new_message),dtype=self._msg_dtype).reshape(self._msg_sz)
            #LOCAL_A = local_update(dA,LOCAL_A,not updates%normalize_every)

            time_update_model = time.time() - tic

            tic = time.time()
            self.sender.send(LOCAL_A,copy=False)
            time_apply_async = time.time() - tic

            if verbose:
                print("|last update", '%2.2e'%time_last_update,
                      "|waiting", '%2.2e'%time_waiting,
                      "|recv", '%2.2e'%time_recv,
                      "|pickle", '%2.2e'%time_pickle,
                      "|len(inds)",'%05d'%len(new_message['inds']),
                      "|update_model",'%2.2e'%time_update_model,
                      "|apply_async", '%2.2e'%time_apply_async)

            updates += 1
            self.iter += 1
            update_counter[i] += 1

            self.save_state(updates,iterations,LOCAL_A)

            if updates == iterations:
                print("Done!")
                for engine in range(self.num_engines):
                    print("Engine %d Updates: %d"%(engine,update_counter[engine]))
                not_done = False

            i = (i + 1)%self.num_engines

        LOCAL_A = self.model.normalize_A(LOCAL_A)
        self.model.A.set_value(LOCAL_A)
