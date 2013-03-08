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
        self._kargs = kargs

        self.parallel_initialized = False

    def init_parallel(self):

        # create client & view
        print "Initializing IPython Cluster with profile =", self.ipython_profile
        rc = parallel.Client(profile=self.ipython_profile)
        dv = rc[:]
        ids = rc.ids
        # scatter 'id', so id=0,1,2 on engines 0,1,2
        dv.scatter('engine_id', rc.ids, flatten=True)
        print("Engine IDs: ", dv['engine_id'])

        with dv.sync_imports():
            import os

        def check_theano_environ():
            global os
            return os.environ['THEANO_FLAGS']

        os.environ['THEANO_FLAGS']='mode=FAST_RUN,floatX=float32'
        print 'local env:'
        print check_theano_environ()

        import theano
        print theano.config.mode, theano.config.device, theano.config.floatX, theano.config.base_compiledir

        def check_pbs_environ():
            import os
            pbs_keys = ['PBS_O_HOST', 'PBS_TASKNUM', 'PBS_NODENUM', 'PBS_VNODENUM']
            pbs_info = []
            for pbs_key in pbs_keys:
                pbs_info.append(os.environ.get(pbs_key,None))
            return pbs_info

        pbs = dv.apply(check_pbs_environ)
        pbs_info = pbs.get()

        if verbose: print(pbs_info)

        self.num_engines = len(ids)
        if verbose: print 'num_engines', self.num_engines

        with dv.sync_imports():
            import theano

        def check_gpu():
            import theano
            return theano.config.mode, theano.config.device, theano.config.floatX, theano.config.base_compiledir

        rs = dv.apply(check_gpu)
        theano_gpu_info = rs.get()
        if verbose: print(theano_gpu_info)

        node_info = defaultdict(list)
        worker_info = {}
        for pbs_ind, pbs_item in enumerate(pbs_info):
            node_info[pbs_item[2]].append(theano_gpu_info[pbs_ind][1])
            if pbs_item[2] is None:
                worker_info[pbs_ind] = (0,0)
            else:
                worker_info[pbs_ind] = (int(pbs_item[2]),int(pbs_item[3]))
        self._worker_info = worker_info

        if verbose:
            for node_ind in sorted(node_info.keys()):
                print 'Node:', node_ind
                print 'Devices:', sorted(node_info[node_ind])


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

        def setup_model(model_class_name,
                        load_model_fname,
                        learner_class_name,
                        learner_kargs,
                        layer_params):

            global l, model

            from time import time as now

            import hdl.models
            from hdl.models import HierarchicalModel
            import hdl.learners

            Model = getattr(hdl.models, model_class_name)
            Learner = getattr(hdl.learners, learner_class_name)

            layer_model = Model()
            t0 = now()
            layer_model.load(load_model_fname,reset_theano=False)
            t_model_load = now() - t0
            t0 = now()
            layer_model._reset_on_load()
            t_model_reset = now() - t0
            l = Learner(model=layer_model, **learner_kargs)

            A = l.model.A.get_value()
            old_type = type(A)
            new_type = type(l.model.A)

            model = HierarchicalModel(model_sequence=[layer_model,],layer_params=layer_params)

            return old_type, new_type, type(l.model.lam_sparse), l.model.lam_sparse.get_value(), t_model_load, t_model_reset

        def check_model():
            return type(l.model.A)

        if verbose: print("setup_model(...)")
        load_model_fname = self.model.save()
        learner_kargs = {}
        for key in self._kargs.keys():
            if key == 'model': continue
            learner_kargs[key] = self._kargs[key]
        r = self.ipython_dv.apply(setup_model,
            self.model.__class__.__name__,load_model_fname, 'SGD', learner_kargs, self._kargs.get('layer_params',{}))
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

                # if normalize_A does any inplace operation, we need to .copy() here:
                new_A = np.frombuffer(buffer(msg), dtype=dtype).reshape(sz).copy()
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
                sender.send(dA,copy=False)

                # subset selection:
                #inds = np.argwhere(dA.sum(0) != 0.).ravel()
                #subset_dA = dA[:,inds]
                #sender.send_pyobj(dict(inds=inds,subset_dA=subset_dA))

            receiver_stream.on_recv(_worker,copy=False)
            iolooper = ioloop.IOLoop.instance()
            iolooper.start()

            return

        self.amr = {}
        if verbose: print "Starting workers"
        for engine_id in range(self.num_engines):
            self.amr[engine_id] = self.ipython_rc[engine_id].apply_async(worker,
                self.zmq_vent_address,self.zmq_sink_address,self._msg_sz,self._msg_dtype)

        if verbose: print('Waiting for engines to get ready...')
        time.sleep(5.)

        print 'Send msg to engines'
        new_A = self.model.A.get_value()
        for engine_id in range(self.num_engines):
            #print("Send msg to engine", engine_id)
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

        send_inds = False
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

            if len(inds) == local_A.shape[1]:
                local_A, update_max = self.model._update_model(local_A,dict(dA=dA))

            else:
                dA_fill = np.zeros_like(local_A)
                dA_fill[:,inds] = dA

                local_A, update_max = self.model._update_model(local_A,dict(dA=dA_fill))

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
            if send_inds:
                new_message = pickle.loads(new_message)
            time_pickle = time.time() - time_pickle_stamp

            tic = time.time()
            time_last_update = tic - time_last_update_stamp
            time_waiting = tic - time_waiting_stamp
            time_last_update_stamp = tic

            tic = time.time()

            if send_inds:
                # pickle and active selection:
                LOCAL_A = local_update_inds(new_message['subset_dA'],
                                           LOCAL_A,
                                           new_message['inds'],
                                           not updates%normalize_every)
            else:
                # no pickle / active selection:
                dA = np.frombuffer(buffer(new_message),dtype=self._msg_dtype).reshape(self._msg_sz)
                LOCAL_A = local_update(dA,LOCAL_A,not updates%normalize_every)

            time_update_model = time.time() - tic

            tic = time.time()
            self.sender.send(LOCAL_A,copy=False)
            time_apply_async = time.time() - tic

            if verbose:
                print "w: %03d, (%03d,%03d)"%(i,self._worker_info[i][0],self._worker_info[i][1]),\
                      "|last update", '%2.2e'%time_last_update,\
                      "|waiting", '%2.2e'%time_waiting,\
                      "|recv", '%2.2e'%time_recv,\
                      "|pickle", '%2.2e'%time_pickle,\
                      "|update_model",'%2.2e'%time_update_model,\
                      "|apply_async", '%2.2e'%time_apply_async

            updates += 1
            self.iter += 1
            update_counter[i] += 1

            self.save_state(updates,iterations,LOCAL_A)

            if updates == iterations:
                print("Done!")
                #for engine in range(self.num_engines):
                #    print("Engine %d Updates: %d"%(engine,update_counter[engine]))
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

    def parallel_model_call(self,batch,output_function='infer_abs',chunk_size=None):

        if not self.parallel_initialized:
            self.initialization_sequence()

        def perform_call(model_call_batch,output_function=output_function,chunk_size=chunk_size):

            global model

            input_batch = model_call_batch
            model_call_output = model(input_batch,output_function=output_function,chunk_size=chunk_size)

            return model_call_output

        if chunk_size is None:
            chunk_size = self.model.T
        not_done = True
        chunks = []
        szt = batch.shape[1]
        ind = 0
        while not_done:
            chunks.append(batch[:,ind:ind+chunk_size])
            ind += chunk_size
            if ind >= szt: not_done = False

        print 'Sending map command:'
        result = self.ipython_dv.map(perform_call,chunks,block=True)
        output = np.hstack(result)

        return output

class SGD_layer(SGD):

    def __init__(self,**kargs):

        super(SGD_layer,self).__init__(**kargs)

        self.model_sequence = kargs['model_sequence']
        self.layer_params = kargs.get('layer_params',{})
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

        def setup_model(model_class_names,
                        load_model_fnames,
                        first_layer_learner_class_name,
                        higher_layer_learner_class_name,
                        learner_kargs,
                        layer_params
                        ):

            global l, model

            from time import time as now

            import hdl.models
            from hdl.models import HierarchicalModel
            import hdl.learners

            all_models = []
            t_model_load = 0.
            t_model_reset = 0.
            for model_class_name, load_model_fname in zip(model_class_names,load_model_fnames):
                Model = getattr(hdl.models, model_class_name)
                layer_model = Model()
                t0 = now()
                layer_model.load(load_model_fname,reset_theano=False)
                t_model_load += now() - t0
                t0 = now()
                layer_model._reset_on_load()
                t_model_reset += now() - t0
                all_models.append(layer_model)

            first_layer_model = all_models[0]
            last_model = all_models[-1]
            model_sequence = all_models[:-1]
            Firstlayer_Learner = getattr(hdl.learners,first_layer_learner_class_name)
            l_firstlayer = Firstlayer_Learner(model=first_layer_model,**learner_kargs)

            Learner = getattr(hdl.learners, higher_layer_learner_class_name)
            l = Learner(first_layer_learner=l_firstlayer,
                        model=last_model,
                        model_sequence=model_sequence,
                        **learner_kargs)

            model = HierarchicalModel(model_sequence=all_models,layer_params=layer_params)

            return len(l.model_sequence), t_model_load, t_model_reset

        def check_multilayer_model():
            if hasattr(l, 'model_sequence'):
                return type(l), len(
                    l.model_sequence), l.model.A.get_value().shape, l.model.M, l.model.D, l.model.NN, l.model.N
            return type(l)

        if verbose: print("setup_model(...)")
        load_model_fnames = [model.save() for model in self.model_sequence] + [self.model.save(),]
        model_class_names = [model.__class__.__name__ for model in self.model_sequence] + [self.model.__class__.__name__,]
        first_layer_learner_class_name = self.first_layer_learner.__class__.__name__
        higher_layer_learner_class_name = 'SGD_layer'

        learner_kargs = {}
        for key in self._kargs.keys():
            if key == 'model': continue
            if key == 'model_sequence': continue
            if key == 'first_layer_learner': continue
            learner_kargs[key] = self._kargs[key]

        print 'model_class_names:', model_class_names
        print 'load_model_fnames:', load_model_fnames
        print 'first_layer_learnere_class_name:', first_layer_learner_class_name
        print 'higher_layer_learner_class_name', higher_layer_learner_class_name

        r = self.ipython_dv.apply(setup_model,
                                    model_class_names,
                                    load_model_fnames,
                                    first_layer_learner_class_name,
                                    higher_layer_learner_class_name,
                                    learner_kargs,
                                    self.layer_params)
        r.wait()
        print(r.get())
        if verbose: print("check_model")
        r = self.ipython_dv.apply(check_multilayer_model)
        r.wait()
        print(r.get())

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

            def _worker(msg_list, sz=sz, dtype=dtype, sender=sender):
                import theano
                import numpy as np

                msg = msg_list[0]

                # if normalize_A does any inplace operation, we need to .copy() here:
                new_A = np.frombuffer(buffer(msg), dtype=dtype).reshape(sz).copy()
                new_A = l.model.normalize_A(new_A)

                l.model.A.set_value(new_A.astype(theano.config.floatX))

                x = l.get_databatch()
                dA = l.model.gradient(x)['dA']
                dA *= l.eta

                param_max = np.max(np.abs(l.model.A.get_value()), axis=0)
                update_max = np.max(np.abs(dA), axis=0)
                update_max = np.max(update_max / param_max)

                l._adapt_eta(update_max)

                # no subset selection:
                sender.send(dA,copy=False)

                # subset selection:
                #inds = np.argwhere(dA.sum(0) != 0.).ravel()
                #subset_dA = dA[:, inds]
                #sender.send_pyobj(dict(inds=inds, subset_dA=subset_dA))

            receiver_stream.on_recv(_worker,copy=False)
            iolooper = ioloop.IOLoop.instance()
            iolooper.start()

            return

        self.amr = {}
        if verbose: print "Starting workers"
        for engine_id in range(self.num_engines):
            self.amr[engine_id] = self.ipython_rc[engine_id].apply_async(worker,
                self.zmq_vent_address,self.zmq_sink_address,self._msg_sz,self._msg_dtype)

        if verbose: print('Waiting for engines to get ready...')
        time.sleep(5.)


        new_A = self.model.A.get_value()
        print 'Sending new_A with shape:', new_A.shape, 'and dtype:', new_A.dtype
        for engine_id in range(self.num_engines):
            #print("Send msg to engine", engine_id)
            self.sender.send(new_A)

        #time.sleep(10.)
        ## DEBUG:
        #for engine_id in range(self.num_engines):
        #    print("Check worker on engine", engine_id)
        #    print(self.amr[engine_id].get()) # this will block, but check for errors
