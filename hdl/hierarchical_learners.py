import learners
import parallel_learners
import models

class HDL(object):
    """
    Hierarchical Directed Learner
    """
    def __init__(self, model_sequence, datasource, **kargs):

        self.model_sequence = model_sequence

        self.datasource = datasource

        self.display_every = kargs.get('display_every',20000)
        self.save_every = kargs.get('save_every',20000)
        self.batchsize = kargs.get('batchsize',48)
        self.default_whitenpatches = kargs.get('default_whitenpatches', 160000)
        self.start_eta_target_maxupdate = kargs.get('start_eta_target_maxupdate',.05)

        self.schedules = []
        self.layer_params = []
        self.extra_learner_kargs = kargs.get('learner_kargs',{})

        default_sched_list = [{'iterations':80000},
                              {'iterations':80000,'change_target':.5},
                              {'iterations':80000,'change_target':.5}]
        sched_list = kargs.get('sched_list',default_sched_list)

        if kargs.has_key('output_functions'):
            output_functions = kargs.get('output_functions')
            assert len(output_functions) == len(self.model_sequence)
        else:
            output_function = kargs.get('output_function','proj_abs')
            output_functions = [output_function,]*len(self.model_sequence)
        for layer in range(len(self.model_sequence)):

            self.layer_params.append({'whitenpatches':self.default_whitenpatches,'output_function':output_functions[layer]})

            self.schedules.append(sched_list)

        self.ipython_profile = kargs.get('ipython_profile',None)
        self.evaluation_object = kargs.get('evaluation_object',None)
        self.iter = 0

    def evaluate(self):
        if not self.evaluation_object is None:
            self.evaluation_object(self)

    def learn(self,layer_start=0):

        l_firstlayer = None

        # learn additional layers:
        for mind, m in enumerate(self.model_sequence):
            if not mind:
                if self.ipython_profile is None:
                    l = learners.SGD(model=m,datasource=self.datasource,display_every=self.display_every,save_every=self.save_every,batchsize=self.batchsize,eta_target_maxupdate=self.start_eta_target_maxupdate,**self.extra_learner_kargs)
                else:
                    l = parallel_learners.SGD(model=m,datasource=self.datasource,display_every=self.display_every,save_every=self.save_every,batchsize=self.batchsize,ipython_profile=self.ipython_profile,eta_target_maxupdate=self.start_eta_target_maxupdate,**self.extra_learner_kargs)
                l_firstlayer = l
            else:
                if self.ipython_profile is None:
                    l = learners.SGD_layer(first_layer_learner=l_firstlayer,model=m,datasource=self.datasource,display_every=self.display_every,save_every=self.save_every,batchsize=self.batchsize,model_sequence=self.model_sequence[:mind],layer_params=self.layer_params,eta_target_maxupdate=self.start_eta_target_maxupdate,**self.extra_learner_kargs)
                else:
                    l = parallel_learners.SGD_layer(first_layer_learner=l_firstlayer,model=m,datasource=self.datasource,display_every=self.display_every,save_every=self.save_every,batchsize=self.batchsize,model_sequence=self.model_sequence[:mind],layer_params=self.layer_params,ipython_profile=self.ipython_profile,eta_target_maxupdate=self.start_eta_target_maxupdate,**self.extra_learner_kargs)

            if layer_start > mind:
                self.model = models.HierarchicalModel(model_sequence=self.model_sequence[:mind+1],layer_params=self.layer_params[:mind+1])
                self.evaluate()
                continue

            print 'Begin learning layer', mind

            print 'Learn whitening...'
            whitenpatches = self.layer_params[mind]['whitenpatches']
            databatch = l.get_databatch(whitenpatches)

            l.model.learn_whitening(databatch)
            print 'Done.'

            print 'Setup model...'
            l.model.setup()
            print 'Done.'

            # update self.model to use the sequence up to this layer
            self.model = models.HierarchicalModel(model_sequence=self.model_sequence[:mind+1],layer_params=self.layer_params[:mind+1])

            sched_list = self.schedules[mind]

            iter0 = self.iter
            self.evaluate()
            for sdict in sched_list:
                if sdict.has_key('change_target'):
                    l.change_target(sdict['change_target'])
                if sdict.has_key('batchsize'):
                    l.batchsize = sdict['batchsize']
                if sdict.has_key('center_basis_functions') and hasattr(l.model,'center_basis_functions'):
                    l.model.center_basis_functions = sdict['center_basis_functions']
                if sdict.has_key('iterations'):
                    l.learn(iterations=sdict['iterations'])
                else:
                    l.learn()

                self.iter = iter0 + l.iter
                self.evaluate()

            from display import display_final
            display_final(self.model_sequence[mind])
