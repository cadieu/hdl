import learners

class HDL(object):
    """
    Hierarchical Directed Learner
    """
    def __init__(self, model_sequence, datasource, **kargs):

        self.model_sequence = model_sequence

        self.datasource = datasource

        self.batchsize = kargs.get('batchsize',48)

        self.schedules = []
        self.layer_params = []
        for layer in range(len(self.model_sequence)):

            self.layer_params.append({'whitenpatches':160000,'output_function':kargs.get('output_function','proj_abs')})

            sched_list = [{'iterations':80000},
                          {'iterations':80000,'change_target':.5},
                          {'iterations':80000,'change_target':.5}]
            self.schedules.append(sched_list)

    def learn(self,layer_start=0):

        l_firstlayer = None

        # learn additional layers:
        for mind, m in enumerate(self.model_sequence):
            if not mind:
                l = learners.SGD(model=m,datasource=self.datasource,display_every=20000,batchsize=self.batchsize)
                l_firstlayer = l
            else:
                l = learners.SGD_layer(first_layer_learner=l_firstlayer,model=m,datasource=self.datasource,display_every=20000,batchsize=self.batchsize,model_sequence=self.model_sequence[:mind],layer_params=self.layer_params)

            if layer_start > mind:
                continue

            whitenpatches = self.layer_params[mind]['whitenpatches']
            databatch = l.get_databatch(whitenpatches)

            l.model.learn_whitening(databatch)
            l.model.setup()

            sched_list = self.schedules[mind]

            for sdict in sched_list:
                if sdict.has_key('change_target'):
                    l.change_target(sdict['change_target'])
                if sdict.has_key('batchsize'):
                    l.batchsize *= sdict['batchsize']
                if sdict.has_key('iterations'):
                    l.learn(iterations=sdict['iterations'])
                else:
                    l.learn()

            from display import display_final
            display_final(self.model_sequence[mind])
