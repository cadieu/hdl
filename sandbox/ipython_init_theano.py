from collections import defaultdict
from IPython import parallel
import argparse

parser = argparse.ArgumentParser(description='Initialize Theano on IPython Cluster GPUs')
parser.add_argument('--profile',type=str,default='nodb',
                    help='profile name of IPython Cluster')
parser.add_argument('--usevnodenum',action='store_true',
                    help='flag for using VNODE_NUM to set GPU number)')
parser.add_argument('--gpus_per_node',type=int,default=2,
                    help='number of gpus available on each node (default = 2)')

args = parser.parse_args()
profile = args.profile
usevnodenum = args.usevnodenum
gpus_per_node = args.gpus_per_node

# create client & view
rc = parallel.Client(profile=profile)
dv = rc[:]

# scatter 'id', so id=0,1,2 on engines 0,1,2
dv.scatter('id', rc.ids, flatten=True)
print("Engine IDs: ", dv['id'])

with dv.sync_imports():
    import os

def check_environ():
    return os.environ['THEANO_FLAGS']

import theano
def check_gpu():
    return theano.config.mode, theano.config.device, theano.config.floatX

if not usevnodenum:

    print 'set on engines:'
    for id in rc.ids:
        rc[id].execute("os.environ['PATH'] = os.environ['PBS_O_PATH']")
        rc[id].execute("os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu%d,floatX=float32'%id")
        #rc[id].execute("os.environ['THEANO_FLAGS']='mode=FAST_RUN,floatX=float32'")
        rs = rc[id].apply(check_environ)
        rc[id].execute("import theano")
        rc[id].execute("reload(theano)")

    rs = dv.apply(check_gpu)
    print rs.get()

else:

    def check_pbs_environ():
        import socket
        host = socket.gethostbyname(socket.gethostname())
        return os.environ['PBS_O_HOST'], \
               os.environ['PBS_TASKNUM'], \
               os.environ['PBS_NODENUM'], \
               os.environ['PBS_VNODENUM'], \
               os.environ['PBS_JOBID'], \
               os.environ['PBS_JOBNAME'], \
               os.environ['PBS_QUEUE'], \
               host

    pbs = dv.apply(check_pbs_environ)
    pbs_info = pbs.get()
    for i in pbs_info:
        print i

    def set_theano_gpu(gpus_per_node=gpus_per_node):
        os.environ['PATH'] = os.environ['PBS_O_PATH']
        vnodenum = int(os.environ['PBS_VNODENUM'])
        gpu_ind = vnodenum%gpus_per_node
        os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu%d,floatX=float32'%gpu_ind
        return gpu_ind, os.environ['THEANO_FLAGS'], os.environ['PATH']

    for id in rc.ids:
        rs = rc[id].apply(set_theano_gpu)
        print rs.get()
        rc[id].execute("import theano",block=True)
        rc[id].execute("reload(theano)",block=True)

    rs = dv.apply(check_gpu)
    theano_gpu_info = rs.get()

    node_gpu = defaultdict(list)
    for pbs_ind, pbs_item in enumerate(pbs_info):
        node_gpu[pbs_item[2]].append(theano_gpu_info[pbs_ind][1])

    for node_ind in sorted(node_gpu.keys()):
        print 'Node:', node_ind
        print 'GPUS:', sorted(node_gpu[node_ind])



