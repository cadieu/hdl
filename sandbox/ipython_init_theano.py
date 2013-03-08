from collections import defaultdict
from IPython import parallel
import argparse

parser = argparse.ArgumentParser(description='Initialize Theano on IPython Cluster')
parser.add_argument('--profile',type=str,default='nodb',
                    help='profile name of IPython Cluster')
parser.add_argument('--usevnodenum',action='store_true',
                    help='flag for using VNODE_NUM to set GPU number, or local compile dir)')
parser.add_argument('--gpus_per_node',type=int,default=-1,
                    help='number of gpus available on each node (default = -1 = no gpus)')

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
def check_theano():
    return theano.config.mode, theano.config.device, theano.config.floatX, theano.config.base_compiledir

if not usevnodenum:

    print 'set on engines:'
    for id in rc.ids:
        rc[id].execute("os.environ['PATH'] = os.environ['PBS_O_PATH']")
        if gpus_per_node > 0:
            rc[id].execute("os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu%d,floatX=float32'%id")
        else:
            rc[id].execute("os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=cpu,floatX=float32'")
        #rc[id].execute("os.environ['THEANO_FLAGS']='mode=FAST_RUN,floatX=float32'")
        rs = rc[id].apply(check_environ)
        rc[id].execute("import theano")
        rc[id].execute("reload(theano)")

    rs = dv.apply(check_theano)
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

    def set_theano(gpus_per_node=gpus_per_node):
        os.environ['PATH'] = os.environ['PBS_O_PATH']
        vnodenum = int(os.environ['PBS_VNODENUM'])
        if gpus_per_node > 0:
            gpu_ind = vnodenum%gpus_per_node
            os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu%d,floatX=float32,base_compiledir=/var/tmp/.theano_%s'%(gpu_ind,vnodenum)
        else:
            gpu_ind = None
            os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=cpu,floatX=float32,base_compiledir=/var/tmp/.theano_%s'%vnodenum
        return gpu_ind, os.environ['THEANO_FLAGS'], os.environ['PATH']

    for id in rc.ids:
        rs = rc[id].apply(set_theano)
        print rs.get()
        rc[id].execute("import theano",block=True)
        rc[id].execute("reload(theano)",block=True)

    rs = dv.apply(check_theano)
    theano_info = rs.get()

    node_info = defaultdict(list)
    for pbs_ind, pbs_item in enumerate(pbs_info):
        node_info[pbs_item[2]].append(theano_info[pbs_ind][1])

    for node_ind in sorted(node_info.keys()):
        print 'Node:', node_ind
        print 'Devices:', sorted(node_info[node_ind])
