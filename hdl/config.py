"""configuration settings
"""

import os
from datetime import datetime
from getpass import getuser

verbose = True
verbose_timing = False

def tstring():
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

username = getuser()
data_dir = '/share/users/%s/data'%username
if not os.path.isdir(data_dir):
    data_dir = '/data'
if os.environ.has_key('HOME'):
    output_dir = os.path.join(os.environ['HOME'],'output')
else:
    output_dir = '/home/%s/output'%username
public_dir = '/share/users/%s/public'%username
scratch_local_dir = '/scratch_local/%s'%username
scratch_dir = '/share/users/%s/scratch'%username

state_dir = os.path.join(output_dir,'hdl','state')
model_dir = os.path.join(output_dir,'hdl','model')
fig_dir = os.path.join(output_dir,'hdl','figures')
tests_dir = os.path.join(output_dir,'hdl','tests')
