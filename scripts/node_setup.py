from hdl.config import scratch_local_dir, public_dir
import os, shutil

if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option('-u', '--bringup', action="store_true", dest='bringup', default=False, help='setup this node from scratch')
    parser.add_option('-d', '--teardown', action="store_true", dest='teardown', default=False, help='tear down this node')

    (options, args) = parser.parse_args()

    print 'node_setup.py call'

    if options.bringup:
        print 'node_setup.py --bringup'
        db_dir = os.path.join(scratch_local_dir,'YouTubeFaces','aligned_images_DB')
        if not os.path.exists(db_dir):
            if not os.path.exists(scratch_local_dir): os.makedirs(scratch_local_dir)

            aligned_db_sourcefile = os.path.join(public_dir,'YouTubeFaces','archive','aligned_images_DB.tar.gz')
            aligned_db_targetdir = os.path.join(scratch_local_dir,'YouTubeFaces')
            if not os.path.exists(aligned_db_targetdir): os.makedirs(aligned_db_targetdir)
            aligned_db_targetfile = os.path.join(aligned_db_targetdir,'aligned_images_DB.tar.gz')
            shutil.copy(src=aligned_db_sourcefile,dst=aligned_db_targetfile)
            untarcommand = 'tar -xzf %s -C %s/'%(aligned_db_targetfile, aligned_db_targetdir)
            print untarcommand
            os.system(untarcommand)

        else:
            print 'aligned_faces dir found', db_dir, 'Nothing to do'

    if options.teardown:
        print 'node_setup.py --teardown START'
        shutil.rmtree(scratch_local_dir)
        print 'node_setup.py --teardown DONE'

