import kaldi_io
#import pytorch
import sys

import test_print

exit(0)
file = "test_data/feats.scp"
for key,mat in kaldi_io.read_mat_scp(file):
	print("key:",key)
	print("mat:",mat[0])
	break;
