import kaldi_io

file = "test_data/feats.scp"
for key,mat in kaldi_io.read_mat_scp(file):
	print("key:",key)
	print("mat:",mat[0])
	break;
