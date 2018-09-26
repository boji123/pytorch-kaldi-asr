#!/bin/bash
#------------------------------------------------------------
#edited by baiji
#this file is included to queue.pl
#--------------------        path        --------------------
MY_PYTORCH_KALDI_DIR=$HOME/work/pytorch-kaldi-asr
export PATH=$MY_PYTORCH_KALDI_DIR:$MY_PYTORCH_KALDI_DIR/kaldi:$MY_PYTORCH_KALDI_DIR/pytorch:$PATH
export PYTHONPATH=$MY_PYTORCH_KALDI_DIR/kaldi:$MY_PYTORCH_KALDI_DIR/pytorch:$PYTHONPATH
#if a user want to change a file in the library, just copy it to the local directory.
#the program will search local first
USER_EDITED_LIBRARY=`pwd`/local
export PATH=$USER_EDITED_LIBRARY:$USER_EDITED_LIBRARY/kaldi:$USER_EDITED_LIBRARY/pytorch:$PATH
export PYTHONPATH=$USER_EDITED_LIBRARY/kaldi:$USER_EDITED_LIBRARY/pytorch:$PYTHONPATH
