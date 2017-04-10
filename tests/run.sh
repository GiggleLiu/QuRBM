#!/bin/bash
source ~/.bashrc
#export LD_LIBRARY_PATH=/fs01/apps/intel/composer_xe_2013/mkl/lib/intel64:/fs01/apps/inte/lcomposer_xe_2013/lib/intel64:$LD_LIBRARY_PATH
#export LD_RUN_PATH=/fs01/apps/intel/composer_xe_2013/lib/intel64:/fs01/apps/intel/composer_xe_2013/mkl/lib/intel64
#source ~/sci/bin/activate
scienv
(python test_carleo.py) &>> log-`date +%Y%m%d%H%M`.log
