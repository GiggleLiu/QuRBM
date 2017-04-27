#!/bin/bash
source ~/.bashrc
#export LD_LIBRARY_PATH=/fs01/apps/intel/composer_xe_2013/mkl/lib/intel64:/fs01/apps/inte/lcomposer_xe_2013/lib/intel64:$LD_LIBRARY_PATH
#export LD_RUN_PATH=/fs01/apps/intel/composer_xe_2013/lib/intel64:/fs01/apps/intel/composer_xe_2013/mkl/lib/intel64
#scienv
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
(python test_sr.py) &>> log-`date +%Y%m%d%H%M`.log
