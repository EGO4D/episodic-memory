#!/bin/sh

( cd DPT && python run_monodepth.py -t dpt_hybrid_nyu -i $1 -o $2 )
