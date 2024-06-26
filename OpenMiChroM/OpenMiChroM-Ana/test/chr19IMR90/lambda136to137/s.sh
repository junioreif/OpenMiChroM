#!/bin/bash
for i in {1..20} 
do
    pdb2ndb.py -f iteration_$i/IMR90OPT_0_block0.pdb -n iteration_$i/IMR90OPT
    ndb2cndb.py -f iteration_$i/IMR90OPT.ndb -n iteration_$i/traj_12878
    rm iteration_$i/IMR90OPT.ndb
done
