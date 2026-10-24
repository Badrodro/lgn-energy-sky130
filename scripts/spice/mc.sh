#!/bin/bash
# usage: mc.sh <WIN> <LIN> <first_seed> <n>  -> prints offset_mV energy_fJ per seed
WIN=$1; LIN=$2; S0=$3; N=$4
for s in $(seq $S0 $((S0+N-1))); do
  sed -e "s/^.param WIN=2 LIN=0.5/.param WIN=$WIN LIN=$LIN/" -e "s/^.tran 20p 1280n/.option seed=$s\n.tran 20p 1280n/" sa.cir > sa_$s.cir
  timeout 300 ngspice -b sa_$s.cir 2>/dev/null | awk -v s=$s '/^DEC/ {d[$2]=$3} /^IAVG/ {i=$2} END {
     flip=-1; for (k=1;k<128;k++) if ((d[k-1]<0) != (d[k]<0)) {flip=k; break}
     if (flip<0) print s, "noflip", -i*1.8*10e-9*1e15; else {vd=-40+80*(5+10*flip)/1280; print s, vd, -i*1.8*10e-9*1e15} }'
  rm -f sa_$s.cir
done
