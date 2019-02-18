for e in 0.01 0.015 0.02 0.025 0.03 0.035 0.04 0.05 0.06 0.07 0.08 0.09; do 
  for s in 1; do # NOTE only one smoothing!!!
    # s_combined.sh $epsilon $dsmooth $csmooth
    sbatch runscripts/s_combined.sh $e $s $s; sleep 1; 
  done; 
done
