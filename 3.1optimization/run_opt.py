import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

for i in range(39,40,1):
    #os.system("python extrapolation_IUV_mix_final.py --frame_index %d"%i)
    os.system("python IUV_optimization_all.py --frame_index %d --iter 16501 --factor 0 --IUV_initial_path /mnt/netdisk1/youxie/Fashion_extrapolation/A1wwPTTzVGS/extrapolation/"%i) #first run the optimization for smoothness
    os.system("python IUV_optimization_all.py --frame_index %d --iter 1001 --factor 1 --IUV_initial_path A1wwPTTzVGS_optimization/"%i) #then run the optimization with full losses
