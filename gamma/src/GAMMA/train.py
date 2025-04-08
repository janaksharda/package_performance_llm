
import copy
import argparse
from datetime import datetime
import astropy.units as u


import glob
import os, sys

script_dir = os.path.dirname(__file__)
module_path = os.path.abspath(os.path.join(script_dir, '../'))
project_path = os.path.abspath(os.path.join(script_dir, '../../../hardware_performance/'))

if module_path not in sys.path:
    sys.path.insert(0,module_path)
if project_path not in sys.path:
    sys.path.insert(0,project_path)

from utils import *
import gamma as gamma
from math import ceil
import importlib
from shutil import copyfile

from hardware_config import hardware_config_2p5d
import pandas as pd
fitness_list = None
fitness = None
stage_idx = 0
prev_stage_value = []
tune_iter = 1
opt = None
MAC_AREA_MAESTRO=4470
MAC_AREA_INT8 = 282
BUF_AREA_perbit = 0.086
L2BUF_AREA_MAESTRO = 4161.536
L1BUF_AREA_MAESTRO = 4505.1889
L2BUF_UNIT = 32768
L1BUF_UNIT = 64



# bias = {"par": {1: "K", 2:"C"}, "order":{1:["K", "C"]}, "tiles": {1:{"K":0.1, "C":0.2}, 2:{"K":0.3}}}
bias = {"par": {1: "K", 2:"C"}, "order":{1:["K", "C","Y", "X"], 2:["K", "C","Y", "X"]}}
# bias = {"par": {1: "K", 2:"C"}}
# bias = {"par": {1: "Y"}}


def get_pe_usage(env, sol, num_pe ):
    util_num_pe = num_pe
    baseline = env.get_indiv_info( sol, num_pe=num_pe)
    best_runtime, best_energy, best_area = baseline
    baseline = np.array(baseline)[:-2]
    for i in range(num_pe-1):
        util_num_pe -= 1
        cur = env.get_indiv_info(sol, num_pe=util_num_pe)
        best_runtime, best_energy, best_area = cur
        cur = np.array(cur)[:-2]
        if sum(baseline!=cur)>1:
            util_num_pe += 1
            break
    return util_num_pe

def train_model(model_defs, input_arg, map_cstr=None, chkpt_file='./chkpt'):
    global opt
    opt = input_arg
    fitness = opt.fitness
    env_list = [None] * len(model_defs)
    constraints = opt.constraints#"area":opt.area_budget* 1e6}
    df_list = [None] * len(model_defs)
    layer_multiplier = opt.model_config.layer_multiplier
    store_output = opt.model_config.store_output
    store_input = opt.model_config.store_input
    store_weight = opt.model_config.store_weight
    store_output_l2 = opt.model_config.store_output_l2
    store_input_l2 = opt.model_config.store_input_l2
    store_weight_l2 = opt.model_config.store_weight_l2
    epochs = opt.epochs
    num_operations_list = []
    for i, dimension in enumerate(model_defs):
        num_operations_list.append(dimension[0]*dimension[1]*dimension[2]*layer_multiplier[i])
    
    ## allocate the number of PEs based on the number of operations
    norm_num_operations = np.array(num_operations_list) / sum(num_operations_list)
    num_pe = int(opt.num_pe)
    num_pe_list = [int(ceil(num_pe * norm_num_operations[i])) for i in range(len(model_defs))]
    pe_util_list = norm_num_operations
    # print(pe_util_list)
    # print(np.argsort(pe_util_list))

    for k in range(epochs):
        
        sorted_list_pe_util = np.argsort(pe_util_list) 
        num_gen = min(k+1, 5)   
        for j in range(len(model_defs)):
            i = sorted_list_pe_util[j]
            dimension = model_defs[i]
            if k > 1:
                env_list[i].NocBW = int(1.1 * env_list[i].NocBW)
                env_list[i].offchipBW = int(1.1 * env_list[i].offchipBW)
            hardware_config = opt.hardware_config
            if env_list[i] is None:
                env_list[i] = gamma.GAMMA(dimension=dimension, num_pe=num_pe_list[i], fitness=fitness, par_RS=opt.parRS, l1_size=opt.l1_size, 
                                package1 = hardware_config,l2_size=opt.l2_size, NocBW=int(opt.NocBW / 80.0), offchipBW=int(opt.offchipBW / 80.0), slevel_min=opt.slevel_min, 
                                slevel_max=opt.slevel_max, fixedCluster=opt.fixedCluster, log_level=opt.log_level, map_cstr=map_cstr, layer_no = i, 
                                layer_multiplier=layer_multiplier[i], vector_width=opt.vector_width, store_output=store_output[i], constraints=constraints,
                                store_input=store_input[i], store_weight=store_weight[i], store_output_l2=store_output_l2[i], store_input_l2=store_input_l2[i], store_weight_l2=store_weight_l2[i])
                # env_list[i].reset_dimension(fitness=fitness, constraints=constraints, dimension=dimension)
                # env_list[i].reset_hw_parm(num_pe=opt.num_pe, l1_size=opt.l1_size, l2_size=opt.l2_size, pe_limit=opt.pe_limit,area_pebuf_only=False, external_area_model=True)
<<<<<<< HEAD
            
            chkpt, pops = env_list[i].run(dimension, stage_idx=0, num_population=opt.num_pop, prev_stage_value=None, num_generations=num_gen,
                                    best_sol_1st=None, init_pop=None, bias=None, uni_base=True, use_factor=opt.use_factor, use_pleteau=False, df_list=df_list)
            df_list[i] = env_list[i].best_df
=======
            print("assigned bw: ", env_list[i].NocBW, env_list[i].offchipBW)
            chkpt, pops = env_list[i].run(dimension, stage_idx=0, num_population=opt.num_pop, prev_stage_value=None, num_generations=num_gen,
                                    best_sol_1st=None, init_pop=None, bias=None, uni_base=True, use_factor=opt.use_factor, use_pleteau=False, df_list=df_list)
            df_list[i] = env_list[i].best_df
            print("Actual BW: ", df_list[i][0][" Offchip BW Req (Elements/cycle)"], df_list[i][0][" NoC BW Req (Elements/cycle)"])
>>>>>>> 6507447 (Remove submodule link and add as regular files)
            
            best_sol = chkpt["best_sol"]
            if (k == epochs - 1) and (j == len(model_defs) - 1):
                print_info = True
            else:
                print_info = False
            observation, chiplet_dict, package_dict = env_list[i].get_indiv_info(best_sol, num_pe=None, print_info = print_info)
            layer_latency, best_runtime, best_energy, best_area, best_pe_util, noc_bw, dram_bw, l2_size = observation
            pe_util_list[i] = best_pe_util
            # print("Mapping:", chkpt["best_sol"])
<<<<<<< HEAD
            if k % 5 == 0 and j == 0:
                print("Assigned bw: ", env_list[i].NocBW, env_list[i].offchipBW)
                print("Actual BW: ", df_list[i][0][" Offchip BW Req (Elements/cycle)"], df_list[i][0][" NoC BW Req (Elements/cycle)"])
                print("Epoch:", k, "Layer:", i, "NoC BW:", noc_bw, " DRAM BW:", dram_bw)
                print("Epoch:", k, "Layer:", i, "NoC BW:", df_list[i][0][" NoC BW Req (Elements/cycle)"], " DRAM BW:", df_list[i][0][" Offchip BW Req (Elements/cycle)"])
                print(f"Reward: {chkpt['best_reward'][0]:.3e}, Runtime: {best_runtime:.0f}(cycles), Area: {best_area:.3f}, Energy: {best_energy.to(u.mJ):.3e}, PE Util: {best_pe_util:.3f}")
                print("_________________________________________________________________________________________")
=======
            print("Epoch:", k, "Layer:", i, "NoC BW:", noc_bw, " DRAM BW:", dram_bw)
            print("Epoch:", k, "Layer:", i, "NoC BW:", df_list[i][0][" NoC BW Req (Elements/cycle)"], " DRAM BW:", df_list[i][0][" Offchip BW Req (Elements/cycle)"])
            print(f"Reward: {chkpt['best_reward'][0]:.3e}, Runtime: {best_runtime:.0f}(cycles), Area: {best_area:.3f}, Energy: {best_energy.to(u.mJ):.3e}, PE Util: {best_pe_util:.3f}")
        
>>>>>>> 6507447 (Remove submodule link and add as regular files)
            if k > 0 and  j == 0 and k < epochs - 25:
                break
        ## save df_list in a single file, filepath = '../../../hardware_performance/final_soln.csv'
        # Combine all dataframes in df_list into a single dataframe after taking transpose for each element in df_list
        if k == epochs - 1:    
            new_df_list = []
            if package_dict != {}:
                ## save the chiplet_dict and package_dict in a file
                filepath_chiplet = os.path.abspath(os.path.join(script_dir, '../../../hardware_performance/'+opt.file_name+'_chiplet.pkl'))
                filepath_package = os.path.abspath(os.path.join(script_dir, '../../../hardware_performance/'+opt.file_name+'_package.pkl'))
                ### save the chiplet_dict and package_dict in a pickle file
                with open(filepath_chiplet, 'wb') as f:
                    pickle.dump(chiplet_dict, f)
                with open(filepath_package, 'wb') as f:
                    pickle.dump(package_dict, f)               

            for i in range(len(df_list)):
                new_df_list.append(df_list[i].transpose())

            final_df = pd.concat(new_df_list, ignore_index=True)
            
            # Define the filepath
            filepath = os.path.abspath(os.path.join(script_dir, '../../../hardware_performance/'+opt.file_name+'.csv'))

            # Save the dataframe to a CSV file
            final_df.to_csv(filepath, index=False)


<<<<<<< HEAD
        
=======
        print("_________________________________________________________________________________________")
>>>>>>> 6507447 (Remove submodule link and add as regular files)
        # chkpt = {
        #     "reward":chkpt['best_reward'][0],
        #     "best_sol":best_sol,
        #     "runtime":best_runtime,
        #     "area":best_area,
        #     "energy":best_energy
        # }

        ## save df_list in a single file, filepath = '../../../hardware_performance/final_soln.csv'
        # # Combine all dataframes in df_list into a single dataframe
        # final_df = pd.concat(df_list, ignore_index=True)

        # # Define the filepath
        # filepath = os.path.abspath(os.path.join(script_dir, '../../../hardware_performance/final_soln.csv'))

        # # Save the dataframe to a CSV file
        # final_df.to_csv(filepath, index=False)





def get_cstr_name(mapping_cstr):
    if mapping_cstr:
        cstr_name = mapping_cstr
    else:
        cstr_name = "free"
    return cstr_name


