from gamma_config import gamma_config
from hardware_config import hardware_config_2p5d
import os, sys
from model_param import model_param
import astropy.units as u


current_path = os.getcwd() + "/"
gamma_path = current_path + "../gamma/src/GAMMA/"
hardware_path = current_path + "../hardware_performance/"

sys.path.insert(2, gamma_path)
from main import gamma

def run_gamma(gamma_config_sample):

    os.chdir(gamma_path)
    gamma(gamma_config_sample)
    os.chdir(hardware_path)

def cacheless_3d_decentralized_IO(mode = "decode", model_name = "nvidia_18p4b", folder = "/nvidia_18p4b/", batch_size = 8, seq_len = 4096):
    if mode == "training":
        num_cores = 16
        num_pe = 16384 //  4
        recompute = True
    elif mode == "decode" or mode == "prefill":
        num_cores = 64
        num_pe = 16384 // 2
        recompute = False
    for locality_factor in [1.0]:#[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        for dram_type in ["fine_grained_hbm"]:
            dram_bw = 100 * 10 ** 3
            dram_bw_per_core = int(dram_bw / num_cores)
            model_config_sample = model_param(batch_size = batch_size, dram_type = dram_type, phase = mode, num_accelerator = 1, recompute = recompute, 
                                              model_name=model_name,  seq_len = seq_len)
            
            hardware_config = hardware_config_2p5d(model_config_sample, dram_type = dram_type, num_cores = num_cores, num_pe = num_pe, 
                                                    dram_bw = dram_bw_per_core, l1_size = 1 * u.kB, l2_size = 10 * u.MB, no_L2 = True, 
                                                    disaggregated_io = True, locality_factor = locality_factor,
                                                    interconnect_assignment={"L1_DRAM": "TSV", "L1_L2": "BEOL", "L2_DRAM": "TSV"},
                                                    interconnect_dist={"L1_DRAM": 0 * u.mm, "L1_L2": 0.0 * u.mm, "L2_DRAM": 0.0 * u.mm})
            filename = mode + "_decentral_200_without_l2_1_" + str(locality_factor).replace(".", "p") + '_' + mode
            gamma_config_sample = gamma_config(constraints = {"power": 200}, hardware_config = hardware_config, model_config = model_config_sample, 
                                            file_name =  folder + filename, num_layer = 5)
            run_gamma(gamma_config_sample)

def cache_3d_decentralized_IO(mode = "decode", model_name = "nvidia_18p4b", folder = "/nvidia_18p4b/", batch_size = 8, seq_len = 4096):
    if mode == "training":
        num_cores = 16
        num_pe = 16384 //  4
        recompute = True
    elif mode == "decode" or mode == "prefill" or mode == "prefill":
        num_cores = 64
        num_pe = 16384 // 2
        recompute = False
    for locality_factor in [1.0]:#[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        for dram_type in ["fine_grained_hbm"]:
            dram_bw = 100 * 10 ** 3
            dram_bw_per_core = int(dram_bw / num_cores)
            model_config_sample = model_param(batch_size = batch_size, dram_type = dram_type, phase = mode, num_accelerator = 1, recompute = recompute,
                                              model_name=model_name, seq_len = seq_len)
            
            hardware_config = hardware_config_2p5d(model_config_sample, dram_type = dram_type, num_cores = num_cores, num_pe = num_pe, 
                                                    dram_bw = dram_bw_per_core, l1_size = 1 * u.kB, l2_size = 10 * u.MB, no_L2 = False, 
                                                    disaggregated_io = True, locality_factor = locality_factor,
                                                    interconnect_assignment={"L1_DRAM": "TSV", "L1_L2": "BEOL", "L2_DRAM": "TSV"},
                                                    interconnect_dist={"L1_DRAM": 0 * u.mm, "L1_L2": 0.0 * u.mm, "L2_DRAM": 0.0 * u.mm})
            filename = mode + "_decentral_200_with_l2_1_" + str(locality_factor).replace(".", "p") + '_' + mode
            gamma_config_sample = gamma_config(constraints = {"power": 200}, hardware_config = hardware_config, model_config = model_config_sample, 
                                            file_name =  folder + filename, num_layer = 5)
            run_gamma(gamma_config_sample) 

def l2cache_2p5d_centralized_IO(mode = "decode", model_name = "nvidia_18p4b", folder = "/nvidia_18p4b/", batch_size = 8, seq_len = 4096):
    if mode == "training":
        num_cores = 16
        num_pe = 16384 //  4
        recompute = True
    elif mode == "decode" or mode == "prefill":
        num_cores = 64
        num_pe = 16384 // 2
        recompute = False
    for dram_type in ["fine_grained_hbm1"]:
        dram_bw = 100 * 10 ** 3
        dram_bw_per_core = int(dram_bw / num_cores)
        model_config_sample = model_param(batch_size = batch_size, dram_type = dram_type, phase = mode, num_accelerator = 1, recompute = recompute,
                                          model_name=model_name, seq_len = seq_len)
        
        hardware_config = hardware_config_2p5d(model_config_sample, dram_type = dram_type, num_cores = num_cores, num_pe = num_pe, 
                                                dram_bw = dram_bw, l1_size = 2 * u.kB, l2_size = 10 * u.MB, no_L2 = False, disaggregated_io = False,
                                                interconnect_assignment={"L1_DRAM": "TSV", "L1_L2": "HB", "L2_DRAM": "2.5D_Si"},
                                                interconnect_dist={"L1_DRAM": 0 * u.mm, "L1_L2": 0.0 * u.mm, "L2_DRAM": 5.5 * u.mm})
        
        gamma_config_sample = gamma_config(constraints = {"power": 200}, hardware_config = hardware_config, model_config = model_config_sample, 
                                           file_name =  folder + mode + "_central_new_l2_2p5d", num_layer = 5)
        run_gamma(gamma_config_sample) 

def l2_cache_3d_dram_2p5d(mode = "decode", model_name = "nvidia_18p4b", folder = "/nvidia_18p4b/", batch_size = 8, seq_len = 4096):
    if mode == "training":
        num_cores = 16
        num_pe = 16384 //  4
        recompute = True
    elif mode == "decode" or mode == "prefill":
        num_cores = 64
        num_pe = 16384 // 2
        recompute = False
    for dram_type in ["hbm3"]:
        dram_bw = 2 * 10 ** 3
        dram_bw_per_core = int(dram_bw / num_cores)
        model_config_sample = model_param(batch_size = batch_size, dram_type = dram_type, phase = mode, num_accelerator = 1, recompute = recompute,
                                          model_name=model_name, seq_len = seq_len)

        hardware_config = hardware_config_2p5d(model_config_sample, dram_type = dram_type, num_cores = num_cores, num_pe = num_pe,
                                                dram_bw = dram_bw, l1_size = 2 * u.kB, l2_size = 10 * u.MB, no_L2 = False, disaggregated_io = False,
                                                interconnect_assignment={"L1_DRAM": "TSV", "L1_L2": "HB", "L2_DRAM": "2.5D_Si"},
                                                interconnect_dist={"L1_DRAM": 0 * u.mm, "L1_L2": 0.0 * u.mm, "L2_DRAM": 5.5 * u.mm})
        
        gamma_config_sample = gamma_config(constraints = {"power": 200}, hardware_config = hardware_config, model_config = model_config_sample,
                                             file_name =  folder + mode + "_l2cache_3d", num_layer = 5)
        run_gamma(gamma_config_sample)

def baseline(mode = "decode", model_name = "nvidia_18p4b", folder = "/nvidia_18p4b/", batch_size = 8, seq_len = 4096):
    if mode == "training":
        num_cores = 16
        num_pe = 16384 //  4
        recompute = True
    elif mode == "decode" or mode == "prefill":
        num_cores = 64
        num_pe = 16384 // 2
        recompute = False
    for dram_type in ["hbm3"]:
        dram_bw = 2 * 10 ** 3
        dram_bw_per_core = int(dram_bw / num_cores)
        model_config_sample = model_param(batch_size = batch_size, dram_type = dram_type, phase = mode, num_accelerator = 1, recompute = recompute,
                                            model_name=model_name, seq_len = seq_len)

        hardware_config = hardware_config_2p5d(model_config_sample, dram_type = dram_type, num_cores = num_cores, num_pe = num_pe,
                                                dram_bw = dram_bw, l1_size = 2 * u.kB, l2_size = 10 * u.MB, no_L2 = False, disaggregated_io = False,
                                                interconnect_assignment={"L1_DRAM": "TSV", "L1_L2": "BEOL", "L2_DRAM": "2.5D_Si"},
                                                interconnect_dist={"L1_DRAM": 0 * u.mm, "L1_L2": 0.0 * u.mm, "L2_DRAM": 5.5 * u.mm})
        
        gamma_config_sample = gamma_config(constraints = {"power": 200}, hardware_config = hardware_config, model_config = model_config_sample,
                                             file_name =  folder + mode + "_baseline", num_layer = 5)
        run_gamma(gamma_config_sample)


def slt(mode = "decode", model_name = "nvidia_18p4b", folder = "/nvidia_18p4b/", batch_size = 8, seq_len = 4096):
    if mode == "training":
        num_cores = 16
        num_pe = 16384 //  4
        recompute = True
    elif mode == "decode" or mode == "prefill":
        num_cores = 64
        num_pe = 16384 // 2
        recompute = False
    for dram_type in ["hbm3"]:
        dram_bw = 100 * 10 ** 3
        dram_bw_per_core = int(dram_bw / num_cores)
        model_config_sample = model_param(batch_size = batch_size, dram_type = dram_type, phase = mode, num_accelerator = 1, recompute = recompute,
                                            model_name=model_name, seq_len = seq_len)

        hardware_config = hardware_config_2p5d(model_config_sample, dram_type = dram_type, num_cores = num_cores, num_pe = num_pe,
                                                dram_bw = dram_bw, l1_size = 2 * u.kB, l2_size = 10 * u.MB, no_L2 = False, disaggregated_io = False,
                                                interconnect_assignment={"L1_DRAM": "TSV", "L1_L2": "BEOL", "L2_DRAM": "2.5D_Si"},
                                                interconnect_dist={"L1_DRAM": 0 * u.mm, "L1_L2": 0.0 * u.mm, "L2_DRAM": 5.5 * u.mm})
        
        gamma_config_sample = gamma_config(constraints = {"power": 200}, hardware_config = hardware_config, model_config = model_config_sample,
                                             file_name =  folder + mode + "_slt", num_layer = 5)
        run_gamma(gamma_config_sample)

if __name__ == "__main__":

    model_name = "llama2_7b"
    mode = "prefill"
    folder = "/" + model_name + "_prefill/"
    if not os.path.exists(current_path + folder):
        os.makedirs(current_path + folder)
    batch_size = 32
    seq_len = 4096
    print("Running for model: ", model_name, " and mode: ", mode)
    print("Running for batch size: ", batch_size)
    cache_3d_decentralized_IO(mode = mode, model_name=model_name, folder=folder, batch_size=batch_size, seq_len = seq_len)
    cacheless_3d_decentralized_IO(mode=mode, model_name=model_name, folder=folder, batch_size=batch_size, seq_len = seq_len)
    l2cache_2p5d_centralized_IO(mode=mode, model_name=model_name, folder=folder, batch_size=batch_size, seq_len = seq_len)
    l2_cache_3d_dram_2p5d(mode=mode, model_name=model_name, folder=folder, batch_size=batch_size, seq_len = seq_len)
    baseline(mode=mode, model_name=model_name, folder=folder, batch_size=batch_size, seq_len = seq_len)
    slt(mode=mode, model_name=model_name, folder=folder, batch_size=batch_size, seq_len = seq_len)
            