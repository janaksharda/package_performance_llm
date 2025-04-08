import numpy as np
import pandas as pd
import astropy.units as u


class layer_performance:
    
    ## read pattern; input, weights are read from DRAM and written to l2 and then from l2 to l1
        ### output is read from l1 and written to l2 and then from l2 to DRAM

    def __init__(self, df, bit_param, tech_param, model_param, layer_no, bw, no_l2 = False):
        self.bit_param = bit_param
        self.tech_param = tech_param
        self.no_l2 = no_l2
        self.noc_bw = bw[0]
        self.dram_bw = bw[1]
        
        self.eff_layer_multiplier = model_param.layer_multiplier[layer_no] * model_param.num_layers

        self.if_read_data = False

        self.read_data(df)
        
    def read_data(self, df):
        self.num_mac_cycles = np.array(df[" Runtime (Cycles)"]) * self.eff_layer_multiplier
        self.num_pe = np.array(df[" NumPEs"])
        self.num_mac = np.array(df[" Num MACs"]) * self.eff_layer_multiplier
        self.vector_width = np.array(df[" Vector Width"])

        self.l1_size = np.array(df[" L1 SRAM Size Req (Bytes)"]) * u.byte
        self.l2_size = np.array(df["  L2 SRAM Size Req (Bytes)"]) * u.byte

        self.l1_input_read = np.array(df[" input l1 read"]) * self.eff_layer_multiplier
        self.l1_input_write = np.array(df[" input l1 write"]) * self.eff_layer_multiplier
        self.l1_weight_read = np.array(df["filter l1 read"]) * self.eff_layer_multiplier
        self.l1_weight_write = np.array(df[" filter l1 write"]) * self.eff_layer_multiplier
        self.l1_output_read = np.array(df["output l1 read"]) * self.eff_layer_multiplier
        self.l1_output_write = np.array(df[" output l1 write"]) * self.eff_layer_multiplier

        self.l2_input_read = np.array(df[" input l2 read"]) * self.eff_layer_multiplier
        self.l2_input_write = 0 # np.array(df[" input l2 write"])
        self.l2_weight_read = np.array(df[" filter l2 read"]) * self.eff_layer_multiplier
        self.l2_weight_write = np.array(df[" filter l2 write"]) * self.eff_layer_multiplier
        self.l2_output_read = 0 # np.array(df[" output l2 read"])
        self.l2_output_write = np.array(df[" output l2 write"]) * self.eff_layer_multiplier
        
        self.L2_DRAM_BW = np.min((np.array(df[" Offchip BW Req (Elements/cycle)"]), self.dram_bw))
        self.L1_L2_BW = np.min((np.array(df[" NoC BW Req (Elements/cycle)"]), self.noc_bw))
        # self.L1_DRAM_BW = self.output_size / self.num_mac_cycles
        # self.L2_vertical_BW = np.array(df[" Vertical L2 BW Req (Elements/cycle)"])

        self.avg_noc_bw = np.array(df[" Avg BW Req"])

        if self.no_l2:
            ### assign l2 read/writes to dram
            self.dram_input_read = self.l2_input_read
            self.dram_input_write = self.l2_input_write

            self.dram_output_read = self.l2_output_read
            self.dram_output_write = self.l2_output_write

            self.dram_weight_read = self.l2_weight_read
            self.dram_weight_write = self.l2_weight_write
        
            self.l2_input_read = 0
            self.l2_input_write = 0

            self.l2_output_read = 0
            self.l2_output_write = 0

            self.l2_weight_read = 0
            self.l2_weight_write = 0

            self.L2_DRAM_BW = 0
            self.L1_DRAM_BW = self.L1_L2_BW
            self.L1_L2_BW = 0
        
        else:
            self.dram_input_read = self.l2_input_write
            self.dram_input_write = 0

            self.dram_output_read = 0
            self.dram_output_write = self.l2_output_read

            self.dram_weight_read = self.l2_weight_write
            self.dram_weight_write = 0

            self.L1_DRAM_BW = 0
            
        self.if_read_data = True

    def mac_area(self):
        self.mac_area = self.num_pe * self.tech_param.mac_area * self.vector_width

    def l1_area(self):
        self.l1_area = self.tech_param.l1_area_per_bit * self.l1_size * self.num_pe
    
    def l2_area(self):
        self.l2_area = self.tech_param.l2_area_per_bit * self.l2_size
    
    def num_reads_l1(self):
        self.l1_num_read = self.bit_param.input_bit * self.l1_input_read + self.bit_param.output_bit * self.l1_output_read + self.bit_param.weight_bit * self.l1_weight_read

    def num_reads_l2(self):
        self.l2_num_read = self.bit_param.input_bit * self.l2_input_read + self.bit_param.output_bit * self.l2_output_read + self.bit_param.weight_bit * self.l2_weight_read
    
    def num_reads_DRAM(self):
        self.dram_num_read = self.bit_param.input_bit * self.dram_input_read + self.bit_param.output_bit * self.dram_output_read + self.bit_param.weight_bit * self.dram_weight_read
    
    def num_writes_l1(self):
        self.l1_num_write = self.bit_param.input_bit * self.l1_input_write + self.bit_param.output_bit * self.l1_output_write + self.bit_param.weight_bit * self.l1_weight_write

    def num_writes_l2(self):
        self.l2_num_write = self.bit_param.input_bit * self.l2_input_write + self.bit_param.output_bit * self.l2_output_write + self.bit_param.weight_bit * self.l2_weight_write
    
    def num_writes_DRAM(self):
        self.dram_num_write = self.bit_param.input_bit * self.dram_input_write + self.bit_param.output_bit * self.dram_output_write + self.bit_param.weight_bit * self.dram_weight_write

    def num_bits_l1_l2(self):
        self.total_l1_l2_bits = self.bit_param.input_bit * self.l2_input_read + self.bit_param.weight_bit * self.l2_weight_read + self.bit_param.output_bit * self.l2_output_write
        
    def num_bits_l1_dram(self):
        if self.no_l2:
            self.total_l1_dram_bits = self.bit_param.input_bit * self.dram_input_read + self.bit_param.weight_bit * self.dram_weight_read + self.bit_param.output_bit * self.dram_output_write
        else:
            self.total_l1_dram_bits = 0 * u.bit
    
    def num_bits_l2_dram(self):
        self.total_l2_dram_bits = self.bit_param.input_bit * self.l2_input_write + self.bit_param.weight_bit * self.l2_weight_write + self.bit_param.output_bit * self.l2_output_read

    def calc_area(self):

        if not(self.if_read_data):
            print("Error: Data not read")
            return
        
        self.mac_area()
        self.l1_area()
        self.l2_area()
    
    def calc_dataflow_param(self):

        if not(self.if_read_data):
            print("Error: Data not read")
            return
        
        self.num_reads_l1()
        self.num_reads_l2()
        self.num_reads_DRAM()
        self.num_writes_l1()
        self.num_writes_l2()
        self.num_writes_DRAM()
        self.num_bits_l1_l2()
        self.num_bits_l1_dram()
        self.num_bits_l2_dram()