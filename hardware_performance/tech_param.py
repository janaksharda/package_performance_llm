import astropy.units as u
import numpy as np


class Tech_param:

    def __init__(self, tech_node, model_param, dram):

        self.datatype = model_param.bit_param.datatype
        self.sram_subarray_bits = [256, 256]
        self.tech_node = tech_node
        self.feature_size = 14 * u.nm

        ### 45nm
        ### INT8-ADD: 0.03 pJ; 36 um^2, INT8-MULT: 0.2 pJ;282 um^2
        ### FP16-ADD: 0.4 pJ; 1360 um^2, FP16-MULT: 1.1 pJ; 1640 um^2

        int8_45nm_add_energy = 0.03 * u.pJ
        int8_45nm_mult_energy = 0.2 * u.pJ
        fp16_45nm_add_energy = 0.4 * u.pJ
        fp16_45nm_mult_energy = 1.1 * u.pJ

        int8_45nm_add_area = 36 * u.um ** 2
        int8_45nm_mult_area = 282 * u.um ** 2
        fp16_45nm_add_area = 1360 * u.um ** 2
        fp16_45nm_mult_area = 1640 * u.um ** 2


        if tech_node == 14:
            if self.datatype == 'int8':
                self.multiplier_energy = int8_45nm_mult_energy * (1.0/8.0)
                self.adder_energy = int8_45nm_add_energy * (1.0/8.0)
                self.mac_energy = self.adder_energy + self.multiplier_energy

                self.multiplier_area = int8_45nm_mult_area * (1.0/8.0)
                self.adder_area = int8_45nm_add_area * (1.0/8.0)

                self.mac_area_per_unit = self.multiplier_area + self.adder_area

                self.sram_cell_size = 300 * self.feature_size * self.feature_size / u.bit
                self.sram_subarray_size = self.sram_subarray_bits[0] * self.sram_subarray_bits[1] * u.bit * self.sram_cell_size
                
                self.metal_pitch = 100 * u.nm
                self.num_routing_layers = 3

                # self.noc_energy_per_bit_per_mm = 0.1 * u.pJ / u.bit / u.mm

                self.cost_per_area = 1 / u.mm ** 2
        
        neurosim_45_7_logic_scaling_area = 0.07386
        neurosim_45_7_logic_scaling_dynamic_power = 0.06349

        if tech_node <= 7:
            if self.datatype == 'int8':
                self.multiplier_energy = int8_45nm_mult_energy * neurosim_45_7_logic_scaling_dynamic_power
                self.adder_energy = int8_45nm_add_energy * neurosim_45_7_logic_scaling_dynamic_power
                self.mac_energy = self.adder_energy + self.multiplier_energy

                self.multiplier_area = int8_45nm_mult_area * neurosim_45_7_logic_scaling_area
                self.adder_area = int8_45nm_add_area * neurosim_45_7_logic_scaling_area

                self.mac_area_per_unit = self.multiplier_area + self.adder_area

                # self.sram_cell_size = 300 * self.feature_size * self.feature_size / u.bit * 1.0/2.0 ### 4MB/mm2
            if self.datatype == 'fp16':
                self.multiplier_energy = fp16_45nm_mult_energy * neurosim_45_7_logic_scaling_dynamic_power
                self.adder_energy = fp16_45nm_add_energy * neurosim_45_7_logic_scaling_dynamic_power
                self.mac_energy = self.adder_energy + self.multiplier_energy

                self.multiplier_area = fp16_45nm_mult_area * neurosim_45_7_logic_scaling_area
                self.adder_area = fp16_45nm_add_area * neurosim_45_7_logic_scaling_area

                self.mac_area_per_unit = self.multiplier_area + self.adder_area

            # self.sram_cell_size = 300 * self.feature_size * self.feature_size / u.bit * 1.0/2.0 ### 4MB/mm2
            

            self.sram_cell_size          = 0.0632 * u.mm ** 2 / (2 * u.Mbit) #self.num_bank_2Mb   
            self.sram_subarray_size = self.sram_subarray_bits[0] * self.sram_subarray_bits[1] * u.bit * self.sram_cell_size

            # self.GB_read_latency         = 0.2708 * 1e-9                               
            # self.GB_write_latency        = 0.2708 * 1e-9                      
            # self.GB_read_energy_per_bit  = 0.0376 * 1e-12   
            # self.GB_write_energy_per_bit = 0.0369 * 1e-12  
            # self.GB_standby_power        = 985.92 * 1e-6 * self.num_bank_2Mb # ~400pA/SRAM cell
            # self.GB_standby_power        = 0.5/0.7 * 985.92 * 1e-6 * self.num_bank_2Mb # ~400pA/SRAM cell      
            
            self.metal_pitch = 100 * u.nm
            self.num_routing_layers = 3

            self.noc_energy_per_bit_per_mm = 0.1 * u.pJ / u.bit / u.mm

            self.cost_per_area = 2 / u.mm ** 2
        
        if tech_node == 5:
            neurosim_scale_factor_logic_dynamic_power = 0.80
            neurosim_scale_factor_logic_area = 0.65
            neurosim_scale_factor_logic_leakage_power = 0.80

            neurosim_scale_factor_dff_read_energy = 0.81
            neurosim_scale_factor_dff_write_energy = 0.81
            neurosim_scale_factor_dff_area = 0.67

            neurosim_scale_factor_sram_area = 0.773
            neurosim_scale_factor_sram_read_energy = 0.924
            neurosim_scale_factor_sram_write_energy = 0.913

        if tech_node == 3:
            neurosim_scale_factor_logic_dynamic_power = 0.71
            neurosim_scale_factor_logic_area = 0.45
            neurosim_scale_factor_logic_leakage_power = 0.71

            neurosim_scale_factor_dff_read_energy = 0.72
            neurosim_scale_factor_dff_write_energy = 0.72
            neurosim_scale_factor_dff_area = 0.51

            neurosim_scale_factor_sram_area = 0.678
            neurosim_scale_factor_sram_read_energy = 0.758
            neurosim_scale_factor_sram_write_energy = 0.724
        
        if tech_node == 2:
            neurosim_scale_factor_logic_dynamic_power = 0.46
            neurosim_scale_factor_logic_area = 0.37
            neurosim_scale_factor_logic_leakage_power = 0.46

            neurosim_scale_factor_dff_read_energy = 0.34
            neurosim_scale_factor_dff_write_energy = 0.37
            neurosim_scale_factor_dff_area = 0.38

            neurosim_scale_factor_sram_area = 0.746
            neurosim_scale_factor_sram_read_energy = 0.733
            neurosim_scale_factor_sram_write_energy = 0.638
        
        if tech_node == 1:
            neurosim_scale_factor_logic_dynamic_power = 0.37
            neurosim_scale_factor_logic_area = 0.26
            neurosim_scale_factor_logic_leakage_power = 0.37

            neurosim_scale_factor_dff_read_energy = 0.32
            neurosim_scale_factor_dff_write_energy = 0.33
            neurosim_scale_factor_dff_area = 0.23

            neurosim_scale_factor_sram_area = 0.424
            neurosim_scale_factor_sram_read_energy = 0.475
            neurosim_scale_factor_sram_write_energy = 0.453
        
        if tech_node < 7:
            self.multiplier_energy = self.multiplier_energy * neurosim_scale_factor_logic_dynamic_power
            self.adder_energy = self.adder_energy * neurosim_scale_factor_logic_dynamic_power
            self.mac_energy = self.adder_energy + self.multiplier_energy

            self.multiplier_area = self.multiplier_area * neurosim_scale_factor_logic_area
            self.adder_area = self.adder_area * neurosim_scale_factor_logic_area
            self.mac_area_per_unit = self.multiplier_area + self.adder_area

            self.sram_cell_size = self.sram_cell_size * neurosim_scale_factor_sram_area
            # self.sram_subarray_size = self.sram_subarray_size * neurosim_scale_factor_sram_area
            
        
        if dram == 'fine_grained_hbm':
            self.dram_epb = 0.9 * u.pJ / u.bit
            self.cost_dram = 100
            self.dram_read_energy = self.dram_epb
            self.dram_write_energy = self.dram_epb
        
        if dram == 'fine_grained_hbm1':
            self.dram_epb = 0.9 * u.pJ / u.bit + 5.5 * 0.1 * u.pJ / u.bit
            self.cost_dram = 100
            self.dram_read_energy = self.dram_epb
            self.dram_write_energy = self.dram_epb

        if dram == 'hbm3':
            self.dram_epb = 3 * u.pJ / u.bit
            self.cost_dram = 150
            self.dram_read_energy = self.dram_epb
            self.dram_write_energy = self.dram_epb
        
        if dram == '1b_node':
            self.dram_read_energy = 0.9 * u.pJ / u.bit
            self.dram_write_energy = 0.9 * u.pJ / u.bit
            self.cost_dram = 100
        
        if dram == 'h1t1c_vwl_64':
            self.dram_read_energy = 3.18 * u.pJ / u.bit
            self.dram_write_energy = 3.18 * u.pJ / u.bit
            self.cost_dram = 100
        
        if dram == 'h1t1c_vwl_128':
            self.dram_read_energy = 3.37 * u.pJ / u.bit
            self.dram_write_energy = 3.37 * u.pJ / u.bit
            self.cost_dram = 100
        
        if dram == 'h1t1c_vbl_64':
            self.dram_read_energy = 3.26 * u.pJ / u.bit
            self.dram_write_energy = 3.26 * u.pJ / u.bit
            self.cost_dram = 100
        
        if dram == 'h1t1c_vbl_128':
            self.dram_read_energy = 3.54 * u.pJ / u.bit
            self.dram_write_energy = 3.54 * u.pJ / u.bit
            self.cost_dram = 100
        
        if dram == 'gct_vwl_64':
            self.dram_read_energy = 6.26 * u.pJ / u.bit
            self.dram_write_energy = 9.22 * u.pJ / u.bit
            self.cost_dram = 100
        
        if dram == 'gct_vwl_128':
            self.dram_read_energy = 7.18 * u.pJ / u.bit
            self.dram_write_energy = 10.90 * u.pJ / u.bit
            self.cost_dram = 100

        if dram == 'gct_vbl_64':
            self.dram_read_energy = 2.76 * u.pJ / u.bit
            self.dram_write_energy = 3.5 * u.pJ / u.bit
            self.cost_dram = 100

        if dram == 'gct_vbl_128':
            self.dram_read_energy = 3.05 * u.pJ / u.bit
            self.dram_write_energy = 3.79 * u.pJ / u.bit
            self.cost_dram = 100

# 3.12 3.26 3.54 3.12 3.26 3.54

        if dram == "si_32":
            self.dram_read_energy = 1.90 * u.pJ / u.bit
            self.dram_write_energy = 1.90 * u.pJ / u.bit
            self.cost_dram = 100
        
        if dram == "si_64":
            self.dram_read_energy = 2.00 * u.pJ / u.bit
            self.dram_write_energy = 2.00 * u.pJ / u.bit
            self.cost_dram = 100
        
        if dram == "si_128":
            self.dram_read_energy = 2.19 * u.pJ / u.bit
            self.dram_write_energy = 2.19 * u.pJ / u.bit
            self.cost_dram = 100
        
        if dram == "iwo_32":
            self.dram_read_energy = 1.90 * u.pJ / u.bit
            self.dram_write_energy = 1.90 * u.pJ / u.bit
            self.cost_dram = 100
        
        if dram == "iwo_64":
            self.dram_read_energy = 2.00 * u.pJ / u.bit
            self.dram_write_energy = 2.00 * u.pJ / u.bit
            self.cost_dram = 100
        
        if dram == "iwo_128":
            self.dram_read_energy = 2.19 * u.pJ / u.bit
            self.dram_write_energy = 2.19 * u.pJ / u.bit
            self.cost_dram = 100

        if dram == "fram":
            self.dram_read_energy = 2.0 * u.pJ / u.bit
            self.dram_write_energy = 2.0 * u.pJ / u.bit
            self.cost_dram = 100
        
        if dram == "ltram_128":
            self.dram_read_energy = 0.11/128 * u.pJ / u.bit
            self.dram_write_energy = 0.11/128 * u.pJ / u.bit
            self.cost_dram = 100
        
        if dram == "ltram_256":
            self.dram_read_energy = 0.54/256 * u.pJ / u.bit
            self.dram_write_energy = 0.54/256 * u.pJ / u.bit
            self.cost_dram = 100
        
        if dram == "ltram_512":
            self.dram_read_energy = 2.4/512 * u.pJ / u.bit
            self.dram_write_energy = 2.4/512 * u.pJ / u.bit
            self.cost_dram = 100
        
        if dram == "ltram_1024":
            self.dram_read_energy = 9.55/1024 * u.pJ / u.bit
            self.dram_write_energy = 9.55/1024 * u.pJ / u.bit
            self.cost_dram = 100
        



    def sram_memory(self, sram_size):

        num_subarrays = sram_size / (self.sram_subarray_bits[0] * self.sram_subarray_bits[1] * u.bit)
        subarray_length = (self.sram_subarray_size) ** 0.5
        
        num_step = np.ceil(np.log2(num_subarrays))
        num_step = int(num_step)

        if num_step // 2 == 1:
            n_1 = (num_step -  1) / 2
            length_wire = subarray_length * (2*(2**n_1 -1) + 0.5)
        else:
            n_1 = (num_step - 2) / 2
            length_wire = subarray_length * (2*(2**n_1 -1) + 2 ** n_1)
        
        access_energy = length_wire * 10 * u.fJ / u.bit / u.mm

        
        if self.tech_node == 14:
            self.l1_read_energy = 1 * u.fJ / u.bit + access_energy
            self.l2_read_energy = 37 * u.fJ / u.bit + access_energy
            self.l1_write_energy = 1 * u.fJ / u.bit + access_energy
            self.l2_write_energy = 37 * u.fJ / u.bit + access_energy

            self.l1_area_per_bit = self.sram_cell_size
            self.l2_area_per_bit = self.sram_cell_size
        
        # if self.tech_node == 3:
        #     self.l1_read_energy = 37 * u.fJ / u.bit * 1.0/2.0 
        #     self.l2_read_energy = 37 * u.fJ / u.bit * 1.0/2.0 + access_energy
        #     # self.l2_read_energy = 1.83 * u.pJ / u.bit#1 * u.fJ / u.bit * 1.0/2.0 + access_energy
        #     self.l1_write_energy = 37 * u.fJ / u.bit * 1.0/2.0 
        #     self.l2_write_energy = 37 * u.fJ / u.bit * 1.0/2.0 + access_energy
        #     # self.l2_write_energy = 10.5 * u.pJ / u.bit#1 * u.fJ / u.bit * 1.0/2.0 + access_energy

        #     self.l1_area_per_bit = self.sram_cell_size
        #     self.l2_area_per_bit = self.sram_cell_size
        
        if self.tech_node <= 7:
            self.l1_read_energy = 1 * u.fJ / u.bit
            self.l2_read_energy = 37 * u.fJ / u.bit + access_energy
            self.l1_write_energy = 1 * u.fJ / u.bit
            self.l2_write_energy = 37 * u.fJ / u.bit + access_energy

            self.l1_area_per_bit = self.sram_cell_size
            self.l2_area_per_bit = self.sram_cell_size
        
        if self.tech_node == 5:
            neurosim_scale_factor_sram_area = 0.773
            neurosim_scale_factor_sram_read_energy = 0.924
            neurosim_scale_factor_sram_write_energy = 0.913

        if self.tech_node == 3:
            neurosim_scale_factor_sram_area = 0.678
            neurosim_scale_factor_sram_read_energy = 0.758
            neurosim_scale_factor_sram_write_energy = 0.724
        
        if self.tech_node == 2:
            neurosim_scale_factor_sram_area = 0.746
            neurosim_scale_factor_sram_read_energy = 0.733
            neurosim_scale_factor_sram_write_energy = 0.638
        
        if self.tech_node == 1:
            neurosim_scale_factor_sram_area = 0.424
            neurosim_scale_factor_sram_read_energy = 0.475
            neurosim_scale_factor_sram_write_energy = 0.453

        if self.tech_node <= 7:
            self.l1_read_energy = self.l1_read_energy * neurosim_scale_factor_sram_read_energy
            self.l2_read_energy = self.l2_read_energy * neurosim_scale_factor_sram_read_energy
            self.l1_write_energy = self.l1_write_energy * neurosim_scale_factor_sram_write_energy
            self.l2_write_energy = self.l2_write_energy * neurosim_scale_factor_sram_write_energy

            self.l1_area_per_bit = self.l1_area_per_bit * neurosim_scale_factor_sram_area
            self.l2_area_per_bit = self.l2_area_per_bit * neurosim_scale_factor_sram_area


        

        



        


