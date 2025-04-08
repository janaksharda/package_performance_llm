import astropy.units as u
from tech_param import Tech_param
# from model_param import model_param
from layer_performance import layer_performance
from interconnect_param import interconnect_epb

class Chiplet:

    def __init__(
            self,
            technology_param = None,
            Model_param = None,
            chiplet_components = ['PE', 'L1'],
            interconnect_type = ['HB'],
            interconnect_dist = {'L1_L2': None, 'L2_DRAM': None, 'L1_DRAM': None},
            l2_dram_routing = True,
            l2_dram_routing_horizontal = True,
            l1_l2_routing_horizontal = True,
            l2_routing_horizontal = False,
            l1_dram_routing_horizontal = False,
            num_cores = 8,
            layer_traces_maestro = None,
            height = None,
            width = None,   
            no_l2 = False,
            bw = [None, None],
            num_horizontal_cores = 1,
            disaggregated_io = False,
            locality_factor = 0.5
        ):

        self.tech_param = technology_param
        self.chiplet_components = chiplet_components
        self.l2_dram_routing = l2_dram_routing
        self.l1_l2_routing_horizontal = l1_l2_routing_horizontal
        self.l2_dram_routing_horizontal = l2_dram_routing_horizontal
        self.l1_dram_routing_horizontal = l1_dram_routing_horizontal
        self.l2_routing_horizontal = l2_routing_horizontal
        self.interconnect_type = interconnect_type
        self.layer_traces_maestro = layer_traces_maestro
        self.num_cores = num_cores
        self.height_assigned = height
        self.width_assigned = width
        self.model_param = Model_param
        self.interconnect_dist = interconnect_dist
        self.no_l2 = no_l2
        self.bw = bw
        self.locality_factor = locality_factor

        self.disaggregated_io = disaggregated_io
        self.num_horizontal_cores = num_horizontal_cores
        self.num_vertical_cores = self.num_cores // self.num_horizontal_cores

        if (self.no_l2) and ('L2' in self.chiplet_components):
            print("Error in Chiplet Configuration")
            assert False
        
        if layer_traces_maestro != None:
            self.assign_layer_performance(layer_traces_maestro)

    def assign_layer_performance(self, layer_traces_maestro):
        self.layer_performance_list = []

        for i, layer_trace in enumerate(layer_traces_maestro):
            if layer_trace is None:
                continue
            self.layer_performance_list.append(layer_performance(layer_trace[0], self.model_param.bit_param, self.tech_param, self.model_param, i, self.bw, no_l2 = self.no_l2))

    def calc_per_core_cum_BW(self):
        self.L1_L2_cum_BW = 0
        self.L2_DRAM_cum_BW = 0
        self.L1_DRAM_cum_BW = 0
        self.L2_vertical_cum_BW = 0

        for layer_performance in self.layer_performance_list:
            if self.disaggregated_io:
                self.L1_L2_cum_BW += layer_performance.L1_L2_BW
                # self.L2_DRAM_cum_BW += layer_performance.L2_DRAM_BW  * self.locality_factor + layer_performance.L2_DRAM_BW * (1-self.locality_factor) * self.num_cores
                self.L2_DRAM_cum_BW += layer_performance.L2_DRAM_BW 
                # self.L1_DRAM_cum_BW += layer_performance.L1_DRAM_BW * self.locality_factor + layer_performance.L1_DRAM_BW * (1-self.locality_factor) * self.num_cores
                self.L1_DRAM_cum_BW += layer_performance.L1_DRAM_BW
                # self.L2_vertical_cum_BW += layer_performance.L2_vertical_BW
            else:
                self.L1_L2_cum_BW += layer_performance.L1_L2_BW * self.num_cores
                self.L2_DRAM_cum_BW += layer_performance.L2_DRAM_BW * self.num_cores
                # self.L2_vertical_cum_BW += layer_performance.L2_vertical_BW
                self.L1_DRAM_cum_BW += layer_performance.L1_DRAM_BW * self.num_cores
        
    
    def calc_data_movement(self):
        self.total_l1_l2_bits = 0 * u.bit
        self.total_l1_dram_bits = 0 * u.bit
        self.total_l2_dram_bits = 0 * u.bit

        for layer_performance in self.layer_performance_list:
            layer_performance.calc_dataflow_param()
            self.total_l1_l2_bits += layer_performance.total_l1_l2_bits
            self.total_l1_dram_bits += layer_performance.total_l1_dram_bits
            self.total_l2_dram_bits += layer_performance.total_l2_dram_bits

    def routing_l1_l2_width(self):
        self.l1_l2_routing_width = 0 * u.mm

        if not(('L1' in self.chiplet_components) or ('L2' in self.chiplet_components)):
            return
        
        self.l1_l2_routing_width = self.L1_L2_cum_BW * self.tech_param.metal_pitch / self.tech_param.num_routing_layers

    def routing_l2_dram_width(self):
        self.l2_dram_routing_width = 0 * u.mm

        if not(('L2' in self.chiplet_components) or ('DRAM' in self.chiplet_components)):
            if not(self.l2_dram_routing):
                return

        self.l2_dram_routing_width = self.L2_DRAM_cum_BW * self.tech_param.metal_pitch / self.tech_param.num_routing_layers
    
    def routing_l1_dram_width(self):
        self.l1_dram_routing_width = 0 * u.mm

        if not(('L1' in self.chiplet_components) or ('DRAM' in self.chiplet_components)):
            return

        self.l1_dram_routing_width = self.L1_DRAM_cum_BW * self.tech_param.metal_pitch / self.tech_param.num_routing_layers
    
    def routing_l2_vertical_width(self):
        self.l2_vertical_width = 0 * u.mm

        if not('L2' in self.chiplet_components):
            return

        # self.l2_vertical_width = self.L2_vertical_cum_BW * self.tech_param.metal_pitch / self.tech_param.num_routing_layers

    def calc_l2_area(self):
        self.total_l2_size_per_core = 0 * u.bit
        self.total_l2_size = 0 * u.bit
        self.l2_area = 0 *u.mm ** 2

        if not ('L2' in self.chiplet_components):
            return 

        for layer_performance in self.layer_performance_list:
            self.total_l2_size_per_core += layer_performance.l2_size
        
        self.total_l2_size = self.total_l2_size_per_core * self.num_cores
        self.tech_param.sram_memory(self.total_l2_size)
        self.l2_area = self.total_l2_size * self.tech_param.l2_area_per_bit

    def calc_l1_area(self):
        self.total_l1_size_per_core = 0 * u.bit
        self.total_l1_size = 0 * u.bit
        self.l1_area = 0 * u.mm ** 2

        if not ('L1' in self.chiplet_components):
            return 
            
        for layer_performance in self.layer_performance_list:
            self.total_l1_size_per_core += layer_performance.l1_size * layer_performance.num_pe
        
        self.total_l1_size = self.total_l1_size_per_core * self.num_cores
        self.tech_param.sram_memory(self.total_l1_size)
        self.l1_area = self.total_l1_size * self.tech_param.l1_area_per_bit

    def calc_mac_area(self):
        self.total_mac_size_per_core = 0 * u.bit
        self.mac_area = 0 * u.mm ** 2
        self.total_num_mac_per_core = 0

        if not ('L1' in self.chiplet_components):
            return 
            
        for layer_performance in self.layer_performance_list:
            self.total_num_mac_per_core += layer_performance.num_pe * layer_performance.vector_width
        
        self.total_num_mac = self.total_num_mac_per_core * self.num_cores
        self.mac_area = self.total_num_mac * self.tech_param.mac_area_per_unit

    

    def adjust_routing_length(self):
        self.total_core_area = 0 * u.mm ** 2

        if 'L1' in self.chiplet_components:
            self.total_core_area += self.mac_area + self.l1_area

        if 'L2' in self.chiplet_components:
            self.total_core_area += self.l2_area
        
        if self.interconnect_type['L1_L2'] in ['TSV', 'HB']:
            if self.disaggregated_io:
                self.l1_l2_routing_width = self.l1_l2_routing_width / self.num_cores 
            else:
                self.l1_l2_routing_width /= 2.0
                    
        if self.interconnect_type['L2_DRAM'] in ['TSV', 'HB']:
            if self.disaggregated_io:
                # print(self.num_cores, self.locality_factor, self.l2_dram_routing_width)
                self.l2_dram_routing_width = self.l2_dram_routing_width / self.num_cores * self.locality_factor + self.l2_dram_routing_width * (1-self.locality_factor)
                # print(self.l2_dram_routing_width)
            else:
                self.l2_dram_routing_width /= 2.0
            
        if self.interconnect_type['L1_DRAM'] in ['TSV', 'HB']:
            if self.disaggregated_io:
                self.l1_dram_routing_width = self.l1_dram_routing_width / self.num_cores * self.locality_factor + self.l1_dram_routing_width * (1-self.locality_factor)
            else:
                self.l1_dram_routing_width /= 2.0
        
        total_width = self.l1_l2_routing_width + self.l2_dram_routing_width + self.l1_dram_routing_width + self.l2_vertical_width

        horizontal_routing_height = self.l1_l2_routing_horizontal * self.l1_l2_routing_width + self.l2_dram_routing_horizontal * self.l2_dram_routing_width + \
                                     self.l1_dram_routing_horizontal * self.l1_dram_routing_width + (1 - self.l1_l2_routing_horizontal) * self.l2_vertical_width
        vertical_routing_width = total_width - horizontal_routing_height


        if self.height_assigned == None and self.width_assigned == None:
            self.length_core = self.total_core_area ** 0.5
            self.height = self.length_core + horizontal_routing_height
            self.width = self.length_core + vertical_routing_width

        elif self.height_assigned != None and self.width_assigned == None:
            self.height_core = self.height_assigned - horizontal_routing_height
            self.width_core = self.total_core_area / self.height_core
            self.width = self.width_core + vertical_routing_width
            self.height = self.height_assigned
        
        elif self.width_assigned != None and self.height_assigned == None:
            self.width_core = self.width_assigned - vertical_routing_width
            self.height_core = self.total_core_area / self.width_core
            self.height = self.height_core + horizontal_routing_height
            self.width = self.width_assigned
        
        else:
            print("Error")

        self.area = self.width * self.height
    

    def routing_energy_l1_l2(self):
        self.l1_l2_routing_energy = 0 * u.pJ
        self.interconnect_link_energy_l1_l2 = 0 * u.pJ
        self.interconnect_drv_energy_l1_l2 = 0 * u.pJ
        self.beol_energy_l1_l2 = 0 * u.pJ

        if not(('L1' in self.chiplet_components) or ('L2' in self.chiplet_components)):
            return
        
        if self.disaggregated_io:
            interconnect_length_beol = self.l1_l2_routing_horizontal * self.width / self.num_horizontal_cores + (1 - self.l1_l2_routing_horizontal) * self.height / self.num_vertical_cores
        else:
            interconnect_length_beol = self.l1_l2_routing_horizontal * self.width + (1 - self.l1_l2_routing_horizontal) * self.height
        if self.interconnect_type['L1_L2'] in ['TSV', 'HB']:
            interconnect_length_beol /= 2.0
        # print("Interconnect Length BEOL: ", interconnect_length_beol)
        beol_energy_l1_l2_per_bit, _ = interconnect_epb('BEOL', wire_length = interconnect_length_beol) 
        self.beol_energy_l1_l2 = beol_energy_l1_l2_per_bit * self.total_l1_l2_bits

        interconnect_link_energy_l1_l2_per_bit, interconnect_drv_energy_l1_l2_per_bit = interconnect_epb(self.interconnect_type['L1_L2'], wire_length = self.interconnect_dist['L1_L2'])
        self.interconnect_link_energy_l1_l2 = interconnect_link_energy_l1_l2_per_bit * self.total_l1_l2_bits
        self.interconnect_drv_energy_l1_l2 = interconnect_drv_energy_l1_l2_per_bit * self.total_l1_l2_bits
    

    def routing_energy_l2_dram(self):
        self.l2_dram_routing_energy = 0 * u.pJ
        self.interconnect_link_energy_l2_dram = 0 * u.pJ
        self.interconnect_drv_energy_l2_dram = 0 * u.pJ
        self.beol_energy_l2_dram = 0 * u.pJ

        if not(('L2' in self.chiplet_components) or ('DRAM' in self.chiplet_components)):
            return
                
        if self.disaggregated_io:
            # interconnect_length_beol = self.l2_dram_routing_horizontal * self.width / self.num_horizontal_cores + (1 - self.l2_dram_routing_horizontal) * self.height / self.num_vertical_cores
            ## include locality factor
            interconnect_length_beol = self.l2_dram_routing_horizontal * self.width / self.num_horizontal_cores + (1 - self.l2_dram_routing_horizontal) * self.height / self.num_vertical_cores
            interconnect_length_beol = interconnect_length_beol * self.locality_factor + interconnect_length_beol * (1-self.locality_factor) * self.num_cores
        else:
            interconnect_length_beol = self.l2_dram_routing_horizontal * self.width + (1 - self.l2_dram_routing_horizontal) * self.height
        if self.interconnect_type['L2_DRAM'] in ['TSV', 'HB']:
            interconnect_length_beol /= 2.0
        # print("Interconnect Length BEOL: ", interconnect_length_beol)
        beol_energy_l2_dram_per_bit, _ = interconnect_epb('BEOL', wire_length = interconnect_length_beol) 
        self.beol_energy_l2_dram = beol_energy_l2_dram_per_bit * self.total_l2_dram_bits

        interconnect_link_energy_l2_dram_per_bit, interconnect_drv_energy_l2_dram_per_bit = interconnect_epb(self.interconnect_type['L2_DRAM'], wire_length = self.interconnect_dist['L2_DRAM'])
        self.interconnect_link_energy_l2_dram = interconnect_link_energy_l2_dram_per_bit * self.total_l2_dram_bits
        self.interconnect_drv_energy_l2_dram = interconnect_drv_energy_l2_dram_per_bit * self.total_l2_dram_bits
                

    def routing_energy_l1_dram(self):
        self.l1_dram_routing_energy = 0 * u.pJ
        self.interconnect_link_energy_l1_dram = 0 * u.pJ
        self.interconnect_drv_energy_l1_dram = 0 * u.pJ
        self.beol_energy_l1_dram = 0 * u.pJ

        if not(('L1' in self.chiplet_components) or ('DRAM' in self.chiplet_components)):
            return
        
        if self.disaggregated_io:
            interconnect_length_beol = self.l1_dram_routing_horizontal * self.width / self.num_horizontal_cores + (1 - self.l1_dram_routing_horizontal) * self.height / self.num_vertical_cores
            interconnect_length_beol = interconnect_length_beol * self.locality_factor + interconnect_length_beol * (1-self.locality_factor) * self.num_cores
        else:
            interconnect_length_beol = self.l1_dram_routing_horizontal * self.width + (1 - self.l1_dram_routing_horizontal) * self.height
        if self.interconnect_type['L1_DRAM'] in ['TSV', 'HB']:
            interconnect_length_beol /= 2.0
        # print("Interconnect Length BEOL: ", interconnect_length_beol)
        beol_energy_l1_dram_per_bit, _ = interconnect_epb('BEOL', wire_length = interconnect_length_beol) 
        self.beol_energy_l1_dram = beol_energy_l1_dram_per_bit * self.total_l1_dram_bits
        interconnect_link_energy_l1_dram_per_bit, interconnect_drv_energy_l1_dram_per_bit = interconnect_epb(self.interconnect_type['L1_DRAM'], wire_length = self.interconnect_dist['L1_DRAM'])
        self.interconnect_link_energy_l1_dram = interconnect_link_energy_l1_dram_per_bit * self.total_l1_dram_bits
        self.interconnect_drv_energy_l1_dram = interconnect_drv_energy_l1_dram_per_bit * self.total_l1_dram_bits
 

    def calc_total_energy(self):
        self.mac_energy = 0 * u.pJ
        self.l1_energy = 0 * u.pJ
        self.l2_energy = 0 * u.pJ
        self.dram_energy = 0 * u.pJ
        self.total_energy = 0 * u.pJ
        self.dram_total_access = 0 * u.bit

        if 'L1' in self.chiplet_components:
            for layer_performance in self.layer_performance_list:
                self.mac_energy += layer_performance.num_mac * self.tech_param.mac_energy

        if 'L1' in self.chiplet_components:
            for layer_performance in self.layer_performance_list:
                self.l1_energy += layer_performance.l1_num_read * self.tech_param.l1_read_energy + layer_performance.l1_num_write * self.tech_param.l1_write_energy

        if 'L2' in self.chiplet_components:
            for layer_performance in self.layer_performance_list:
                self.l2_energy += layer_performance.l2_num_read * self.tech_param.l2_read_energy + layer_performance.l2_num_write * self.tech_param.l2_write_energy

        if 'DRAM' in self.chiplet_components:
            for layer_performance in self.layer_performance_list:
                self.dram_energy += layer_performance.dram_num_read * self.tech_param.dram_read_energy + layer_performance.dram_num_write * self.tech_param.dram_write_energy
                self.dram_total_access += layer_performance.dram_num_read + layer_performance.dram_num_write
        self.routing_energy = self.interconnect_link_energy_l2_dram + self.interconnect_drv_energy_l2_dram + \
                              self.interconnect_link_energy_l1_l2 + self.interconnect_drv_energy_l1_l2 +\
                              self.interconnect_link_energy_l1_dram + self.interconnect_drv_energy_l1_dram 
        self.beol_energy = self.beol_energy_l1_l2 + self.beol_energy_l2_dram + self.beol_energy_l1_dram
        self.routing_energy += self.beol_energy
        self.core_energy = self.mac_energy + self.l1_energy + self.l2_energy + self.dram_energy
        self.total_energy += self.core_energy + self.routing_energy
                              
        

    def calc_latency(self):
        latency_list = []

        for layer_performance in self.layer_performance_list:
            latency_list.append(layer_performance.num_mac_cycles / self.num_cores)
        self.latency = max(latency_list)

    def calc_area(self):

        self.calc_mac_area()
        self.calc_l1_area()
        self.calc_l2_area()

        self.calc_per_core_cum_BW()
        self.calc_data_movement()

        self.routing_l1_l2_width()
        self.routing_l2_dram_width()
        self.routing_l1_dram_width()
        self.routing_l2_vertical_width()

        self.adjust_routing_length()
    
    def calc_energy(self):
        self.routing_energy_l1_l2()
        self.routing_energy_l2_dram()
        self.routing_energy_l1_dram()

        self.calc_total_energy()
    
    def calc_cost(self):

        if ('PE' in self.chiplet_components) or ('L1' in self.chiplet_components) or ('L2' in self.chiplet_components):
            self.cost = self.area * self.tech_param.cost_per_area
        elif 'DRAM' in self.chiplet_components:
            self.cost = self.tech_param.cost_dram

            
             
    def print_info(self):

        info_str = (
            f"Chiplet Components: {self.chiplet_components}\n"
            f"--Total Area: {self.area.to(u.mm ** 2)}\n"
            f"----MAC Area: {self.mac_area.to(u.mm ** 2)}\n"
            f"----L1 Area: {self.l1_area.to(u.mm ** 2)}\n"
            f"----L2 Area: {self.l2_area.to(u.mm ** 2)}\n"
            f"----> L1_L2 Routing Width: {self.l1_l2_routing_width.to(u.mm)}\n"
            f"----> L2_DRAM Routing Width: {self.l2_dram_routing_width.to(u.mm)}\n"
            f"----> L1_DRAM Routing Width: {self.l1_dram_routing_width.to(u.mm)}\n"
            f"----> L2 Vertical Routing Width: {self.l2_vertical_width.to(u.mm)}\n"
            f"--Total Energy: {self.total_energy.to(u.mJ)}\n"
            f"----MAC Energy: {self.mac_energy.to(u.mJ)}\n"
            f"----L1 Energy: {self.l1_energy.to(u.mJ)}\n"
            f"----L2 Energy: {self.l2_energy.to(u.mJ)}\n"
            f"----DRAM Energy: {self.dram_energy.to(u.mJ)}\n"
            f"----Routing Energy: {self.routing_energy.to(u.mJ)}\n"
            f"----Core Energy: {self.core_energy.to(u.mJ)}\n"
            f"------L1_L2_interconnect Energy: {(self.interconnect_link_energy_l1_l2 + self.interconnect_drv_energy_l1_l2).to(u.mJ)}\n"
            f"------L2_DRAM_interconnect Energy: {(self.interconnect_link_energy_l2_dram + self.interconnect_drv_energy_l2_dram).to(u.mJ)}\n"
            f"------L1_DRAM_interconnect Energy: {(self.interconnect_link_energy_l1_dram + self.interconnect_drv_energy_l1_dram).to(u.mJ)}\n"
            f"------BEOL Energy: {self.beol_energy.to(u.mJ)}\n"
            f"--Latency: {self.latency}\n"
            f"--Cost: {self.cost}\n"
            f"--L1_L2 Cumulative BW: {self.L1_L2_cum_BW}\n"
            f"--L2_DRAM Cumulative BW: {self.L2_DRAM_cum_BW}\n"
            f"--L1_DRAM Cumulative BW: {self.L1_DRAM_cum_BW}\n"
            f"----Total L1_L2 Bits: {self.total_l1_l2_bits.to(u.GB)}\n"
            f"----Total L1_DRAM Bits: {self.total_l1_dram_bits.to(u.GB)}\n"
            f"----Total L2_DRAM Bits: {self.total_l2_dram_bits.to(u.GB)}\n"
            f"----Total L2 Size: {self.total_l2_size.to(u.MB)}\n"
            f"----Total L1 Size: {self.total_l1_size.to(u.kB)}\n"
        )

        ## make a dictionary and return it
        dict_info = {
            'Chiplet Components': self.chiplet_components,
            'Total Area': self.area.to(u.mm ** 2),
            'MAC Area': self.mac_area.to(u.mm ** 2),
            'L1 Area': self.l1_area.to(u.mm ** 2),
            'L2 Area': self.l2_area.to(u.mm ** 2),
            'L1_L2 Routing Width': self.l1_l2_routing_width.to(u.mm),
            'L2_DRAM Routing Width': self.l2_dram_routing_width.to(u.mm),
            'L1_DRAM Routing Width': self.l1_dram_routing_width.to(u.mm),
            'L2 Vertical Routing Width': self.l2_vertical_width.to(u.mm),
            'Total Energy': self.total_energy.to(u.mJ),
            'MAC Energy': self.mac_energy.to(u.mJ),
            'L1 Energy': self.l1_energy.to(u.mJ),
            'L2 Energy': self.l2_energy.to(u.mJ),
            'DRAM Energy': self.dram_energy.to(u.mJ),
            'Routing Energy': self.routing_energy.to(u.mJ),
            'Core Energy': self.core_energy.to(u.mJ),
            'L1_L2_interconnect Energy': (self.interconnect_link_energy_l1_l2 + self.interconnect_drv_energy_l1_l2).to(u.mJ),
            'L2_DRAM_interconnect Energy': (self.interconnect_link_energy_l2_dram + self.interconnect_drv_energy_l2_dram).to(u.mJ),
            'L1_DRAM_interconnect Energy': (self.interconnect_link_energy_l1_dram + self.interconnect_drv_energy_l1_dram).to(u.mJ),
            'BEOL Energy': self.beol_energy.to(u.mJ),
            'Latency': self.latency,
            'Cost': self.cost,
            'L1_L2 Cumulative BW': self.L1_L2_cum_BW,
            'L2_DRAM Cumulative BW': self.L2_DRAM_cum_BW,
            'L1_DRAM Cumulative BW': self.L1_DRAM_cum_BW,
            'Total L1_L2 Bits': self.total_l1_l2_bits.to(u.GB),
            'Total L1_DRAM Bits': self.total_l1_dram_bits.to(u.GB),
            'Total L2_DRAM Bits': self.total_l2_dram_bits.to(u.GB),
            'Total L2 Size': self.total_l2_size.to(u.MB),
            'Total L1 Size': self.total_l1_size.to(u.kB)
        }

        print(info_str)
        return dict_info
        




