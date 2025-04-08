import astropy.units as u
from package import Package
# from model_param import model_param
from tech_param import Tech_param
from chiplet import Chiplet

class hardware_config:

    ## specify the hardware config, such as num cores, num pe, dram bw, noc bw, package config, etc
    def __init__(self, num_cores = 64, num_pe = 16384 // 2, dram_bw = 9600, l1_size = 1 * u.kB, 
                 l2_size = 2 * u.MB, no_L2 = True, disaggregated_io = False, locality_factor = 1.0,
                 interconnect_assignment = {"L1_DRAM": "HB", "L1_L2": "BEOL", "L2_DRAM": "2.5D_Si"},
                 interconnect_dist = {"L1_DRAM": 0 * u.mm, "L1_L2": 0.0 * u.mm, "L2_DRAM": 5.5 * u.mm}):
        self.num_cores = num_cores
        # self.num_cores = 16
        # self.num_pe = 16384 // 4 
        self.num_pe = num_pe
        self.vector_width = 16
        ### min(Peak internal bandwidth, package I/O) ==>  min(Peak internal bandwidth/8, package I/O/8) 
        self.dram_bw = dram_bw #int(9600)  ### 1 Gbps/pin * 1e6 pins/mm2; 100mm2 ==> 1.25e7 GBps; with HB not an issue, with TSV: 
        self.noc_channels = int(2 * 10 ** 7) 
        self.l1_size = l1_size
        self.l2_size = l2_size
        self.no_l2 = no_L2
        self.disaggregated_io = disaggregated_io
        self.locality_factor = locality_factor
        self.interconnect_assignment = interconnect_assignment
        self.interconnect_dist = interconnect_dist

        self.package = Package()
        
        ## specify the package config
    
class hardware_config_2p5d(hardware_config):
    def __init__(self, model_param_1, dram_type = "hbm3", num_cores = 64, num_pe = 16384 // 2, dram_bw = 9600, l1_size = 1 * u.kB, 
                 l2_size = 2 * u.MB, no_L2 = True, disaggregated_io = False, locality_factor = 1.0,
                 interconnect_assignment = {"L1_DRAM": "HB", "L1_L2": "BEOL", "L2_DRAM": "2.5D_Si"},
                 interconnect_dist = {"L1_DRAM": 0 * u.mm, "L1_L2": 0.0 * u.mm, "L2_DRAM": 5.5 * u.mm}):
        super().__init__(num_cores = num_cores, num_pe = num_pe, dram_bw = dram_bw, l1_size = l1_size, l2_size = l2_size, 
                         no_L2 = no_L2, disaggregated_io = disaggregated_io, locality_factor = locality_factor,
                         interconnect_assignment = interconnect_assignment, interconnect_dist = interconnect_dist)
        
        # self.package = Package()
        self.package_config_2p5d_l2_dram(model_param_1, dram_type = dram_type)

    def package_config_2p5d_l2_dram(self, model_param_1, dram_type = "hbm3"):
        

        self.model_param_1 = model_param_1
        
        interconnect_assignment = self.interconnect_assignment
        interconnect_dist = self.interconnect_dist
        technology_param_1 = Tech_param(3, self.model_param_1, dram_type)
        self.bw = [self.noc_channels, self.dram_bw]
        disaggregated_io = self.disaggregated_io
        chiplet_list = []
        flag = 0
        if self.no_l2:
            chiplet_components = ['PE', 'L1']
        else:
            if interconnect_assignment['L1_L2'] == 'BEOL':
                chiplet_components = ['PE', 'L1', 'L2']
            else:
                chiplet_components = ['PE', 'L1']
                flag = 1


        
        l1_chiplet = Chiplet(
            technology_param = technology_param_1,
            Model_param = self.model_param_1,
            chiplet_components = chiplet_components,
            interconnect_type = interconnect_assignment,
            interconnect_dist = interconnect_dist,
            l2_dram_routing_horizontal = True,
            l1_l2_routing_horizontal = True,
            l2_routing_horizontal = False,
            l1_dram_routing_horizontal = False,
            num_cores = self.num_cores,
            height = 10 * u.mm,
            width = None,
            no_l2 = self.no_l2,
            bw = self.bw,
            disaggregated_io = disaggregated_io,
            num_horizontal_cores = int(self.num_cores ** 0.5),
            locality_factor=self.locality_factor
            
        )
        chiplet_list.append(l1_chiplet)
        if not(self.no_l2) and flag == 1:
            l2_chiplet = Chiplet(
                technology_param = technology_param_1,
                Model_param = self.model_param_1,
                chiplet_components = ['L2'],
                interconnect_type = interconnect_assignment,
                interconnect_dist = interconnect_dist,
                l2_dram_routing_horizontal = True,
                l1_l2_routing_horizontal = True,
                l2_routing_horizontal = False,
                l1_dram_routing_horizontal = False,
                num_cores = self.num_cores,
                height = 11 * u.mm,
                width = None,
                no_l2 = self.no_l2,
                bw = self.bw
            )
            chiplet_list.append(l2_chiplet)
        dram_chiplet = Chiplet(
            technology_param = technology_param_1,
            Model_param = self.model_param_1,
            chiplet_components = ['DRAM'],
            interconnect_type = interconnect_assignment,
            interconnect_dist = interconnect_dist,
            l2_dram_routing_horizontal = True,
            l1_l2_routing_horizontal = True,
            l2_routing_horizontal = False,
            l1_dram_routing_horizontal = False,
            num_cores = self.num_cores,
            height = 11 * u.mm,
            width = None,
            no_l2 = self.no_l2,
            bw = self.bw,
            disaggregated_io = disaggregated_io,
            num_horizontal_cores = int(self.num_cores ** 0.5),
            locality_factor=self.locality_factor
        )
        chiplet_list.append(dram_chiplet)
        self.package = Package(chiplet_list = chiplet_list)
