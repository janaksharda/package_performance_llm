from chiplet import Chiplet
from tech_param import Tech_param
# from model_param import model_param
import astropy.units as u
import pandas as pd
from utilities import arrange_pandas_df


class Package:

    def __init__(self,
                  chiplet_list = [],
                  interconnect_list = [],
                  interconnect_dist = []
        ):
        self.chiplet_list = chiplet_list
        # self.interconnect_assignment = interconnect_assignment(chiplet_list, interconnect_list, interconnect_dist)

    def assign_layer_performance(self, layer_traces_maestro):
        for chiplet in self.chiplet_list:
            chiplet.assign_layer_performance(layer_traces_maestro)
    
    def system_level_analysis(self, print_info = False):
        ## use chiplet.py file to get chiplet level PPA and add it

        self.mac_area = 0 * u.mm ** 2
        self.mac_energy = 0 * u.pJ

        self.l1_area = 0 * u.mm ** 2
        self.l2_area = 0 * u.mm ** 2

        self.l1_size = 0 * u.kB
        self.l2_size = 0 * u.MB

        self.l1_energy = 0 * u.pJ
        self.l2_energy = 0 * u.pJ

        self.dram_energy = 0 * u.pJ

        self.total_area = 0 * u.mm ** 2
        self.total_energy = 0 * u.pJ

        self.latency_list = []
        self.dram_bw_list = []
        self.noc_bw_list = []

        self.cost = 0
        # chiplet_str = ""
        # package_str = ""
        package_dict = {}
        chiplet_dict = {}
        dict_info = {}

        chiplet_dict = {}

        for chiplet in self.chiplet_list:
            chiplet.calc_area()
            chiplet.calc_energy()
            chiplet.calc_latency()
            chiplet.calc_cost()

            self.mac_area += chiplet.mac_area

            self.l1_area += chiplet.l1_area
            self.l2_area += chiplet.l2_area
            self.total_area += chiplet.area

            self.mac_energy += chiplet.mac_energy
            self.l1_energy += chiplet.l1_energy
            self.l2_energy += chiplet.l2_energy
            self.dram_energy += chiplet.dram_energy

            self.l1_size += chiplet.total_l1_size
            self.l2_size += chiplet.total_l2_size



            self.latency_list.append(chiplet.latency)

            self.dram_bw_list.append(chiplet.L2_DRAM_cum_BW)
            self.noc_bw_list.append(chiplet.L1_L2_cum_BW)

            self.total_energy += chiplet.total_energy
            self.cost += chiplet.cost


            if print_info:
                # chiplet_str += chiplet.print_info()
                ## chiplet_name  =  sum of all chiplet.component
                chiplet_name = ""
                for component in chiplet.chiplet_components:
                    chiplet_name += component + "_"

                chiplet_dict[chiplet_name] = chiplet.print_info()

        self.latency = max(self.latency_list) * u.ns
        self.dram_bw = max(self.dram_bw_list)
        self.noc_bw = max(self.noc_bw_list)

        if print_info:
            package_dict = self.print_info()
            
        return chiplet_dict, package_dict
    
    def print_info(self):
        info_str = (
            f"Package Information\n"
            f"--Total Area: {self.total_area.to(u.mm ** 2)}\n"
            f"----MAC Area: {self.mac_area.to(u.mm ** 2)}\n"
            f"----L1 Area: {self.l1_area.to(u.mm ** 2)}\n"
            f"----L2 Area: {self.l2_area.to(u.mm ** 2)}\n"
            f"--Total Energy: {self.total_energy.to(u.mJ)}\n"
            f"----MAC Energy: {self.mac_energy.to(u.mJ)}\n"
            f"----L1 Energy: {self.l1_energy.to(u.mJ)}\n"
            f"----L2 Energy: {self.l2_energy.to(u.mJ)}\n"
            f"----DRAM Energy: {self.dram_energy.to(u.mJ)}\n"
            f"--Latency: {self.latency.to(u.ms)}\n"
            f"--L1 Size: {self.l1_size.to(u.kB)}\n"
            f"--L2 Size: {self.l2_size.to(u.MB)}\n"
            f"--Cost: {self.cost}\n"
        )
        print(info_str)

        ### make a dictionary and return it
        dict_info = {
            "Total Area": self.total_area.to(u.mm ** 2),
            "MAC Area": self.mac_area.to(u.mm ** 2),
            "L1 Area": self.l1_area.to(u.mm ** 2),
            "L2 Area": self.l2_area.to(u.mm ** 2),
            "Total Energy": self.total_energy.to(u.mJ),
            "MAC Energy": self.mac_energy.to(u.mJ),
            "L1 Energy": self.l1_energy.to(u.mJ),
            "L2 Energy": self.l2_energy.to(u.mJ),
            "DRAM Energy": self.dram_energy.to(u.mJ),
            "Latency": self.latency.to(u.ms),
            "L1 Size": self.l1_size.to(u.kB),
            "L2 Size": self.l2_size.to(u.MB),
            "Cost": self.cost,
        }
        return dict_info
        # return info_str


# if __name__ == "__main__":
#     sample_package = Package()

#     model_param_1 = model_param()
#     layer_traces_maestro = arrange_pandas_df(pd.read_csv("best_soln_llama2_13b.csv"))

#     interconnect_assignment = {"L1_DRAM": "BEOL", "L1_L2": "HB", "L2_DRAM": "2.5D_Si", "L1_DRAM:": "2.5D_Si"}
#     interconnect_dist = {"L1_DRAM": 0 * u.mm, "L1_L2": 0.1 * u.mm, "L2_DRAM": 0.1* u.mm, "L1_DRAM": 0.1 * u.mm}
#     technology_param_1 = Tech_param(14, model_param_1, "hbm3")

#     l1_chiplet = Chiplet(
#         technology_param = technology_param_1,
#         Model_param = model_param_1,
#         chiplet_components = ['PE', 'L1'],
#         interconnect_type = interconnect_assignment,
#         interconnect_dist = interconnect_dist,
#         l2_dram_routing_horizontal = True,
#         l1_l2_routing_horizontal = True,
#         l2_routing_horizontal = False,
#         l1_dram_routing_horizontal = False,
#         num_cores = 8,
#         layer_traces_maestro = layer_traces_maestro,
#         height = None,
#         width = None,
#         no_l2 = False,
#     )

#     l2_chiplet = Chiplet(
#         technology_param = technology_param_1,
#         Model_param = model_param_1,
#         chiplet_components = ['L2'],
#         interconnect_type = interconnect_assignment,
#         interconnect_dist = interconnect_dist,
#         l2_dram_routing_horizontal = True,
#         l1_l2_routing_horizontal = True,
#         l2_routing_horizontal = False,
#         l1_dram_routing_horizontal = False,
#         num_cores = 8,
#         layer_traces_maestro = layer_traces_maestro,
#         height = None,
#         width = None,
#         no_l2 = False,
#     )
#     dram_chiplet = Chiplet(
#         technology_param = technology_param_1,
#         Model_param = model_param_1,
#         chiplet_components = ['DRAM'],
#         interconnect_type = interconnect_assignment,
#         interconnect_dist = interconnect_dist,
#         l2_dram_routing_horizontal = True,
#         l1_l2_routing_horizontal = True,
#         l2_routing_horizontal = False,
#         l1_dram_routing_horizontal = False,
#         num_cores = 8,
#         layer_traces_maestro = layer_traces_maestro,
#         height = None,
#         width = None,
#         no_l2 = False,
#     )
#     sample_package.chiplet_list = [l1_chiplet, l2_chiplet, dram_chiplet]
#     sample_package.system_level_analysis()
