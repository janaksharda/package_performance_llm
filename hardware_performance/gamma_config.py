    # parser.add_argument('--fitness1', type=str, default="latency", choices=('latency', 'energy', 'power', 'EDP', 'area'), help='First objective')
    # parser.add_argument('--fitness2', type=str, default="energy", choices=('latency', 'energy', 'power', 'EDP', 'area'), help='Second objective')
    # parser.add_argument('--num_pop', type=int, default=20,help='Number of populations')
    # parser.add_argument('--parRS', default=False, action='store_true', help='Parallize across R S dimension')
    # parser.add_argument('--epochs', type=int, default=2, help='Number of epochs (i.e., Numbers of generations)')
    # parser.add_argument('--outdir', type=str, default="outdir", help='Output directiory')
    # parser.add_argument('--num_pe', type=int, default=1024, help='Number of PEs')
    # parser.add_argument('--l1_size', type=int, default=-1, help='L1 size (local buffer size)')
    # parser.add_argument('--l2_size', type=int, default=-1, help='L2 size (global buffer size)')
    # parser.add_argument('--NocBW', type=int, default=-1, help='Network-on-Chip BW')
    # parser.add_argument('--offchipBW', type=int, default=-1, help='Off-chip BW')
    # parser.add_argument('--hwconfig', type=str, default=None, help='HW configuration file')
    # parser.add_argument('--model', type=str, default="resnet18", help='Model to run')
    # parser.add_argument('--num_layer', type=int, default=2, help='Number of layers to optimize')
    # parser.add_argument('--singlelayer', type=int, default=0, help='The layer index to optimize')
    # parser.add_argument('--slevel_min', type=int, default=2, help='Minimum number of parallelization level')
    # parser.add_argument('--slevel_max', type=int, default=2, help='Maximum number of parallelization level')
    # parser.add_argument('--fixedCluster', type=int, default=0, help='Rigid cluster size')
    # parser.add_argument('--log_level', type=int, default=1, help='Detail: 2, runtimeinfo: 1')
    # parser.add_argument('--costmodel_cstr', type=str, default=None, help='Constraint from Cost model')
    # parser.add_argument('--mapping_cstr', type=str, default=None, help='Mapping constraint')
    # parser.add_argument('--accel_cstr', type=str, default=None, help='Constraint from the HW type configuration of the accelerator under design')
    # parser.add_argument('--area_budget', type=float, default=-1, help='The area budget (mm2). Set to -1 if no area upper-bound')
    # parser.add_argument('--pe_limit', type=int, default=-1, help='Number of Processing Element budget. Set to -1 if no num_PE upper-bound')
    # parser.add_argument('--use_factor',default=False, action='store_true', help='To only use factor as tile size.')

import astropy.units as u   
# from model_param import model_param

class gamma_config:

    def __init__(self,
            fitness = ["ranking", "layer_latency", "energy"],
            # fitness = ["layer_latency"],
            constraints = {"power": 200},
            num_pop = 50,
            parRS = False,
            epochs = 50,
            outdir = "outdir",
            num_pe = 10000,
            l1_size = -1,
            l2_size = -1,
            NocBW = -1,
            offchipBW = -1,
            hwconfig = None,
            model = "decode_phase_shapes",
            num_layer = 12,
            singlelayer = 0,
            slevel_min = 2,
            slevel_max = 2,
            fixedCluster = 0,
            log_level = 1,
            costmodel_cstr = None,
            mapping_cstr = None,
            accel_cstr = None,
            area_budget = -1,
            pe_limit = -1,
            use_factor = True,
            hardware_config = None,
            model_config = None,
            file_name = None
        ):
        self.fitness = fitness
        self.constraints = constraints
        self.num_pop = num_pop
        self.parRS = parRS
        self.epochs = epochs
        self.outdir = outdir
        self.num_pe = num_pe
        self.l1_size = l1_size
        self.l2_size = l2_size
        self.NocBW = NocBW
        self.offchipBW = offchipBW
        self.hwconfig = hwconfig
        self.model = model if model_config is None else model_config.model_name
        self.num_layer = num_layer
        self.singlelayer = singlelayer
        self.slevel_min = slevel_min
        self.slevel_max = slevel_max
        self.fixedCluster = fixedCluster
        self.log_level = log_level
        self.costmodel_cstr = costmodel_cstr
        self.mapping_cstr = mapping_cstr
        self.accel_cstr = accel_cstr
        self.area_budget = area_budget
        self.pe_limit = pe_limit
        self.use_factor = use_factor
        self.vector_width = 16
        self.file_name = file_name

        if hardware_config is not None:
            self.hardware_config = hardware_config
            self.num_pe = int(hardware_config.num_pe)
            self.l1_size = int(hardware_config.l1_size / u.B)
            self.l2_size = int(hardware_config.l2_size / u.B)
            self.NocBW = int(hardware_config.noc_channels)
            self.offchipBW = int(hardware_config.dram_bw)
            self.vector_width = int(hardware_config.vector_width)
            self.constraints["noc_bw"] = self.NocBW
            self.constraints["dram_bw"] = self.offchipBW

        self.model_config = model_config# if model_config is not None else model_param()  # Assuming model_param is defined elsewhere
    