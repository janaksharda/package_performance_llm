import astropy.units as u
from model_operations import get_decode_phase_shapes, get_fine_tune_shapes, get_prefill_shapes
import numpy as np

class model_param:
    def __init__(self, batch_size = 1, dram_type = '1b', phase = 'fine_tune', num_accelerator = 12, recompute = True, model_name = 'llama2_7b', seq_len = 2048):   

        self.model_name = model_name
        self.phase = phase
        self.bit_param = def_bit_param()

        self.seq_len_cache = seq_len

        if self.model_name == "random":
            self.num_layers = 40
            self.layer_multiplier = [4, 40, 40, 2, 1]
            self.batch_size = batch_size
            self.h_dim = np.random.randint(5120, 8192)
            self.num_heads = np.random.randint(32, 40)
            self.intermediate_dim = np.random.randint(1024, 13824)

        if self.phase == 'fine_tune' or self.phase == "training":
            self.bit_param = def_bit_param(input_bit = 16 * u.bit,
                  output_bit = 16 * u.bit,
                  weight_bit = 16 * u.bit,
                  datatype = 'fp16')

        self.recompute = recompute
        print(self.recompute)
        if self.recompute:
            self.backprop_store = 'none'
        else:
            self.backprop_store = 'all'


        if self.model_name == 'llama2_13b':
            self.num_layers = 40
            if self.phase == 'decode' or self.phase == 'prefill':
                self.layer_multiplier = [4, 40, 40, 2, 1]
            elif self.phase == 'training':
                self.layer_multiplier = [8, 80, 80, 4, 2, 1, 3, 2, 2, 40, 3, 3] 
            
            if self.recompute:
                self.layer_multiplier = [8, 80, 80, 4, 2, 1, 3, 2, 2, 40, 3, 3]            
            if self.phase != "prefill":
                 self.seq_len_cache = 4096
            self.batch_size = batch_size
            self.h_dim = 5120
            self.num_heads = 40
            self.intermediate_dim = 13824
        elif self.model_name == 'llama2_7b':
            self.num_layers = 32
            self.batch_size = batch_size
            self.layer_multiplier = [4, 32, 32, 2, 1]
            if self.phase == 'fine_tune':
                self.layer_multiplier = [8, 64, 32, 4, 1, 1, 3, 2, 2, 32, 3, 3]
            if self.recompute:
                self.layer_multiplier = [8, 64, 64, 4, 1, 1, 3, 2, 2, 32, 3, 3]  ## 1b            
            if self.phase != "prefill":
                 self.seq_len_cache = 4096
            self.h_dim = 4096
            self.num_heads = 32
            self.intermediate_dim = 13824
        elif self.model_name == 'llama2_70b':
            self.num_layers = 80
            self.layer_multiplier = [4, 80, 80, 2, 1]            
            if self.phase != "prefill":
                 self.seq_len_cache = 4096
            self.batch_size = 32
            self.h_dim = 8192
            self.num_heads = 80
            self.intermediate_dim = 28672
        elif self.model_name == 'nvidia_1p7b':
            self.num_layers = 24
            self.layer_multiplier = [3, 24, 24, 1, 1]            
            if self.phase != "prefill":
                 self.seq_len_cache = 2048
            if self.recompute:
                self.layer_multiplier = [6, 48, 48, 2, 2, 1, 2, 1, 2, 24, 3, 3]
            self.batch_size = batch_size
            self.h_dim = 2304
            self.num_heads = 24
            self.intermediate_dim = 4 * self.h_dim
        elif self.model_name == 'nvidia_3p6b':
            self.num_layers = 30
            self.layer_multiplier = [3, 32, 32, 1, 1]
            if self.recompute:
                self.layer_multiplier = [6, 64, 64, 2, 2, 1, 2, 1, 2, 32, 3, 3]            
            if self.phase != "prefill":
                 self.seq_len_cache = 2048
            self.batch_size = batch_size
            self.h_dim = 2304
            self.num_heads = 24
            self.intermediate_dim = 4 * self.h_dim
        elif self.model_name == 'nvidia_7p5b':
            self.num_layers = 36
            self.layer_multiplier = [3, 36, 36, 1, 1]
            if self.recompute:
                self.layer_multiplier = [6, 72, 72, 2, 2, 1, 2, 1, 2, 36, 3, 3]            
            if self.phase != "prefill":
                 self.seq_len_cache = 2048
            self.batch_size = batch_size
            self.h_dim = 4096
            self.num_heads = 32
            self.intermediate_dim = 4 * self.h_dim
        elif self.model_name == 'nvidia_18p4b':
            self.num_layers = 40
            self.layer_multiplier = [4, 40, 40, 2, 1]
            if self.recompute:
                self.layer_multiplier = [8, 80, 80, 4, 2, 1, 3, 2, 2, 40, 3, 3]            
            if self.phase != "prefill":
                 self.seq_len_cache = 2048
            self.batch_size = batch_size
            self.h_dim = 6144
            self.num_heads = 48
            self.intermediate_dim = 4 * self.h_dim
        ####
        elif self.model_name == 'nvidia_39p1b':
            self.num_layers = 24
            self.total_layers = 48
            self.layer_multiplier = [4, int(self.total_layers), int(self.total_layers), 2, 1]
            if self.recompute:
                self.layer_multiplier = [8, int(2*self.total_layers), int(2*self.total_layers), 4, 2, 1, 3, 2, 2, int(self.total_layers), 3, 3]            
            if self.phase != "prefill":
                 self.seq_len_cache = 2048
            self.batch_size = batch_size
            self.h_dim = 8192
            self.num_heads = 64
            self.intermediate_dim = 4 * self.h_dim
        elif self.model_name == 'nvidia_76p1b':
            self.num_layers = 18
            self.total_layers = 60
            self.layer_multiplier = [4, int(self.total_layers), int(self.total_layers), 2, 1]
            if self.recompute:
                self.layer_multiplier = [8, int(2*self.total_layers), int(2*self.total_layers), 4, 2, 1, 3, 2, 2, int(self.total_layers), 3, 3]            
            if self.phase != "prefill":
                 self.seq_len_cache = 2048
            self.batch_size = batch_size
            self.h_dim = 10240
            self.num_heads = 80
            self.intermediate_dim = 4 * self.h_dim
        elif self.model_name == 'nvidia_145p6b':
            self.num_layers = 12
            self.total_layers = 80
            self.layer_multiplier = [4, int(self.total_layers), int(self.total_layers), 2, 1]
            if self.recompute:
                self.layer_multiplier = [8, int(2*self.total_layers), int(2*self.total_layers), 4, 2, 1, 3, 2, 2, int(self.total_layers), 3, 3]            
            if self.phase != "prefill":
                 self.seq_len_cache = 2048
            self.batch_size = batch_size
            self.h_dim = 12288
            self.num_heads = 96
            self.intermediate_dim = 4 * self.h_dim
        elif self.model_name == 'nvidia_310p1b':
            self.num_layers = 7
            self.total_layers = 96
            self.layer_multiplier = [4, int(self.total_layers), int(self.total_layers), 2, 1]
            if self.recompute:
                self.layer_multiplier = [8, int(2*self.total_layers), int(2*self.total_layers), 4, 2, 1, 3, 2, 2, int(self.total_layers), 3, 3]            
            if self.phase != "prefill":
                 self.seq_len_cache = 2048
            self.batch_size = batch_size
            self.h_dim = 16384
            self.num_heads = 128
            self.intermediate_dim = 4 * self.h_dim
        elif self.model_name == 'nvidia_529p6b':
            self.num_layers = 4
            self.total_layers = 105
            self.layer_multiplier = [4, int(self.total_layers), int(self.total_layers), 2, 1]
            if self.recompute:
                self.layer_multiplier = [8, int(2*self.total_layers), int(2*self.total_layers), 4, 2, 1, 3, 2, 2, int(self.total_layers), 3, 3]            
            if self.phase != "prefill":
                 self.seq_len_cache = 2048
            self.batch_size = batch_size
            self.h_dim = 16384
            self.num_heads = 128
            self.intermediate_dim = 4 * self.h_dim
        # elif self.model_name == 'nvidia_529p6':
        #     self.num_layers = 
        #     self.total_layers = 
        #     self.layer_multiplier = [4, int(self.total_layers), int(self.total_layers), 2, 1]
        #     if self.recompute:
        #         self.layer_multiplier = [8, int(2*self.total_layers), int(2*self.total_layers), 4, 2, 1, 3, 2, 2, int(self.total_layers), 3, 3]            
        #     if self.phase != "prefill":
        #          self.seq_len_cache = 2048
        #     self.batch_size = batch_size
        #     self.h_dim = 
        #     self.num_heads = 128
        #     self.intermediate_dim = 4 * self.h_dim
        
        # if self.phase == 'decode':
        #     self.store_output = [True, True, True, False, False]
        #     self.store_input = [False, False, False, False, False]
        #     self.store_weight = [True, True, True, True, True]
        #     self.store_output_l2 = [True, True, True, True, True]
        #     self.store_input_l2 = [True, True, True, True, True]
        #     self.store_weight_l2 = [True, True, True, True, True]
        if self.phase == 'decode':
            self.store_output = [True, True, True, False, False]
            self.store_input = [False, False, False, False, False]
            self.store_weight = [True, True, True, True, True]
            self.store_output_l2 = [True, True, True, True, True]
            self.store_input_l2 = [True, True, True, True, True]
            self.store_weight_l2 = [True, True, True, True, True]
        elif self.phase == 'prefill':
            self.store_output = [True, False, True, False, False]
            self.store_input = [False, False, True, False, False]
            self.store_weight = [True, False, False, True, True]
            self.store_output_l2 = [True, True, True, True, True]
            self.store_input_l2 = [True, True, True, True, True]
            self.store_weight_l2 = [True, True, True, True, True]

        elif self.phase == 'training':
            self.store_output = [False, False, False, False, True, False, False, False, False, False, False, False]
            self.store_input = [False, False, False, False, False, True, True, True, True, False, True, True]
            self.store_weight = [True, False, False, True, True, True, True, True, True, True, True, True]
            self.store_output_l2 = [True, True, True, True, True, True, True, True, True, True, True, True]
            self.store_input_l2 = [True, True, True, True, True, True, True, True, True, True, True, True]
            self.store_weight_l2 = [True, True, True, True, True, True, True, True, True, True, True, True]

        
        check_len = (len(self.store_output) != len(self.layer_multiplier)) + \
                    (len(self.store_input) != len(self.layer_multiplier)) + \
                    (len(self.store_weight) != len(self.layer_multiplier)) + \
                    (len(self.store_output_l2) != len(self.layer_multiplier)) + \
                    (len(self.store_input_l2) != len(self.layer_multiplier)) + \
                    (len(self.store_weight_l2) != len(self.layer_multiplier))
        if check_len:
            raise ValueError('Length of store_output, store_input, store_weight, store_output_l2, store_input_l2, store_weight_l2 should be equal to length of layer_multiplier')


        if self.phase == 'decode':
            self.model_name = 'decode_phase_shapes' + str(self.batch_size) 
            self.file_path = '../gamma/data/model/' + self.model_name + '.csv'

            get_decode_phase_shapes(seq_len_cache = self.seq_len_cache,
                                    batch_size = self.batch_size,
                                    h_dim = self.h_dim,
                                    num_heads = self.num_heads,
                                    intermediate_dim = self.intermediate_dim,
                                    file_path = self.file_path)
        
        elif self.phase == 'training':
            self.file_path = '../gamma/data/model/' + self.model_name + '.csv'

            get_fine_tune_shapes(seq_len = self.seq_len_cache,
                                    batch_size = self.batch_size,
                                    h_dim = self.h_dim,
                                    num_heads = self.num_heads,
                                    intermediate_dim = self.intermediate_dim,
                                    file_path = self.file_path)
        elif self.phase == 'prefill':
            self.model_name = 'prefill_shapes' + str(self.batch_size) 
            self.file_path = '../gamma/data/model/' + self.model_name + '.csv'

            get_prefill_shapes(seq_len = self.seq_len_cache,
                                    batch_size = self.batch_size,
                                    h_dim = self.h_dim,
                                    num_heads = self.num_heads,
                                    intermediate_dim = self.intermediate_dim,
                                    file_path = self.file_path)
        
class def_bit_param:

    def __init__(self,
                  input_bit = 8 * u.bit,
                  output_bit = 16 * u.bit,
                  weight_bit = 8 * u.bit,
                  datatype = 'int8'
                  ):
        self.input_bit = input_bit
        self.output_bit = output_bit
        self.weight_bit = weight_bit
        self.datatype = datatype