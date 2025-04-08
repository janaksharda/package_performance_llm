## read and print all the pickle files present in this directory
import os
import pickle
from astropy import units as u

mode = "decode"
model_name = ["nvidia_1p7b", "nvidia_3p6b", "nvidia_7p5b", "nvidia_18p4b", "llama2_7b", "llama2_13b", "nvidia_39p1b", "nvidia_76p1b", "nvidia_145p6b", "nvidia_310p1b", "nvidia_529p6b"]
batch_size = [8, 16, 32, 64, 128]
model_name = model_name[5]
batch_size = batch_size[2]

# folder = "/final_results_llama2_7b_batch_size_4/"
# folder = "/nvidia_18p4b_8batch/"

folder = "/" + model_name + "_" + str(batch_size) + "batch/"

os.chdir(os.getcwd() + folder)

def read_pickle_files(filename, keys):
    '''
    kwargs: list of keys, read filename.pkl in a dict "dict", and return dict[keys[0]][keys[1]]... '''
    
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    for key in keys:
        data = data[key]
    return data

def read_all__files(keys, component_type):
    ## save a csv file with key as filename, and values as the data returned from read_pickle_files
    filenames = os.listdir()
    data = {}
    for filename in filenames:
        if filename.endswith('.pkl') and component_type in filename and mode in filename:
            data_with_unit = read_pickle_files(filename, keys)
            unit = data_with_unit.unit
            data[filename+'_'+str(unit)] = data_with_unit.value
    ## save the data in a csv file
    filename = ''
    for key in keys:
        filename += '_' + key
    filename = filename[1:]
    with open(filename + '_' + mode + '.csv', 'w') as f:
        for key in data:
            f.write(key + ', ' + str(data[key]) + '\n')
    return data

def save_ppa_data():
    '''
    Save PPA (Power, Performance, Area) data in CSV format with organized hierarchy of designs
    '''
    # Define the metrics we want to extract
    metrics = ["Latency", "Total Area", "Total Energy"]
    
    # Define the design patterns to look for, in specific order
    design_patterns = [
        "baseline",
        "l2cache_3d",
        "central_new_l2_2p5d",
        "decentral_200_with_l2_1_1p0",
        "decentral_200_without_l2_1_1p0",
        "slt"
    ]
    
    # Initialize results dictionary
    results = {}
    
    # Get all pickle files containing 'package' and 'training'
    files = [f for f in os.listdir() if f.endswith('.pkl') and 'package' in f and mode in f]
    
    # Sort files according to design patterns
    sorted_files = []
    for pattern in design_patterns:
        for file in files:
            if pattern in file:
                sorted_files.append(file)
    
    # Extract data for each file and metric
    for filename in sorted_files:
        results[filename] = {}
        for metric in metrics:
            try:
                data_with_unit = read_pickle_files(filename, [metric])
                results[filename][metric] = f"{data_with_unit.value:.8f}"
            except:
                results[filename][metric] = "N/A"
    
    # Write to CSV file
    with open('ppa_summary_' + mode + '.csv', 'w') as f:
        # Write header
        f.write('Design,' + ','.join(metrics) + '\n')
        
        # Write data rows
        for filename in sorted_files:
            row = [filename]
            for metric in metrics:
                row.append(str(results[filename][metric]))
            f.write(','.join(row) + '\n')
    
    return results

def save_component_wise(metric = 'Area'):
    '''
    Save component-wise data in CSV format with organized hierarchy of designs

    row 1: filename: baseline, l2cache_3d, central_new_l2_2p5d, decentral_200_with_l2_1
    column 1: component names (MAC, L1, L2, NoC, total area)
    fill in the rest as the component_wise area. 
    '''
    # Define the components we want to track
    if metric == 'Area':
        components = ["MAC Area", "L1 Area", "L2 Area", "NoC Area", "Total Area"]
    elif metric == 'Energy':
        components = ["MAC Energy", "L1 Energy", "L2 Energy", "DRAM Energy", "NoC Energy", "Total Energy"]
    elif metric == 'Size':
        components = ["L1 Size", "L2 Size"]
    
    # Define the design patterns in order
    design_patterns = [
        "baseline",
        "l2cache_3d",
        "central_new_l2_2p5d",
        "decentral_200_with_l2_1_1p0",
        "decentral_200_without_l2_1_1p0"
    ]
    
    # Initialize results dictionary
    results = {}
    
    # Get all pickle files containing 'package' and 'training'
    files = [f for f in os.listdir() if f.endswith('.pkl') and 'package' in f and mode in f]
    
    # Sort files according to design patterns
    sorted_files = []
    for pattern in design_patterns:
        for file in files:
            if pattern in file:
                sorted_files.append(file)
                break
    
    # Extract area data for each component and design
    for filename in sorted_files:
        results[filename] = {}
        for component in components:
            try:
                data_with_unit = read_pickle_files(filename, [component])
                results[filename][component] = f"{data_with_unit.value:.8f}"
            except:
                results[filename][component] = "N/A"
    
    # Write to CSV file
    with open('component_wise_' + metric + '_' + mode + '.csv', 'w') as f:
        # Write header row with design names
        f.write('Component,' + ','.join(sorted_files) + '\n')
        
        # Write component rows
        for component in components:
            row = [component]
            for filename in sorted_files:
                row.append(str(results[filename][component]))
            f.write(','.join(row) + '\n')
    
    return results

if __name__ == '__main__':
    # keys = ["Total Area"]
    # data = read_all__files(keys, 'package')

    ## open a single pickle file with 'package' in the name and iterate through all the keys
    key_list = []
    for filename in os.listdir():
        print(filename)
        if filename.endswith('.pkl') and 'package' in filename and mode in filename:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                for key in data.keys():
                    key_list.append(key)
                break
    print(key_list)
    for key in key_list:
        data = read_all__files([key], 'package')
        print(data)
        print('\n\n\n')
    
    # Save PPA data to CSV file
    results = save_ppa_data()

    # Save component-wise area data to CSV file
    results = save_component_wise(metric='Area')
    results = save_component_wise(metric='Energy')
    results = save_component_wise(metric='Size')

