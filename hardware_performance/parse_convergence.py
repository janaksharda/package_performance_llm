import numpy as np

def parse_convergence_file(filename):
    area_values = []
    energy_values = []
    latency_values = []
    
    with open(filename, 'r') as file:
        lines = file.readlines()
        
    for i, line in enumerate(lines):
        if "Package Information" in line:
            # Search for values in the next few lines after "Package Information"
            for j in range(i, min(i+20, len(lines))):
                if "--Total Area:" in lines[j]:
                    area = float(lines[j].split(':')[1].split()[0].split(' ')[0])
                    area_values.append(area)
                elif "--Total Energy:" in lines[j]:
                    energy = float(lines[j].split(':')[1].split()[0].split(' ')[0])
                    energy_values.append(energy)
                elif "--Latency:" in lines[j]:
                    latency = float(lines[j].split(':')[1].strip().split(' ')[0])
                    latency_values.append(latency)
    
    return area_values, energy_values, latency_values

def write_output_file(area_values, energy_values, latency_values, output_filename):
    with open(output_filename, 'w') as f:
        # Initialize lists for fitness1 and fitness2
        runs_fitness1 = []  # odd numbered runs
        fitness1_latency_values = []
        fitness1_area_values = []
        fitness1_energy_values = []
        runs_fitness2 = []  # even numbered runs
        fitness2_latency_values = []
        fitness2_area_values = []
        fitness2_energy_values = []
        
        for i in range(len(area_values)):
            if i % 2 == 0:  # even numbered runs
                runs_fitness2.append(f"run[{i//2}] {area_values[i]} {energy_values[i]} {latency_values[i]}")
                fitness2_latency_values.append(latency_values[i])
                fitness2_area_values.append(area_values[i])
                fitness2_energy_values.append(energy_values[i])


            else:  # odd numbered runs
                runs_fitness1.append(f"run[{i//2}] {area_values[i]} {energy_values[i]} {latency_values[i]}")
                fitness1_latency_values.append(latency_values[i])
                fitness1_area_values.append(area_values[i])
                fitness1_energy_values.append(energy_values[i])
        
        # Write fitness1 (odd numbered runs)
        f.write("fitness1\n")
        for run in runs_fitness1:
            f.write(run + "\n")
        
        # Write fitness2 (even numbered runs)
        f.write("\nfitness2\n")
        for run in runs_fitness2:
            f.write(run + "\n")
        
        return fitness1_latency_values, fitness1_area_values, fitness1_energy_values, fitness2_latency_values, fitness2_area_values, fitness2_energy_values

def main():
    input_file = "convergence.out"
    output_file = "parsed_convergence.txt"
    
    try:
        area_values, energy_values, latency_values = parse_convergence_file(input_file)
        fitness1_latency_values, fitness1_area_values, fitness1_energy_values, fitness2_latency_values, fitness2_area_values, fitness2_energy_values = write_output_file(area_values, energy_values, latency_values, output_file)
        print(f"Successfully parsed {input_file} and wrote results to {output_file}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    # run1_f1 = fitness1_values[0:49]
    # run2_f1 = fitness1_values[50:99]
    # run3_f1 = fitness1_values[100:149]
    # run4_f1 = fitness1_values[150:199]
    # run1_f2 = fitness2_values[0:49]
    # run2_f2 = fitness2_values[50:99]
    # run3_f2 = fitness2_values[100:149]
    # run4_f2 = fitness2_values[150:199]

    run1_f1_latency = fitness1_latency_values[0:49]
    run2_f1_latency = fitness1_latency_values[50:99]
    run3_f1_latency = fitness1_latency_values[100:149]
    run4_f1_latency = fitness1_latency_values[150:199]
    run1_f1_area = fitness1_area_values[0:49]
    run2_f1_area = fitness1_area_values[50:99]
    run3_f1_area = fitness1_area_values[100:149]
    run4_f1_area = fitness1_area_values[150:199]
    run1_f1_energy = fitness1_energy_values[0:49]
    run2_f1_energy = fitness1_energy_values[50:99]
    run3_f1_energy = fitness1_energy_values[100:149]
    run4_f1_energy = fitness1_energy_values[150:199]
    run1_f2_latency = fitness2_latency_values[0:49]
    run2_f2_latency = fitness2_latency_values[50:99]
    run3_f2_latency = fitness2_latency_values[100:149]
    run4_f2_latency = fitness2_latency_values[150:199]
    run1_f2_area = fitness2_area_values[0:49]
    run2_f2_area = fitness2_area_values[50:99]
    run3_f2_area = fitness2_area_values[100:149]
    run4_f2_area = fitness2_area_values[150:199]
    run1_f2_energy = fitness2_energy_values[0:49]
    run2_f2_energy = fitness2_energy_values[50:99]
    run3_f2_energy = fitness2_energy_values[100:149]
    run4_f2_energy = fitness2_energy_values[150:199]

    ### print mean and std for each run
    # print("run1_f1 mean: ", np.mean(run1_f1_latency), np.mean(run1_f1_area), np.mean(run1_f1_energy))
    print("run1_f1 std: ", np.std(run1_f1_latency), np.std(run1_f1_area), np.std(run1_f1_energy))
    # print("run2_f1 mean: ", np.mean(run2_f1_latency), np.mean(run2_f1_area), np.mean(run2_f1_energy))
    print("run2_f1 std: ", np.std(run2_f1_latency), np.std(run2_f1_area), np.std(run2_f1_energy))
    # print("run3_f1 mean: ", np.mean(run3_f1_latency), np.mean(run3_f1_area), np.mean(run3_f1_energy))
    print("run3_f1 std: ", np.std(run3_f1_latency), np.std(run3_f1_area), np.std(run3_f1_energy))
    # print("run4_f1 mean: ", np.mean(run4_f1_latency), np.mean(run4_f1_area), np.mean(run4_f1_energy))
    print("run4_f1 std: ", np.std(run4_f1_latency), np.std(run4_f1_area), np.std(run4_f1_energy))
    # print("run1_f2 mean: ", np.mean(run1_f2_latency), np.mean(run1_f2_area), np.mean(run1_f2_energy))
    print("run1_f2 std: ", np.std(run1_f2_latency), np.std(run1_f2_area), np.std(run1_f2_energy))
    # print("run2_f2 mean: ", np.mean(run2_f2_latency), np.mean(run2_f2_area), np.mean(run2_f2_energy))
    print("run2_f2 std: ", np.std(run2_f2_latency), np.std(run2_f2_area), np.std(run2_f2_energy))
    # print("run3_f2 mean: ", np.mean(run3_f2_latency), np.mean(run3_f2_area), np.mean(run3_f2_energy))
    print("run3_f2 std: ", np.std(run3_f2_latency), np.std(run3_f2_area), np.std(run3_f2_energy))
    # print("run4_f2 mean: ", np.mean(run4_f2_latency), np.mean(run4_f2_area), np.mean(run4_f2_energy))
    print("run4_f2 std: ", np.std(run4_f2_latency), np.std(run4_f2_area), np.std(run4_f2_energy))


    # fitness1 = [run1_f1, run2_f1, run3_f1, run4_f1]
    # fitness2 = [run1_f2, run2_f2, run3_f2, run4_f2]

    fitness1 = [run1_f1_latency, run2_f1_latency, run3_f1_latency, run4_f1_latency, run1_f1_area, run2_f1_area, run3_f1_area, run4_f1_area, run1_f1_energy, run2_f1_energy, run3_f1_energy, run4_f1_energy]
    fitness2 = [run1_f2_latency, run2_f2_latency, run3_f2_latency, run4_f2_latency, run1_f2_area, run2_f2_area, run3_f2_area, run4_f2_area, run1_f2_energy, run2_f2_energy, run3_f2_energy, run4_f2_energy]

    ## save the fitness values to a csv file
    import pandas as pd
    fitness1_df = pd.DataFrame(fitness1)
    fitness2_df = pd.DataFrame(fitness2)
    fitness1_df.to_csv('fitness1.csv', index=False)
    fitness2_df.to_csv('fitness2.csv', index=False)

if __name__ == "__main__":
    main()