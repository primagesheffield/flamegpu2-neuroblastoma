from stat import S_ISREG, ST_CTIME, ST_MODE, ST_MTIME
import os, sys, time, re, math
import shutil
import numpy as np
from matplotlib import cm
from JSON import save
from JSON import EnvMini
import shlex, subprocess
import csv
import json

"""
This script is intended to simplify execution of batch runs of the orchestrated USFD
Reading inputs from <name>.csv and writing outputs to <name>_out.csv
"""
if len(sys.argv) != 3:
    print("python3 %s <input_file> <gpu_count>" % (sys.argv[0]))
    exit()
elif int(sys.argv[2]) > 4 or int(sys.argv[2]) < 1:
    print("%d is an invalid number of GPUs." %(int(sys.argv[2])))
    print("This script doesn't actually check hardware counts, modify script if required.");
    exit()
    
FGPUNB_EXECUTABLE = "build/bin/Release/orchestrator_FGPUNB"
RUN_STRING = "%s -d %d -i \"%s\" --primage \"%s\""
INPUT_CSV = None
with open(sys.argv[1]) as csvfile:
    INPUT_CSV = csv.reader(csvfile)
OUTPUT_CSV_NAME = sys.argv[1].rsplit(".", 1)[0] + "_out.csv"
TEMP_OUTPUT_CSV_NAME = sys.argv[1].rsplit(".", 1)[0] + "_out_t.csv"
INPUT_FILE_PATH = "temp/input_%d.json"
OUTPUT_FILE_PATH = "temp/output_%d.json"
max_threads = int(sys.argv[2]); # This refers to the number of available GPUs to utilise
threads = [None]*max_threads;
thread_j = [None]*max_threads; # Input index
thread_j2 = [None]*max_threads; # Output index
errs = [None]*max_threads;

for i in range(max_threads):
  errs[i] = open("out/err_%d.txt"%(i), "wb");

# Manually create the output CSVs headings
csv_outputs = [[
"index",
"in_seed",
"in_steps",
"in_TERT_rarngm",
"in_ATRX_inact",
"in_V_tumour",
"in_O2",
"in_cellularity_0", "in_cellularity_1", "in_cellularity_2", "in_cellularity_3", "in_cellularity_4", "in_cellularity_5",
"in_orchestrator_time",
"in_MYCN_amp",
"in_ALT",
"in_ALK",
"in_gradiff",
"in_histology_init",
"in_nb_telomere_length_mean",
"in_nb_telomere_length_sd",
"in_sc_telomere_length_mean",
"in_sc_telomere_length_sd",
"in_extent_of_differentiation_mean",
"in_extent_of_differentiation_sd",
"in_nb_necro_signal_mean",
"in_nb_necro_signal_sd",
"in_nb_apop_signal_mean",
"in_nb_apop_signal_sd",
"in_sc_necro_signal_mean",
"in_sc_necro_signal_sd",
"in_sc_apop_signal_mean",
"in_sc_apop_signal_sd",
"in_sc_apop_signal_sd",
"in_drug_effects_0", "in_drug_effects_1", "in_drug_effects_2", "in_drug_effects_3", "in_drug_effects_4", "in_drug_effects_5",
"in_start_effects",
"in_end_effects",
"out_delta_O2",
"out_O2",
"out_delta_ecm",
"out_ecm",
#"out_material_properties", # Not used, static output
#"out_diffusion_coefficient", # Not used, static output
"out_total_volume_ratio_updated",
"out_cellularity_0", "out_cellularity_1", "out_cellularity_2", "out_cellularity_3", "out_cellularity_4", "out_cellularity_5",
"out_tumour_volume",
"out_ratio_VEGF_NB_SC",
"out_nb_telomere_length_mean",
"out_nb_telomere_length_sd",
"out_sc_telomere_length_mean",
"out_sc_telomere_length_sd",
"out_nb_necro_signal_mean",
"out_nb_necro_signal_sd",
"out_nb_apop_signal_mean",
"out_nb_apop_signal_sd",
"out_sc_necro_signal_mean",
"out_sc_necro_signal_sd",
"out_sc_apop_signal_mean",
"out_sc_apop_signal_sd",
"out_extent_of_differentiation_mean",
"out_extent_of_differentiation_sd"]]

#for i in range(RUNS):
#  csv_outputs.append([0 for j in range(len(csv_outputs[0]))]);
  
# Build a dict of columns to map column heading to column index for each csv
d_csv_outputs = {};
for i in range(len(csv_outputs[0])):
  d_csv_outputs[csv_outputs[0][i]] = i;
d_INPUT_CSV = {};
for i in range(len(INPUT_CSV[0])):
  d_INPUT_CSV[INPUT_CSV[0][i]] = i;
  
# Init output csv to 0's
#for row in range(1, len(csv_outputs)):
#  for col in range(len(csv_outputs[row])):
#    csv_outputs[row][col] = 0;  

completed = 0;
started = 0;
"""
@param j Index of configuration being used (row in output file)
@param g Index of gpu
"""
def startRun(j, g) :
    global started;
    # Add a row to the output csv
    thread_j2[g] = len(csv_outputs)
    csv_outputs.append([0 for j in range(len(csv_outputs[0]))]);
    csv_outputs[thread_j2[g]][d_csv_outputs["index"]] = j
    # Generate input file for config
    env_mini = EnvMini();
    seed = 12
    steps = 336
    for i in range(len(INPUT_CSV[0])):
      # Handle the various input cases
      if INPUT_CSV[0][i] == "seed":
        seed = int(INPUT_CSV[j][i])
      elif INPUT_CSV[0][i] == "steps":
        steps = int(INPUT_CSV[j][i])
      elif INPUT_CSV[0][i].startswith("cellularity") or INPUT_CSV[0][i].startswith("drug_effects"):
        split_name = INPUT_CSV[0][i].rsplit('_', 1)
        if not hasattr(env_mini, split_name[0]):
            setattr(env_mini, split_name[0], [0 for i in range(6)])
        t = getattr(env_mini, split_name[0])
        t[int(split_name[1])] = float(INPUT_CSV[j][i])
        setattr(env_mini, split_name[0], t)
      elif float(int(INPUT_CSV[j][i])) == float(INPUT_CSV[j][i]):
        setattr(env_mini, INPUT_CSV[0][i], int(INPUT_CSV[j][i]))
      else:
        setattr(env_mini, INPUT_CSV[0][i], float(INPUT_CSV[j][i]))
      # Copy the value across to the output csv
      csv_outputs[thread_j2[g]][d_csv_outputs["in_"+INPUT_CSV[0][i]]] = INPUT_CSV[j][i]
    # Export the CSV
    save(INPUT_FILE_PATH%(g), env_mini, seed=seed, steps=steps)
    # Launch model
    EXEC_STRING = RUN_STRING %(FGPUNB_EXECUTABLE, g, INPUT_FILE_PATH%(g), OUTPUT_FILE_PATH%(g))
    args = shlex.split(EXEC_STRING)
    threads[g] = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=errs[g])
    thread_j[g] = j;
    started += 1;

"""
@param g Index of gpu
"""
def cleanupRun(g) :
    global completed
    # Detect output files
    file = OUTPUT_FILE_PATH%(g)
    # Open file step file
    with open(file) as json_file:
        data = json.load(json_file)
        # Update output tables
        for key, val in data["primage"]:
            # Special case for cellularity array
            if isinstance(val, list):
                for i in range(val):
                    csv_outputs[thread_j2[g]][d_csv_outputs["out_"+key+"_"+i]] = float(val[i])
            else:
                csv_outputs[thread_j2[g]][d_csv_outputs["out_"+key]] = float(val)

    # Delete input/output files
    #shutil.move(OUTPUT_FILE_PATH%(g), OUTPUT_SPARE_PATH%(thread_j[g]));
    os.remove(OUTPUT_FILE_PATH%(g));
    os.remove(INPUT_FILE_PATH%(g));
    threads[g] = None;
    thread_j[g] = None;
    thread_j2[g] = None;
    completed += 1
    print("\r%d/%d Completed!"%(completed, RUNS), end = '')
    
    # Track how many runs for current J have completed
    # Every 10 runs export updated csv.
    if completed % 10 == 0:        
        csv_writer = csv.writer(open(TEMP_OUTPUT_CSV_NAME, 'w'));
        for j in range(completed + 1):
            csv_writer.writerow(csv_outputs[j]);


# Find a free thread
for j in range(1, len(INPUT_CSV[0])):
    isDone = True;
    while isDone:
        for g in range(max_threads):
            if isinstance(threads[g], subprocess.Popen):
                threads[g].poll();
                if threads[g].returncode != None:
                    # Process has exited
                    cleanupRun(g);
                else:
                    # Process still active
                    continue;
            # Thread is not active, so launch new instance of model
            try:
                startRun(j, g);
                isDone = False
                break;
            except:
                raise
                print("Unexpected error:", sys.exc_info()[0])
                isDone = False

# Wait for all threads to return before exiting
while completed + 1 < len(INPUT_CSV[0]):
    for g in range(max_threads):
        if isinstance(threads[g], subprocess.Popen):
            threads[g].poll();
            if threads[g].returncode != None:
                # Process has exited
                if (threads[g].returncode!=0):
                    errs[g].write(str.encode("Final run on gpu '%d' exited with return code %d\n" %(g, threads[g].returncode)))
                cleanupRun(g);
    
# Output the new output CSVs
csv_writer = csv.writer(open(OUTPUT_CSV_NAME, 'w'));
for row in csv_outputs:
    csv_writer.writerow(row);
os.remove(TEMP_OUTPUT_CSV_NAME%(g))