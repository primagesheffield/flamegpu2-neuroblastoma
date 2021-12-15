#include "header.h"

const bool ENSEMBLE = true;
int main(int argc, const char ** argv) {
    flamegpu::ModelDescription model("PRIMAGE: Neuroblastoma");
    defineModel(model);

    if (!ENSEMBLE) {
        flamegpu::CUDASimulation cuda_model(model, argc, argv);
#ifdef VISUALISATION
        flamegpu::visualiser::ModelVis&  m_vis = defineVisualisation(model, cuda_model);
#endif
        /**
         * Execution
         */
        cuda_model.simulate();

#ifdef VISUALISATION
        m_vis.join();
#endif
    } else {
        /**
         * Load LHC_hetNB.csv
         */
        std::vector<std::string> column_names;
        std::map<std::string, std::vector<float>> columns;
        const std::string csv_path = "../../../inputs/LHC_hetNB.csv";
        std::ifstream csv_file(csv_path);
        if (!csv_file.is_open()) {
            fprintf(stderr, "Unable to open %s\n", csv_path.c_str());
            return EXIT_FAILURE;
        }
        // Read file line by line
        std::string line, word;
        // Extract the first line in the file, to make headers
        {
            std::getline(csv_file, line);
            std::stringstream ss(line);
            while (std::getline(ss, word, ',')) {
                columns.insert({ word, std::vector<float>{} });
                column_names.push_back(word);
            }
        }
        // Process the remaining lines
        while (std::getline(csv_file, line)) {
            std::stringstream ss(line);
            int i = 0;
            while (std::getline(ss, word, ',')) {
                columns[column_names[i]].push_back(static_cast<float>(stod(word)));
                ++i;
            }
        }
        csv_file.close();
        /**
         * Create a run plan
         */
        const unsigned int CONFIG_COUNT = static_cast<unsigned int>(columns["Index"].size());
        const unsigned int RUNS_PER_CONFIG = 10;
        flamegpu::RunPlanVector runs(model, CONFIG_COUNT * RUNS_PER_CONFIG);
        {
            runs.setSteps(3024);
            for (unsigned int j = 0; j < RUNS_PER_CONFIG; ++j) {
                for (unsigned int i = 0; i < CONFIG_COUNT; ++i) {
                    const unsigned int ij = i * RUNS_PER_CONFIG + j;
                    runs[ij].setRandomSimulationSeed((j+12) * 84673);  // Something something prime number
                    runs[ij].setProperty<float>("O2", columns["O2"][i]);
                    runs[ij].setProperty<float>("cellularity", columns["cellularity"][i]);
                    runs[ij].setProperty<float>("theta_sc", columns["theta_sc"][i]);
                    runs[ij].setProperty<float>("degdiff", columns["degdiff"][i]);
                    for (int c = 0; c < 23; ++c) {
                        std::stringstream ss;
                        ss << "fraction " << (c+1);
                        runs[ij].setProperty<float>("clones_dummy", c, columns[ss.str()][i]);
                    }
                }
            }
        }
        /**
         * Create a logging config
         */
        flamegpu::StepLoggingConfig step_log_cfg(model);
        {
            step_log_cfg.setFrequency(1);
            step_log_cfg.logEnvironment("O2");
            step_log_cfg.logEnvironment("Nscl_count");
            step_log_cfg.logEnvironment("Nnbl_count");
            step_log_cfg.logEnvironment("NB_living_count");
            step_log_cfg.logEnvironment("NB_living_degdiff_average");
        }
        /**
         * Create Model Runner
         */
        flamegpu::CUDAEnsemble cuda_ensemble(model, argc, argv);
        cuda_ensemble.Config().concurrent_runs = 1;
        cuda_ensemble.Config().devices = { 0 };
        cuda_ensemble.Config().out_directory = "ensemble_out";
        cuda_ensemble.Config().out_format = "json";
        cuda_ensemble.setStepLog(step_log_cfg);
        cuda_ensemble.simulate(runs);
    }


    return 0;
}
