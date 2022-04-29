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
        const std::string csv_path = "../../../inputs/LHC_Cal3.csv";
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
	runs.setRandomPropertySeed(34523);  // Ensure that repeated runs use the same Random values to init ALK
        runs.setPropertyUniformRandom<float>("cellularity", 0, 1);
        runs.setPropertyUniformRandom<float>("O2", 0, 1);
        runs.setPropertyUniformRandom<int>("ALK", 0, 2);
        {
            runs.setSteps(336);
            for (unsigned int j = 0; j < RUNS_PER_CONFIG; ++j) {
                for (unsigned int i = 0; i < CONFIG_COUNT; ++i) {
                    const unsigned int ij = i * RUNS_PER_CONFIG + j;
                    runs[ij].setOutputSubdirectory(std::to_string(static_cast<int>(columns["Index"][i])));
                    runs[ij].setRandomSimulationSeed((j+12) * 84673);  // Something something prime number
                    runs[ij].setProperty<float>("scpro_jux", columns["scpro_jux"][i]);
                    runs[ij].setProperty<float>("nbdiff_jux", columns["nbdiff_jux"][i]);
                    runs[ij].setProperty<float>("nbdiff_amount", columns["nbdiff_amount"][i]);
                    runs[ij].setProperty<float>("nbapop_jux", columns["nbapop_jux"][i]);
                    runs[ij].setProperty<float>("P_cycle_sc", columns["P_cycle_sc"][i]);
                    runs[ij].setProperty<float>("P_cycle_nb", columns["P_cycle_nb"][i]);
	            runs[ij].setProperty<int>("histology_init", 0);
	            runs[ij].setProperty<int>("gradiff", 1);
	            runs[ij].setProperty<int>("MYCN_amp", 1);
	            runs[ij].setProperty<int>("TERT_rarngm", 0);
	            runs[ij].setProperty<int>("ATRX_inact", 0);
	            runs[ij].setProperty<int>("ALT", 0);
	            std::array<unsigned int, 336> chemo_start = { 0 };
	            std::array<unsigned int, 336> chemo_end = { 336 };
	            std::array<float, 6> chemo_effects = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
	            runs[ij].setProperty<unsigned int, 336>("chemo_start", chemo_start);
	            runs[ij].setProperty<unsigned int, 336>("chemo_end", chemo_end);
	            runs[ij].setProperty<float, 6>("chemo_effects", chemo_effects);
		}
	    }
        }
        /**
         * Create a logging config
         */
        flamegpu::StepLoggingConfig step_log_cfg(model);
        {
            step_log_cfg.setFrequency(1);
            step_log_cfg.logEnvironment("validation_Nscl");
            step_log_cfg.logEnvironment("validation_Nnbl");
            step_log_cfg.logEnvironment("validation_cellularity");
            step_log_cfg.logEnvironment("validation_tumour_volume");
            step_log_cfg.logEnvironment("grid_dims");
            step_log_cfg.logEnvironment("histogram_nbl");
            step_log_cfg.logEnvironment("histogram_nba");
            step_log_cfg.logEnvironment("histogram_nbn");
            step_log_cfg.logEnvironment("histogram_scl");
            step_log_cfg.logEnvironment("histogram_sca");
            step_log_cfg.logEnvironment("histogram_scn");
            step_log_cfg.agent("GridCell").logSum<int>("has_cells");
            step_log_cfg.agent("GridCell").logSum<int>("has_living_cells");
        }
        /**
         * Create Model Runner
         */
        flamegpu::CUDAEnsemble cuda_ensemble(model, argc, argv);
        cuda_ensemble.Config().concurrent_runs = 1;
        cuda_ensemble.Config().devices = {0};
        cuda_ensemble.Config().out_directory = "sense_calibration_B29FF3BE_v3";
        cuda_ensemble.Config().out_format = "json";
        cuda_ensemble.setStepLog(step_log_cfg);
        cuda_ensemble.simulate(runs);
    }


    return 0;
}
