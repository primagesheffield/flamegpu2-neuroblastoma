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
	/*
	Load the LHC for the hetNB study for v13.9.
	*/
        std::vector<std::string> column_names;
        std::map<std::string, std::vector<float>> columns;
        const std::string csv_path = "../../../inputs/LHC_tt1.csv";
        std::ifstream csv_file(csv_path);
        if (!csv_file.is_open()) {
            fprintf(stderr, "Unable to open %s\n", csv_path.c_str());
            return EXIT_FAILURE;
        }
	//Read the LHC line by line.
        std::string line, word;
	//Use the first line in the LHC file to create headers.
        {
            std::getline(csv_file, line);
            std::stringstream ss(line);
            while (std::getline(ss, word, ',')) {
                columns.insert({ word, std::vector<float>{} });
                column_names.push_back(word);
            }
        }
	//Process the remaining lines.
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
	runs.setRandomPropertySeed(34523);  // Ensure that repeated runs use the same Random values to init gradiff.
	runs.setProperty<int>("gradiff", 0);
        {
            runs.setSteps(3024);
            for (unsigned int j = 0; j < RUNS_PER_CONFIG; ++j) {
                for (unsigned int i = 0; i < CONFIG_COUNT; ++i) {
                    const unsigned int ij = i * RUNS_PER_CONFIG + j;
                    runs[ij].setOutputSubdirectory(std::to_string(static_cast<int>(columns["Index"][i])));
                    runs[ij].setRandomSimulationSeed((j+12) * 84673);  // Something something prime number
	            runs[ij].setProperty<float>("MYCN_tt", columns["MYCN_tt"][i]);
                    runs[ij].setProperty<float>("MAPK_RAS_tt", columns["MAPK_RAS_tt"][i]);
                    runs[ij].setProperty<float>("JAB1_tt", columns["JAB1_tt"][i]);
                    runs[ij].setProperty<float>("CHK1_tt", columns["CHK1_tt"][i]);
                    runs[ij].setProperty<float>("ID2_tt", columns["ID2_tt"][i]);
                    runs[ij].setProperty<float>("IAP2_tt", columns["IAP2_tt"][i]);
                    runs[ij].setProperty<float>("HIF_tt", columns["HIF_tt"][i]);
                    runs[ij].setProperty<float>("BNIP3_tt", columns["BNIP3_tt"][i]);
                    runs[ij].setProperty<float>("VEGF_tt", columns["VEGF_tt"][i]);
                    runs[ij].setProperty<float>("p53_tt", columns["p53_tt"][i]);
                    runs[ij].setProperty<float>("p73_tt", columns["p73_tt"][i]);
                    runs[ij].setProperty<float>("p21_tt", columns["p21_tt"][i]);
                    runs[ij].setProperty<float>("p27_tt", columns["p27_tt"][i]);
                    runs[ij].setProperty<float>("Bcl2_Bclxl_tt", columns["Bcl2_Bclxl_tt"][i]);
                    runs[ij].setProperty<float>("BAK_BAX_tt", columns["BAK_BAX_tt"][i]);
                    runs[ij].setProperty<float>("CAS_tt", columns["CAS_tt"][i]);
                    runs[ij].setProperty<float>("CDS1_tt", columns["CDS1_tt"][i]);
                    runs[ij].setProperty<float>("CDC25C_tt", columns["CDC25C_tt"][i]);
                    runs[ij].setProperty<float>("ALT_tt", columns["ALT_tt"][i]);
                    runs[ij].setProperty<float>("telo_tt", columns["telo_tt"][i]);
		    if (j < 8){
                    	runs[ij].setProperty<float>("O2", 2.0f/72.0f+(30.0f/72.0f)*(j/7.0f));
			runs[ij].setProperty<float>("delivery_tt", 2.0f/72.0f+(30.0f/72.0f)*(j/7.0f));
		    } else if (j == 8) {
                        runs[ij].setProperty<float>("O2", 45.0f/72.0f);
                        runs[ij].setProperty<float>("delivery_tt", 45.0f/72.0f);
		    } else {
                        runs[ij].setProperty<float>("O2", 58.0f/72.0f);
                        runs[ij].setProperty<float>("delivery_tt", 58.0f/72.0f);
		    }
		    for (int c = 0; c < 23; c++) {
			if (c < 1){
				runs[ij].setProperty<float>("clones_dummy", c, 0.166666667f);
			} else if (c < 6) {
                                runs[ij].setProperty<float>("clones_dummy", c, (c+1)*0.166666667f);
			} else {
                                runs[ij].setProperty<float>("clones_dummy", c, 1.0f);
			}
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
        cuda_ensemble.Config().out_directory = "TT_WT_1to1000_results";
        cuda_ensemble.Config().out_format = "json";
        cuda_ensemble.setStepLog(step_log_cfg);
        cuda_ensemble.simulate(runs);
    }

    return 0;
}
