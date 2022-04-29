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
         * Create the template run plan
         */
        flamegpu::RunPlanVector runs_control(model, 10);
        runs_control.setRandomPropertySeed(34523);  // Ensure that repeated runs use the same Random values to init ALK
        {
            runs_control.setSteps(336);
            runs_control.setRandomSimulationSeed(12, 1);
	    runs_control.setPropertyUniformRandom<float>("cellularity", 0, 1);
	    runs_control.setPropertyUniformRandom<float>("O2", 0, 1);
            // specialised: P_apopChemo
            // specialised: P_DNA_damage_pathways
            runs_control.setProperty<int>("histology_init", 0);
            runs_control.setProperty<int>("gradiff", 1);
            // Mutations
            runs_control.setProperty<int>("MYCN_amp", 1);
            runs_control.setProperty<int>("TERT_rarngm", 0);
            runs_control.setProperty<int>("ATRX_inact", 0);
            runs_control.setProperty<int>("ALT", 0);
            runs_control.setPropertyUniformRandom<int>("ALK", 0, 2);
            // Chemo
            std::array<unsigned int, 336> chemo_start = { 0 };
            std::array<unsigned int, 336> chemo_end = { 336 };
            std::array<float, 6> chemo_effects = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
            runs_control.setProperty<unsigned int, 336>("chemo_start", chemo_start);
            runs_control.setProperty<unsigned int, 336>("chemo_end", chemo_end);
            runs_control.setProperty<float, 6>("chemo_effects", chemo_effects);
        }
        // Create an empty run plan vector to build all the specialised copies into
        flamegpu::RunPlanVector runs(model, 0);
        flamegpu::RunPlanVector runs_t = runs_control;
	runs_t.setOutputSubdirectory("0");
	std::array<float, 6> chemo_effects1 = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
	runs_t.setProperty<float, 6>("chemo_effects", chemo_effects1);
        runs += runs_t;
        runs_t.setOutputSubdirectory("0.3");
        std::array<float, 6> chemo_effects2 = {0.3f, 0.3f, 0.3f, 0.3f, 0.3f, 0.3f };
        runs_t.setProperty<float, 6>("chemo_effects", chemo_effects2);
        runs += runs_t;
        runs_t.setOutputSubdirectory("0.6");
        std::array<float, 6> chemo_effects3 = {0.6f, 0.6f, 0.6f, 0.6f, 0.6f, 0.6f };
        runs_t.setProperty<float, 6>("chemo_effects", chemo_effects3);
        runs += runs_t;
        runs_t.setOutputSubdirectory("0.9");
        std::array<float, 6> chemo_effects4 = {0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f };
        runs_t.setProperty<float, 6>("chemo_effects", chemo_effects4);
        runs += runs_t;
        runs_t.setOutputSubdirectory("1.0");
        std::array<float, 6> chemo_effects5 = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
        runs_t.setProperty<float, 6>("chemo_effects", chemo_effects5);
        runs += runs_t;
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
