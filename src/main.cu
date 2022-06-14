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
         * Create a run plan
         */
        flamegpu::RunPlanVector runs_control(model, 100);
        {
            runs_control.setOutputSubdirectory("control");
            runs_control.setSteps(336);
            runs_control.setRandomSimulationSeed(12, 1);
	    runs_control.setRandomPropertySeed(34523);
	    runs_control.setProperty<int>("histology_init", 0);
	    runs_control.setProperty<int>("gradiff", 1);
	    runs_control.setPropertyUniformRandom<float>("cellularity", 0.0f, 1.0f);
	    runs_control.setPropertyUniformRandom<float>("O2", 0.0f, 1.0f);
	    runs_control.setProperty<int>("MYCN_amp", 1);
	    runs_control.setProperty<int>("TERT_rarngm", 0);
	    runs_control.setProperty<int>("ATRX_inact", 0);
	    runs_control.setProperty<int>("ALT", 0);
	    runs_control.setPropertyUniformRandom<int>("ALK", 0, 2);
            std::array<unsigned int, 336> chemo_start = { 0 };
            std::array<unsigned int, 336> chemo_end = { 336 };
            std::array<float, 6> chemo_effects = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
            runs_control.setProperty<unsigned int, 336>("chemo_start", chemo_start);
            runs_control.setProperty<unsigned int, 336>("chemo_end", chemo_end);
            runs_control.setProperty<float, 6>("chemo_effects", chemo_effects);
        }
        flamegpu::RunPlanVector runs = runs_control;
        {  // 1
            flamegpu::RunPlanVector runs_1 = runs_control;
            runs_1.setOutputSubdirectory("group_1");
	    runs_1.setProperty<float>("V_tumour", 0.0f, 2*powf(2000.0f, 3));
            runs += runs_1;
        }
        {  // 2
            flamegpu::RunPlanVector runs_2 = runs_control;
            runs_2.setOutputSubdirectory("group_2");
            runs_2.setProperty<float>("V_tumour", 0.0f, 3*powf(2000.0f, 3));
            runs += runs_2;
        }
        {  // 3
            flamegpu::RunPlanVector runs_3 = runs_control;
            runs_3.setOutputSubdirectory("group_3");
            runs_3.setProperty<float>("V_tumour", 0.0f, 4*powf(2000.0f, 3));
            runs += runs_3;
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
