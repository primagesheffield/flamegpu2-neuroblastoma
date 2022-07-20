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
        flamegpu::RunPlanVector runs(model, 10);
        {
            runs.setSteps(336);
            runs.setRandomSimulationSeed(12, 1);
	    runs.setRandomPropertySeed(34523);
	    runs.setProperty<int>("orchestrator_time", 2);
	    runs.setProperty<int>("histology_init", 1);
	    runs.setProperty<int>("gradiff", 2);
	    runs.setProperty<float>("V_tumour", 1000000000*12.7281f);
	    std::array<float, 6> cellularity = {0.050349f, 3.1e-05f, 1.3e-05f, 0.178432f, 6.5e-05f, 3.7e-05f};
	    runs.setProperty<float,6>("cellularity", cellularity);
            runs.setProperty<float>("O2", 0.5f);
	    runs.setPropertyUniformRandom<float>("O2", 0.0f, 1.0f);
	    runs.setProperty<int>("MYCN_amp", 0);
	    runs.setProperty<int>("TERT_rarngm", 0);
	    runs.setProperty<int>("ATRX_inact", 1);
	    runs.setProperty<int>("ALT", 1);
	    runs.setProperty<int>("ALK", 0);
            std::array<unsigned int, 200> chemo_start = { 0 };
            std::array<unsigned int, 200> chemo_end = { 0 };
            std::array<float, 1200> chemo_effects = { 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f };
            runs.setProperty<unsigned int, 200>("chemo_start", chemo_start);
            runs.setProperty<unsigned int, 200>("chemo_end", chemo_end);
            runs.setProperty<float, 1200>("chemo_effects", chemo_effects);
	    runs.setProperty<float>("extent_of_differentiation_mean", 0.5021078944206241f);
	    runs.setProperty<float>("extent_of_differentiation_sd", 0.06140489950776104f);
	    runs.setProperty<float>("nb_telomere_length_mean", 59.99548683166502f);
            runs.setProperty<float>("nb_telomere_length_sd", 0.06703593879938122f);
	    runs.setProperty<float>("sc_telomere_length_mean", 37.955759048461914f);
	    runs.setProperty<float>("sc_telomere_length_sd", 3.2330457925796496f);
            runs.setProperty<float>("nb_apop_signal_mean", 0.14850002676248558f);
            runs.setProperty<float>("nb_apop_signal_sd", 0.49572743475437164f);
            runs.setProperty<float>("sc_apop_signal_mean", 0.15056806057691574f);
            runs.setProperty<float>("sc_apop_signal_sd", 0.5006335914134976f);
            runs.setProperty<float>("nb_necro_signal_mean", 0.04005729965865612f);
            runs.setProperty<float>("nb_necro_signal_sd", 0.21160535365343086f);
            runs.setProperty<float>("sc_necro_signal_mean", 0.011475176643580203f);
            runs.setProperty<float>("sc_necro_signal_sd", 0.11930776908993729f);
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
            step_log_cfg.logEnvironment("extent_of_differentiation_mean");
            step_log_cfg.logEnvironment("nb_telomere_length_mean");
            step_log_cfg.logEnvironment("nb_apop_signal_mean");
            step_log_cfg.logEnvironment("nb_necro_signal_mean");
            step_log_cfg.logEnvironment("extent_of_differentiation_sd");
            step_log_cfg.logEnvironment("nb_telomere_length_sd");
            step_log_cfg.logEnvironment("nb_apop_signal_sd");
            step_log_cfg.logEnvironment("nb_necro_signal_sd");
            step_log_cfg.logEnvironment("sc_telomere_length_mean");
            step_log_cfg.logEnvironment("sc_apop_signal_mean");
            step_log_cfg.logEnvironment("sc_necro_signal_mean");
            step_log_cfg.logEnvironment("sc_telomere_length_sd");
            step_log_cfg.logEnvironment("sc_apop_signal_sd");
            step_log_cfg.logEnvironment("sc_necro_signal_sd");
            step_log_cfg.agent("GridCell").logSum<int>("has_cells");
            step_log_cfg.agent("GridCell").logSum<int>("has_living_cells");
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
