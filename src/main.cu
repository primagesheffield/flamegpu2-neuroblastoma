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
         * Initialisation
         */
        if (cuda_model.getSimulationConfig().input_file.empty()) {
            // Not sure if a default init is necessary yet
        }
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
        flamegpu::RunPlanVector runs(model, 100);
        {
            runs.setSteps(100);
            runs.setRandomSimulationSeed(12, 1);
        }
        /**
         * Create a logging config
         */
        flamegpu::StepLoggingConfig step_log_cfg(model);
        {
            step_log_cfg.setFrequency(1);
            step_log_cfg.logEnvironment("O2");
            step_log_cfg.logEnvironment("Cs_02");
            step_log_cfg.logEnvironment("C50_necro");
            step_log_cfg.logEnvironment("telo_critical");
            step_log_cfg.logEnvironment("P_DNA_damageHypo");
            step_log_cfg.logEnvironment("P_DNA_damagerp");
            step_log_cfg.logEnvironment("step_size");
            step_log_cfg.logEnvironment("glycoEff");
            step_log_cfg.logEnvironment("apop_critical");
            step_log_cfg.agent("Neuroblastoma").logCount();
            step_log_cfg.agent("Neuroblastoma").logMean<int>("apop");
            step_log_cfg.agent("Neuroblastoma").logMean<int>("necro");
            step_log_cfg.agent("Neuroblastoma").logMean<int>("necro_signal");
            step_log_cfg.agent("Neuroblastoma").logMean<int>("apop_signal");
            step_log_cfg.agent("Neuroblastoma").logMean<unsigned int>("cycle");
            step_log_cfg.agent("Schwann").logCount();
            step_log_cfg.agent("Schwann").logMean<int>("apop");
            step_log_cfg.agent("Schwann").logMean<int>("necro");
            step_log_cfg.agent("Schwann").logMean<int>("necro_signal");
            step_log_cfg.agent("Schwann").logMean<int>("apop_signal");
            step_log_cfg.agent("Schwann").logMean<unsigned int>("cycle");
            step_log_cfg.agent("Schwann").logMean<int>("hypoxia");
            step_log_cfg.agent("Schwann").logMean<int>("DNA_damage");
            step_log_cfg.agent("Schwann").logMean<int>("ATP");
            step_log_cfg.agent("Schwann").logMean<int>("nutrient");
            step_log_cfg.agent("Schwann").logMean<int>("telo_count");
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
