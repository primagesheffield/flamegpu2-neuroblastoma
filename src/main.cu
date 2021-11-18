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
        flamegpu::RunPlanVector runs(model, 256);
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
            step_log_cfg.logEnvironment("Cs_O2");
            step_log_cfg.logEnvironment("C50_necro");
            step_log_cfg.logEnvironment("telo_critical");
            step_log_cfg.logEnvironment("P_DNA_damageHypo");
            step_log_cfg.logEnvironment("P_DNA_damagerp");
            step_log_cfg.logEnvironment("step_size");
            step_log_cfg.logEnvironment("glycoEff");
            step_log_cfg.logEnvironment("apop_critical");
            step_log_cfg.logEnvironment("nbapop_jux");
            step_log_cfg.logEnvironment("nbapop_para");
            step_log_cfg.logEnvironment("Nscl_count");
            step_log_cfg.logEnvironment("Nnbl_count");
            step_log_cfg.logEnvironment("MYCN_amp");
            step_log_cfg.logEnvironment("TERT_rarngm");
            step_log_cfg.logEnvironment("ATRX_inact");
            step_log_cfg.logEnvironment("ALT");
            step_log_cfg.logEnvironment("ALK");
            step_log_cfg.logEnvironment("telo_count");
            step_log_cfg.logEnvironment("P_necrorp");
            step_log_cfg.logEnvironment("P_necroIS");
            step_log_cfg.logEnvironment("bb_min_x");
            step_log_cfg.logEnvironment("bb_min_y");
            step_log_cfg.logEnvironment("bb_min_z");
            step_log_cfg.logEnvironment("bb_max_x");
            step_log_cfg.logEnvironment("bb_max_y");
            step_log_cfg.logEnvironment("bb_max_z");
            step_log_cfg.agent("Neuroblastoma").logCount();
            step_log_cfg.agent("Neuroblastoma").logMean<int>("apop");
            step_log_cfg.agent("Neuroblastoma").logMean<int>("necro");
            step_log_cfg.agent("Neuroblastoma").logMean<int>("necro_signal");
            step_log_cfg.agent("Neuroblastoma").logMean<int>("apop_signal");
            step_log_cfg.agent("Neuroblastoma").logMean<unsigned int>("cycle");
            step_log_cfg.agent("Neuroblastoma").logMean<int>("CAS");
            step_log_cfg.agent("Neuroblastoma").logMean<float>("CAS_fn");
            step_log_cfg.agent("Neuroblastoma").logMean<int>("BAK_BAX");
            step_log_cfg.agent("Neuroblastoma").logMean<int>("hypoxia");
            step_log_cfg.agent("Neuroblastoma").logMean<int>("ATP");
            step_log_cfg.agent("Neuroblastoma").logMean<float>("BAK_BAX_fn");
            step_log_cfg.agent("Neuroblastoma").logMean<int>("p53");
            step_log_cfg.agent("Neuroblastoma").logMean<int>("p73");
            step_log_cfg.agent("Neuroblastoma").logMean<int>("Bcl2_Bclxl");
            step_log_cfg.agent("Neuroblastoma").logMean<int>("IAP2");
            step_log_cfg.agent("Neuroblastoma").logMean<int>("MYCN");
            step_log_cfg.agent("Neuroblastoma").logMean<float>("IAP2_fn");
            step_log_cfg.agent("Neuroblastoma").logMean<int>("MYCN_amp");
            step_log_cfg.agent("Neuroblastoma").logMean<int>("ALK");
            step_log_cfg.agent("Neuroblastoma").logMean<float>("MYCN_fn");
            step_log_cfg.agent("Neuroblastoma").logMean<int>("TERT_rarngm");
            step_log_cfg.agent("Neuroblastoma").logMean<int>("ATRX_inact");
            step_log_cfg.agent("Neuroblastoma").logMean<int>("BNIP3");
            step_log_cfg.agent("Neuroblastoma").logMean<float>("BNIP3_fn");
            step_log_cfg.agent("Neuroblastoma").logMean<int>("HIF");
            step_log_cfg.agent("Neuroblastoma").logMean<float>("HIF_fn");

            step_log_cfg.agent("Neuroblastoma").logMean<float>("p53_fn");
            step_log_cfg.agent("Neuroblastoma").logMean<float>("p73_fn");
            step_log_cfg.agent("Neuroblastoma").logMean<float>("Bcl2_Bclxl_fn");
            step_log_cfg.agent("Neuroblastoma").logMean<int>("CHK1");
            step_log_cfg.agent("Neuroblastoma").logMean<int>("DNA_damage");
            step_log_cfg.agent("Neuroblastoma").logMean<int>("telo_count");
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
            step_log_cfg.agent("Schwann").logMean<int>("dummy_Nn");
            step_log_cfg.agent("GridCell").logMin<unsigned int>("Nnbn_grid");
            step_log_cfg.agent("GridCell").logMax<unsigned int>("Nnbn_grid");
            step_log_cfg.agent("GridCell").logMean<unsigned int>("Nnbn_grid");
            step_log_cfg.agent("GridCell").logMin<unsigned int>("Nscn_grid");
            step_log_cfg.agent("GridCell").logMax<unsigned int>("Nscn_grid");
            step_log_cfg.agent("GridCell").logMean<unsigned int>("Nscn_grid");
            step_log_cfg.logEnvironment("grid_origin");
            step_log_cfg.logEnvironment("grid_dims");
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
