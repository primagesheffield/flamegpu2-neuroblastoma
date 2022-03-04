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
        flamegpu::RunPlanVector runs_control(model, 10);
        {
            runs_control.setOutputSubdirectory("control");
            runs_control.setSteps(336);
            runs_control.setRandomSimulationSeed(12, 1);
            runs_control.setProperty<float>("O2", 0.39f);
            runs_control.setProperty<float>("cellularity", 0.25f);
            runs_control.setProperty<float>("theta_sc", 0.05f);
            runs_control.setPropertyUniformDistribution<float>("degdiff", 0.0f, 0.405f);
            std::array<unsigned int, 336> chemo_start = { 0 };
            std::array<unsigned int, 336> chemo_end = { 336 };
            std::array<float, 6> chemo_effects = { 0, 0, 0, 0, 0, 0 };
            runs_control.setProperty<unsigned int, 336>("chemo_start", chemo_start);
            runs_control.setProperty<unsigned int, 336>("chemo_end", chemo_end);
            runs_control.setProperty<float, 6>("chemo_effects", chemo_effects);
        }
        flamegpu::RunPlanVector runs = runs_control;
        {  // 1a
            flamegpu::RunPlanVector runs_1a = runs_control;
            runs_1a.setOutputSubdirectory("group_1a");
            std::array<float, 6> chemo_effects = { 0.3f, 0.3f, 0.3f, 0.3f, 0.3f, 0.3f };
            runs_1a.setProperty<float, 6>("chemo_effects", chemo_effects);
            runs += runs_1a;
        }
        {  // 1b
            flamegpu::RunPlanVector runs_1b = runs_control;
            runs_1b.setOutputSubdirectory("group_1b");
            std::array<float, 6> chemo_effects = { 0.6f, 0.6f, 0.6f, 0.6f, 0.6f, 0.6f };
            runs_1b.setProperty<float, 6>("chemo_effects", chemo_effects);
            runs += runs_1b;
        }
        {  // 1c
            flamegpu::RunPlanVector runs_1c = runs_control;
            runs_1c.setOutputSubdirectory("group_1c");
            std::array<float, 6> chemo_effects = { 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f };
            runs_1c.setProperty<float, 6>("chemo_effects", chemo_effects);
            runs += runs_1c;
        }
        {  // 2a
            flamegpu::RunPlanVector runs_2a = runs_control;
            runs_2a.setOutputSubdirectory("group_2a");
            std::array<unsigned int, 336> chemo_end = { 168 };
            std::array<float, 6> chemo_effects = { 0.3f, 0.3f, 0.3f, 0.3f, 0.3f, 0.3f };
            runs_2a.setProperty<unsigned int, 336>("chemo_end", chemo_end);
            runs_2a.setProperty<float, 6>("chemo_effects", chemo_effects);
            runs += runs_2a;
        }
        {  // 2b
            flamegpu::RunPlanVector runs_2b = runs_control;
            runs_2b.setOutputSubdirectory("group_2b");
            std::array<unsigned int, 336> chemo_end = { 168 };
            std::array<float, 6> chemo_effects = { 0.6f, 0.6f, 0.6f, 0.6f, 0.6f, 0.6f };
            runs_2b.setProperty<unsigned int, 336>("chemo_end", chemo_end);
            runs_2b.setProperty<float, 6>("chemo_effects", chemo_effects);
            runs += runs_2b;
        }
        {  // 2c
            flamegpu::RunPlanVector runs_2c = runs_control;
            runs_2c.setOutputSubdirectory("group_2c");
            std::array<unsigned int, 336> chemo_end = { 168 };
            std::array<float, 6> chemo_effects = { 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f };
            runs_2c.setProperty<unsigned int, 336>("chemo_end", chemo_end);
            runs_2c.setProperty<float, 6>("chemo_effects", chemo_effects);
            runs += runs_2c;
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
            step_log_cfg.agent("GridCell").logSum<int>("has_cells");
            step_log_cfg.agent("GridCell").logSum<int>("has_living_cells");
        }
        /**
         * Create Model Runner
         */
        flamegpu::CUDAEnsemble cuda_ensemble(model, argc, argv);
        cuda_ensemble.Config().concurrent_runs = 1;
        cuda_ensemble.Config().devices = { 0, 1, 2, 3 };
        cuda_ensemble.Config().out_directory = "sensitivity_runs_grid3";
        cuda_ensemble.Config().out_format = "json";
        cuda_ensemble.setStepLog(step_log_cfg);
        cuda_ensemble.simulate(runs);
    }


    return 0;
}
