#include "header.h"

extern std::array<double, 100> P_apopChemo_data;
extern std::array<double, 100> P_DNA_damage_pathways_data;
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
        flamegpu::RunPlanVector runs_control(model, 100);
        runs_control.setRandomPropertySeed(34523);  // Ensure that repeated runs use the same Random values to init ALK
        {
            runs_control.setSteps(336);
            runs_control.setRandomSimulationSeed(12, 1);
            // specialised: output subdir
            for (int i = 0; i < 10; ++i) {              
                for (int j = 0; j < 10; ++j) {
                  const int id = i * 10 + j;
                  runs_control[id].setProperty<float>("cellularity", (i+1)/11.0f);
                  runs_control[id].setProperty<float>("O2", j/9.0f);
                }
            }
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
        for (int i = 0; i < 100; ++i) {
            flamegpu::RunPlanVector runs_t = runs_control;
            // Dynamically generate a name for sub directory
            char subdir[80];
            sprintf(subdir, "lhc_%d", i+1);
            runs_t.setOutputSubdirectory(subdir);
            // Fill in specialised parameters
            runs_t.setProperty<float>("P_apopChemo", static_cast<float>(P_apopChemo_data[i]));
            runs_t.setProperty<float>("P_DNA_damage_pathways", static_cast<float>(P_DNA_damage_pathways_data[i]));                                
            // Append to the main run plan vector
            runs += runs_t;
          
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
        cuda_ensemble.Config().devices = { 0, 1, 2, 3 };
        cuda_ensemble.Config().out_directory = "sense_calibration_B29FF3BE_v3";
        cuda_ensemble.Config().out_format = "json";
        cuda_ensemble.setStepLog(step_log_cfg);
        cuda_ensemble.simulate(runs);
    }


    return 0;
}

std::array<double, 100> P_apopChemo_data = {
0.263683829, 0.365846659, 0.17511055, 0.776373969, 0.105233647, 0.507810489, 0.946478213, 0.488070018, 0.057081886, 0.815723774, 0.65112591, 0.277916755, 0.756349494, 0.149481759, 0.721965432, 0.700068351, 0.922793117, 0.711995192, 0.003508979, 0.354717301, 0.062038242, 0.914729146, 0.459143313, 0.076859575, 0.956512259, 0.842371762, 0.864247687, 0.797352178, 0.497671381, 0.01480133, 0.30443597, 0.73908521, 0.115593511, 0.762996402, 0.566349368, 0.097024464, 0.969317188, 0.021541191, 0.605542861, 0.581107898, 0.512380216, 0.311709151, 0.532948106, 0.12678033, 0.192839579, 0.421619287, 0.575517005, 0.228357574, 0.384638341, 0.34478956, 0.202088163, 0.63120265, 0.806421159, 0.322365531, 0.162116499, 0.216459446, 0.856919048, 0.159749117, 0.333036659, 0.130480455, 0.990490189, 0.619279992, 0.397439864, 0.087827861, 0.821713582, 0.235745431, 0.554501385, 0.523539142, 0.247548671, 0.439245201, 0.446964524, 0.977390328, 0.291541353, 0.031018324, 0.257698004, 0.590948457, 0.877389729, 0.284007067, 0.675570819, 0.984393829, 0.475982215, 0.628307207, 0.641170036, 0.690981116, 0.667588897, 0.469996997, 0.838727588, 0.183566258, 0.884952558, 0.744836133, 0.900858173, 0.787359336, 0.401131032, 0.890296329, 0.418984582, 0.68707485, 0.545630126, 0.040693785, 0.370366826, 0.933432582
};
std::array<double, 100> P_DNA_damage_pathways_data = {
0.847615078, 0.755081161, 0.804289303, 0.607797079, 0.223975427, 0.364125108, 0.432240383, 0.448191226, 0.822147846, 0.244610851, 0.614693551, 0.863162288, 0.810152531, 0.732493112, 0.189556843, 0.707293194, 0.29378822, 0.398390304, 0.174980208, 0.009895193, 0.646097844, 0.269507771, 0.023268255, 0.791490663, 0.508720496, 0.406241475, 0.070919038, 0.838800195, 0.149394559, 0.287886526, 0.635251854, 0.424276706, 0.929254013, 0.203279255, 0.474873762, 0.621270777, 0.665553055, 0.675249809, 0.157764035, 0.581831515, 0.030631252, 0.948035485, 0.686832637, 0.950025842, 0.114103169, 0.136832769, 0.271411132, 0.568952336, 0.783282829, 0.105295151, 0.382077681, 0.169597721, 0.352369266, 0.996367526, 0.198410934, 0.593482915, 0.98036631, 0.961403141, 0.729399281, 0.765350905, 0.699738228, 0.547444244, 0.93257774, 0.452526789, 0.216425804, 0.885428668, 0.911373861, 0.305041324, 0.529000701, 0.124465626, 0.019120979, 0.55332826, 0.48382508, 0.049681446, 0.085228061, 0.239950851, 0.373559324, 0.740372317, 0.650604054, 0.773671751, 0.874301665, 0.095599942, 0.32934115, 0.573408154, 0.068589637, 0.319104727, 0.901087439, 0.416018513, 0.519087743, 0.896090412, 0.257658635, 0.337477489, 0.859519955, 0.341403914, 0.492763544, 0.977653185, 0.461214119, 0.059283317, 0.710411226, 0.536970138
};