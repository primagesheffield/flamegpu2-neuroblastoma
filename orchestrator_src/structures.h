#ifndef ORCHESTRATOR_SRC_STRUCTURES_H_
#define ORCHESTRATOR_SRC_STRUCTURES_H_

#include <array>
#include <vector>

struct Version {
    int number[3] = {13, 8, 0};
    bool warn_flag = false;
};
/**
 * Properties in the file input to the orchestrator interface to FGPUNB
 */
struct OrchestratorInput {
    unsigned int seed;
    unsigned int steps;
    // Environment block
    int TERT_rarngm;
    int ATRX_inact;
    float V_tumour;
    float O2;
    std::array<float, 6> cellularity;
    int orchestrator_time;
    int MYCN_amp;
    int ALT;
    int ALK;
    int gradiff;
    int histology_init;
    float nb_telomere_length_mean;
    float nb_telomere_length_sd;
    float sc_telomere_length_mean;
    float sc_telomere_length_sd;
    float extent_of_differentiation_mean;
    float extent_of_differentiation_sd;
    float nb_necro_signal_mean;
    float nb_necro_signal_sd;
    float nb_apop_signal_mean;
    float nb_apop_signal_sd;
    float sc_necro_signal_mean;
    float sc_necro_signal_sd;
    float sc_apop_signal_mean;
    float sc_apop_signal_sd;
    std::vector<float> drug_effects;  // 6x length of start_effects and end_effects
    std::vector<int> start_effects;
    std::vector<int> end_effects;
};
/**
 * Properties in the file output by the orchestrator interface to FGPUNB
 */
struct OrchestratorOutput {
    float delta_O2;
    float O2;
    float delta_ecm;
    float ecm;
    float material_properties = 0;  // Unused
    float diffusion_coefficient = 0;  // Unused
    float total_volume_ratio_updated = 0;  // Unused
    std::array<float, 6> cellularity;
    float tumour_volume;
    float ratio_VEGF_NB_SC;
    float nb_telomere_length_mean;
    float nb_telomere_length_sd;
    float sc_telomere_length_mean;
    float sc_telomere_length_sd;
    float nb_necro_signal_mean;
    float nb_necro_signal_sd;
    float nb_apop_signal_mean;
    float nb_apop_signal_sd;
    float sc_necro_signal_mean;
    float sc_necro_signal_sd;
    float sc_apop_signal_mean;
    float sc_apop_signal_sd;
    float extent_of_differentiation_mean;
    float extent_of_differentiation_sd;
};

#endif  // ORCHESTRATOR_SRC_STRUCTURES_H_
