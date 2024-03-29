#ifndef ORCHESTRATOR_SRC_STRUCTURES_H_
#define ORCHESTRATOR_SRC_STRUCTURES_H_

#include <array>
#include <vector>
#include <string>

struct Version {
    /**
     * This function returns the model's semantic version [major, minor, patch].
     * This manually updated constant allows JSON exports to show the model version.
     */
    int number[3] = {13, 9, 4};
    /**
     * Previous versions of the model used this flag to detect mismatches between input file version and model version
     * This is not currently implemented
     */
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
    unsigned int cell_count;
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
    std::string calibration_file;
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
    unsigned int cell_count;
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
/**
 * fitting parameters which can be set via a secondary input file
 * Refer to environment.cu for their detail and/or default values
 */
struct CalibrationInput {
    float MYCN_fn11;
    float MAPK_RAS_fn11;
    float MAPK_RAS_fn01;
    float p53_fn;
    float p73_fn;
    float HIF_fn;
    float P_cycle_nb;
    float P_cycle_sc;
    float P_DNA_damageHypo;
    float P_DNA_damagerp;
    float P_unrepDNAHypo;
    float P_unrepDNArp;
    float P_necroIS;
    float P_telorp;
    float P_apopChemo;
    float P_DNA_damage_pathways;
    float P_apoprp;
    float P_necrorp;
    float scpro_jux;
    float nbdiff_jux;
    float nbdiff_amount;
    float nbapop_jux;
    float mig_sc;
};

#endif  // ORCHESTRATOR_SRC_STRUCTURES_H_
