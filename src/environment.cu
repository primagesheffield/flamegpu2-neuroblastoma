#include "header.h"

/**
 * Version 13.9+ is the PRIMAGE-facing version of the neuroblastoma model.
 * It supports precise action of chemotherapeutics on intracellular protein species.
 * This file contains all the parameters and initial conditions of the model,
 * either defined directly or derived.
 */

void data_layer_0(flamegpu::ModelDescription& model) {
    auto& env = model.Environment();
    /**
     * Integration with imaging biomarkers, part 1
     */
    // Initial volume of tumour (cubic micron).
    env.newProperty<float>("V_tumour", powf(2000.0f, 3));  // 2e6);
    // Factor by which the tumour can expand in all three directions.
    // According to Aherne and Buck (1971), the volume doubling time is around 60 days.
    env.newProperty<float>("boundary_max", 1.26f);
    // Displacement magnitude in the boundary
    env.newProperty<float, 3>("displace", { 2, 2, 2 });
    /**
     * Integration with imaging biomarkers, part 2
     */
    // Initial cellularity in the tumour (continuous, 0.05 to 0.95).
    // (living, apoptotic, necrotic neuroblasts and Schwann cells)
    // This is always set via input file
    env.newProperty<float, 6>("cellularity", {});
    // The initial total number of cells, from orchestrator.
    env.newProperty<int>("total_cell_init", 1000);
    // Histology type (0 is neuroblastoma, 1 is ganglioneuroblastoma, 2 is nodular ganglioneuroblastoma, 3 is intermixed ganglioneuroblastoma, 4 is ganglioneuroma, 5 is maturing ganglioneuroma, and 6 is mature ganglioneuroma).
    // If it is a ganglioneuroblastoma or a ganglioneuroma, assign the subtype stochastically.
    env.newProperty<int>("histology_init", 0);
    // DERIVED: Runtime decided based on histology_init
    env.newProperty<int>("histology", 0);
    // DERIVED: Grade of differentiation for neuroblastoma (0 is undifferentiated, 1 is pooly differentiated, and 2 is differentiating).
    env.newProperty<int>("gradiff", 0);
    // DERIVED: Fraction of Schwann cells in the cell population (continuous, 0 to 1).
    env.newProperty<float>("theta_sc", 0.5);
    // A parameter controlling the migration of Schwann cells into the tumour between successive loops.
    env.newProperty<float>("mig_sc", 1.0f);

    /**
     * Integration with imaging biomarkers, part 3
     */
     // DERIVED: Initial oxygen level (continuous, 0 to 1).
     // scaled by the level in the kidney, 72 mmHg (Carreau et al., 2011).
    env.newProperty<float>("O2", 0);
    // Time points at which chemo cycles begin.
    std::array<unsigned int, CHEMO_LEN> chemo_start = { 0, 240 };
    env.newProperty<unsigned int, CHEMO_LEN>("chemo_start", chemo_start);
    // Time points at which chemo cycles end.
    std::array<unsigned int, CHEMO_LEN> chemo_end = { 96, 336 };
    env.newProperty<unsigned int, CHEMO_LEN>("chemo_end", chemo_end);
    // Probabilities that CHK1, JAB1, HIF, MYCN, TEP1, and p53 are inhibited by chemotherapy.
    std::array<float, 6 * CHEMO_LEN> chemo_effects = { 0.602666f, 0.602666f, 0.602666f, 0.602666f, 0.602666f, 0.602666f };
    env.newProperty<float, 6 * CHEMO_LEN>("chemo_effects", chemo_effects);
    // Flag variables read by cells to detect chemo (see toggle_chemo())
    env.newProperty<int>("CHEMO_ACTIVE", 0);
    env.newProperty<int>("CHEMO_OFFSET", -1);
    // Number of cycles of chemotherapy.
    env.newProperty<unsigned int>("chemo_number", 2);
}
/**
 * integration with genetic/molecular biomarkers of neuroblasts
 */
void data_layer_1(flamegpu::ModelDescription& model) {
    auto& env = model.Environment();
    // MYCN amplification status (categorical, 0 or 1), default (-1) means unknown.
    env.newProperty<int>("MYCN_amp", 0);
    // TERT rearrangement status (categorical, 0 or 1), default (-1) means unknown.
    env.newProperty<int>("TERT_rarngm", 0);
    // ATRX inactivation status (categorical, 0 or 1), default (-1) means unknown.
    env.newProperty<int>("ATRX_inact", 0);
    // Alternative lengthening of telomeres status (categorical, 0 or 1), default (-1) means unknown.
    env.newProperty<int>("ALT", 0);
    // ALK amplification or activating mutation status (discrete, 0 or 1 or 2), default (-1) means unknown.
    // 0 means wild type, 1 means ALK amplification or activation, and 2 means other RAS mutations.
    env.newProperty<int>("ALK", 0);
}
/**
 * integration with genetic/molecular biomarkers of neuroblasts
 */
void data_layer_2(flamegpu::ModelDescription& model) {
    auto& env = model.Environment();
    // MYCN_fnxx: Functional activity of MYCN (continuous, 0 to 1), default (-1) means unknown.
    // This value depends on MYCN_amp and ALK.
    const float MYCN_fn11 = 0.331408f;
    const float MYCN_fn00 = 0.8f * 0.71f * MYCN_fn11;
    const float MYCN_fn10 = 0.71f * MYCN_fn11;
    const float MYCN_fn01 = 0.8f * MYCN_fn11;
    // MAPK_RAS_fnxx: Functional activity of MAPK/RAS signalling (continuous, 0 to 1), default (-1) means unknown.
    // This value depends on MYCN_amp and ALK.
    const float MAPK_RAS_fn11 = 0.985655f;
    const float MAPK_RAS_fn10 = 0.77f * MAPK_RAS_fn11;
    const float MAPK_RAS_fn01 = 0.869528f;
    const float MAPK_RAS_fn00 = 0.77f * MAPK_RAS_fn01;
    // Functional activity of p53 signalling (continuous, 0 to 1), default (-1) means unknown.
    const float p53_fn = 0.277201f;
    // Functional activity of p73 signalling (continuous, 0 to 1), default (-1) means unknown.
    const float p73_fn = 0.974863f;
    // Functional activity of HIF signalling (continuous, 0 to 1), default (-1) means unknown.
    const float HIF_fn = 0.606737f;

    env.newProperty<float>("MYCN_fn11", MYCN_fn11);
    env.newProperty<float>("MYCN_fn00", MYCN_fn00);
    env.newProperty<float>("MYCN_fn10", MYCN_fn10);
    env.newProperty<float>("MYCN_fn01", MYCN_fn01);
    env.newProperty<float>("MAPK_RAS_fn11", MAPK_RAS_fn11);
    env.newProperty<float>("MAPK_RAS_fn10", MAPK_RAS_fn10);
    env.newProperty<float>("MAPK_RAS_fn01", MAPK_RAS_fn01);
    env.newProperty<float>("MAPK_RAS_fn00", MAPK_RAS_fn00);
    env.newProperty<float>("p53_fn", p53_fn);
    env.newProperty<float>("p73_fn", p73_fn);
    env.newProperty<float>("HIF_fn", HIF_fn);
}
/**
 * integration with genetic/molecular biomarkers of neuroblasts
 * Activity levels of various species/pathways (continuous, 0 to 1), default (-1) means unknown.
 * Assumed to be one and not selected for calibration.
 */
void data_layer_3(flamegpu::ModelDescription& model) {
    auto& env = model.Environment();
    env.newProperty<float>("CHK1_fn", 1.0f);
    env.newProperty<float>("p21_fn", 1.0f);
    env.newProperty<float>("p27_fn", 1.0f);
    env.newProperty<float>("CDC25C_fn", 1.0f);
    env.newProperty<float>("CDS1_fn", 1.0f);
    env.newProperty<float>("ID2_fn", 1.0f);
    env.newProperty<float>("IAP2_fn", 1.0f);
    env.newProperty<float>("BNIP3_fn", 1.0f);
    env.newProperty<float>("JAB1_fn", 1.0f);
    env.newProperty<float>("Bcl2_Bclxl_fn", 1.0f);
    env.newProperty<float>("BAK_BAX_fn", 1.0f);
    env.newProperty<float>("CAS_fn", 1.0f);
    env.newProperty<float>("VEGF_fn", 1.0f);
}
/**
 * Physical parameters (neuroblasts and Schwann cells).
 */
void physical_parameters(flamegpu::ModelDescription& model) {
    auto& env = model.Environment();
    // Initial cell density of the cellular region in the tumour (cells per cubic micron).
    env.newProperty<float>("rho_tumour", static_cast<float>(9.39e-05));
    // Cell radius in microns at the beginning of the cell cycle.
    env.newProperty<float>("R_cell", 11 / 2.0f);
    // Half of a voxel's side length in microns.
    env.newProperty<float>("R_voxel", 15);
}
/**
 * Cell cycle parameters (neuroblasts and Schwann cells).
 */
void cell_cycle_parameters(flamegpu::ModelDescription& model) {
    auto& env = model.Environment();
    // Durations of G1, S, G2, and M in hours (Harper and Brooks, 2005).
    // Note, if these durations change, the static maths in calc_R() need updating too.
    std::array<unsigned int, 4> cycle_stages = { 12, 6, 4, 2 };
    // Perform a scan over the values, to calculate where the boundaries lie
    for (unsigned int i = 1; i < sizeof(cycle_stages) / sizeof(unsigned int); ++i) {
        cycle_stages[i] += cycle_stages[i - 1];
    }
    env.newProperty<unsigned int, 4>("cycle_stages", cycle_stages);
    // Efficiency of glycolysis compared to oxidative phosphorylation.
    env.newProperty<float>("glycoEff", 1 / 15.0f);  // du Plessis et al. (2015).
    // Basal probability of cycling for neuroblasts.
    env.newProperty<float>("P_cycle_nb", 0.0457f);  // Calibration #LHC_Cal6, index 9.
    // Basal probability of cycling for Schwann cells.
    env.newProperty<float>("P_cycle_sc", 0.0325f);  // Calibration #LHC_Cal6, index 9.
}
/**
 * Stress-Related parameters (neuroblasts and Schwann cells).
 */
void stress_related_parameters(flamegpu::ModelDescription& model) {
    auto& env = model.Environment();
    // Concentration (M) or partial pressure of oxygen (mmHg) at which 50 % of the tumour cell population die through necrosis.
    // However, this is used to calculate the probability that a living cell is hypoxic. This is time-independent, i.e at equilibrium.
    env.newProperty<float>("C50_necro", static_cast<float>(1.2 / 2.2779e-4 / 32));
    // Maximum number of telomere units in a cell.
    env.newProperty<int>("telo_maximum", 60);
    // Maximum number of telomere units in a senescent cell.
    env.newProperty<int>("telo_critical", 20);
    // Probability of gaining one unit of telomere in an hour, when telomerase or ALT is active.
    env.newProperty<float>("P_telorp", 0.08895382f);  // Calibration LHC#564
    // Probability of gaining DNA damage in an hour due to chemotherapy.
    env.newProperty<float>("P_apopChemo", 0.644f*0.5f);
    // Probability of gaining DNA damage in an hour due to hypoxia.
    env.newProperty<float>("P_DNA_damageHypo", 0.772947675f);  // Calibration LHC#564
    // Probability of repairing DNA damage in an hour.
    env.newProperty<float>("P_DNA_damagerp", 0.771497002f);  // Calibration LHC#564
    // Probability of gaining unreplicated DNA in an hour.
    env.newProperty<float>("P_unrepDNA", 0.0f);
    // Probability of gaining unreplicated DNA in an hour due to hypoxia.
    env.newProperty<float>("P_unrepDNAHypo", 0.434578817f);  // Calibration LHC#564
    // Probability of repairing unreplicated DNA in an hour.
    env.newProperty<float>("P_unrepDNArp", 0.890953082f);  // Calibration LHC#564
}
/**
 * Cell death parameters (neuroblasts and Schwann cells).
 */
void cell_death_parameters(flamegpu::ModelDescription& model) {
    auto& env = model.Environment();
    // Probability of DNA damages triggering CAS-independent pathways to induce an apoptotic signal in an hour.
    env.newProperty<float>("P_DNA_damage_pathways", 0.256f*0.5f);
    // Number of apoptotic signals needed to kill the cell.
    env.newProperty<int>("apop_critical", 3);
    // Probability of losing an apoptotic signal in an unstressed cell in an hour.
    env.newProperty<float>("P_apoprp", 0.03f);
    // Probability of secondary necrosis in an hour.
    env.newProperty<float>("P_2ndnecro", 0.2f);
    // Probability of the immune system triggering a necrotic signal in a living cell per necrotic cell present per hour.
    env.newProperty<float>("P_necroIS", 0.57675841f);  // Calibration LHC#564
    // Probability of losing a necrotic signal in an unstressed cell in an hour..
    env.newProperty<float>("P_necrorp", 0.98970852f);  // Calibration LHC#564
}
/**
 * NB-SC crosstalk parameters.
 */
void nb_sc_crosstalk_parameters(flamegpu::ModelDescription& model) {
    auto& env = model.Environment();
    // Scaling factor for the influence of neuroblasts on Schwann cell proliferation, juxtacrine.
    float scpro_jux = 0.0374f*0.1f;    // Calibration LHC_Cal3b#163, modified.
    // Scaling factor for the influence of Schwann cells on neuroblast differentiation, juxtacrine.
    float nbdiff_jux = 0.000521f;    // Calibration LHC_Cal6, index 9.
    // Amount of neuroblast differentiation achieved in an hour, triggered by Schwann cells.
    float nbdiff_amount = 0.01f;    // Assumed
    // Scaling factor for the influence of Schwann cells on neuroblast apoptosis, juxtacrine.
    float nbapop_jux = 0.0304f;    // Calibration LHC_Cal3b#163
    // Scaling factor for the influence of neuroblasts on Schwann cell proliferation, paracrine.
    float scpro_para = scpro_jux / 10;      // Assumed
    // Scaling factor for the influence of Schwann cells on neuroblast differentiation, paracrine.
    float nbdiff_para = nbdiff_jux / 10;    // Assumed
    // Scaling factor for the influence of Schwann cells on neuroblast apoptosis, paracrine.
    float nbapop_para = nbapop_jux / 10;    // Assumed

    env.newProperty<float>("scpro_jux", scpro_jux);
    env.newProperty<float>("nbdiff_jux", nbdiff_jux);
    env.newProperty<float>("nbdiff_amount", nbdiff_amount);
    env.newProperty<float>("nbapop_jux", nbapop_jux);
    env.newProperty<float>("scpro_para", scpro_para);
    env.newProperty<float>("nbdiff_para", nbdiff_para);
    env.newProperty<float>("nbapop_para", nbapop_para);
}
/**
 * Mechanical model parameters.
 */
void mechanical_model_parameters(flamegpu::ModelDescription& model) {
    auto& env = model.Environment();
    // Minimum overlap below which two cells cannot interact (microns).
    // Set to zero so that when two cells are just touching, they stop interacting immediately, i.e. no bouncing off.
    env.newProperty<float>("min_overlap", static_cast<float>(-4e-6 * 1e6 * 0));
    // Linear force law parameter in N m-1.
    env.newProperty<float>("k1", static_cast<float>(2.2e-3));  // Pathmanathan et al. (2009).
    // A cell's search distance in the search for neighbours (microns).
    env.newProperty<float>("R_neighbours", 3.15f*env.getProperty<float>("R_cell"));  // Calibrated by trial and error, Message radius must be > this
    // Number of other cells allowed within a cell's search distance before contact inhibition activates.
    env.newProperty<int>("N_neighbours", 2);  // Assumed
    // Factor by which a cell magnifies the force acting on it upon contact inhibition.
    env.newProperty<float>("k_locom", 2.0f);  // Assumed
    // Viscosity in N s m-1.
    env.newProperty<float>("mu", 0.4f);  // Pathmanathan et al. (2009).
    // Time step for the mechanical model in seconds.
    env.newProperty<float>("dt", 36);  // Pathmanathan et al. (2009).
}
/**
 * Microenvironment parameters.
 */
void microenvironment_parameters(flamegpu::ModelDescription& model) {
    auto& env = model.Environment();
    //  Production rate of oxygen given in moles per cell per hour.
    env.newProperty<float>("P_O20", static_cast<float>(-250e-12 * 60.0 / 80000.0));  // Negative because it is actually consumption (Grimes et al., 2014).
    // Concentration scale or maximum partial pressure of oxygen in moles per dm3 (M).
    env.newProperty<float>("Cs_O2", static_cast<float>(72.0 / 2.2779e-4 / 32.0));  // Oxygen level in the kidney (Carreau et al., 2011).
    // If this is on, the initial oxygen level is taken as the equilibrium and the vasculature will return the O2 level to this equilibrium at the end of every time step.
    env.newProperty<int>("staticO2", 0);
    // Number of angiogenic signals needed to update the vasculature.
    env.newProperty<int>("ang_critical", 100);  // In an experiment (Utzinger et al., 2015), it took 100 hours for two microvessel fragments to inosculate to the vascular network.
    // Production rate of matrix by one living Schwann cell (cubic microns per hour).
    // Protein production rate (Conlon et al., 2003), proportion of collagen in the output (DeClerck et al., 1987), and the volume of hydrated collagen I (Levick, 1987).
    env.newProperty<float>("P_matrix", static_cast<float>(3.125e-12 * 0.12 * 1.89e12));
    // Probability of an apoptotic or necrotic cell being engulfed by an immune cell in an hour.
    env.newProperty<float>("P_lysis", 0.35f);  // Jagiella et al. (2016).
}
/**
 *  Initial conditions (neuroblasts).
 */
void nb_initial_conditions(flamegpu::ModelDescription& model) {
    auto& env = model.Environment();
    // A flag (continuous, 0 to 4) indicating the cell's position in the cell cycle, default (-1) means random initialisation.
    env.newProperty<int>("cycle", -1);
    // A flag (Boolean variable) indicating if the cell is apoptotic, default (-1) means it is not apoptotic (0).
    env.newProperty<int>("apop", -1);
    // Number of apoptotic signals (categorical, 0, 1, 2, and 3), default (-1) means setting it to zero.
    env.newProperty<int>("apop_signal", -1);
    // A flag (Boolean variable) indicating if the cell is necrotic, default (-1) means it is not necrotic (zero).
    env.newProperty<int>("necro", -1);
    // Number of necrotic signals (categorical, integers from 0 to 168), default (-1) means setting it to zero.
    env.newProperty<int>("necro_signal", -1);  // Maximum is 168 (Warren et al., 2016).
    // DERIVED: Number of telomere units (categorical, integers from 0 to 60).
    env.newProperty<int>("telo_count", -1);
}
/**
 *  Initial conditions (Schwann cells).
 */
void sc_initial_conditions(flamegpu::ModelDescription& model) {
    auto& env = model.Environment();
    // A flag (continuous, 0 to 4) indicating the cell's position in the cell cycle, default (-1) means random initialisation.
    env.newProperty<int>("cycle_sc", -1);
    // A flag (Boolean variable) indicating if the cell is apoptotic, default (-1) means it is not apoptotic (zero).
    env.newProperty<int>("apop_sc", -1);
    // Number of apoptotic signals (categorical, 0, 1, 2, and 3), default (-1) means setting it to zero.
    env.newProperty<int>("apop_signal_sc", -1);
    // A flag (Boolean variable) indicating if the cell is necrotic, default (-1) means it is not necrotic (zero).
    env.newProperty<int>("necro_sc", -1);
    // Number of necrotic signals (categorical, integers from 0 to 168), default (-1) means setting it to zero.
    env.newProperty<int>("necro_signal_sc", -1);  // Maximum is 168 (Warren et al., 2016).
    // Number of telomere units (categorical, integers from 0 to 60).
    env.newProperty<int>("telo_count_sc", -1);  // -1 means random initialisation between 35 and 45, inclusive.
}
void internal_derived(flamegpu::ModelDescription& model) {
    auto& env = model.Environment();
    env.newProperty<unsigned int>("step_size", 1);
    // DERIVED: Half of the initial tumour length in one dimension in microns (derived).
    env.newProperty<float>("R_tumour", 0);
    // DERIVED: Tumour boundary in microns (derived).
    env.newProperty<float, 3>("bc_minus", { 0, 0, 0 });
    env.newProperty<float, 3>("bc_plus", { 0, 0, 0 });
    // DERIVED: Voxel volume in cubic microns (derived).
    env.newProperty<float>("V_grid", 0);
    // DERIVED: Voxel side area in microns squared (derived).
    env.newProperty<float>("A_grid", 0);
    // Add this before accessing O2grid
    // It's the location of the virtual/active grid origin within the full grid
    env.newProperty<unsigned int, 3>("grid_origin", { 0, 0, 0 });
    // Current size of the active/virtual O2grid
    env.newProperty<unsigned int, 3>("grid_dims", { 0, 0, 0 });
    // Span kind of acts as the radius(ignoring the center cell)
    env.newProperty<unsigned int, 3>("grid_span", { 0, 0, 0 });
    // If this is different to grid_dims, then different cells perform CAexpand operation to init their value inside alter()
    env.newProperty<unsigned int, 3>("grid_span_old", { 0, 0, 0 });
    // This variable acts as a switch to lock P_O2v to 0
    env.newProperty<int>("P_O2v_OFF", 0);
    // DERIVED: This variable represents vasculature()
    env.newProperty<float>("P_O2v", 0);
    // DERIVED
    env.newProperty<float>("matrix_dummy", 0);
    // Force resolution must do this many steps before exit
    env.newProperty<int>("min_force_resolution_steps", 2);
    // Flag to tell alter method to not process matrix/o2 on init pass
    env.newProperty<int>("SKIP_ALTER", 1);
    // This value is updated by vasculature(), but doesn't appear to be consumed (Py is just reporting it each step).
    env.newProperty<int>("ang_signal", 0);
    // These two values are consumed by Neuroblastoma_sense()
    env.newProperty<unsigned int>("Nnbl_count", 0);
    env.newProperty<unsigned int>("Nscl_count", 0);
    // VALIDATION:
    env.newProperty<unsigned int>("force_resolution_steps", 0);
    env.newMacroProperty<unsigned int>("validation_Nnbl");
    env.newMacroProperty<unsigned int>("validation_Nscl");
    env.newProperty<unsigned int>("validation_Nnbl", 0);
    env.newProperty<unsigned int>("validation_Nscl", 0);
    // Histograms
    env.newMacroProperty<unsigned int, 42>("histogram_nbl");
    env.newMacroProperty<unsigned int, 42>("histogram_nba");
    env.newMacroProperty<unsigned int, 42>("histogram_nbn");
    env.newMacroProperty<unsigned int, 42>("histogram_scl");
    env.newMacroProperty<unsigned int, 42>("histogram_sca");
    env.newMacroProperty<unsigned int, 42>("histogram_scn");
    env.newProperty<unsigned int, 42>("histogram_nbl", {});
    env.newProperty<unsigned int, 42>("histogram_nba", {});
    env.newProperty<unsigned int, 42>("histogram_nbn", {});
    env.newProperty<unsigned int, 42>("histogram_scl", {});
    env.newProperty<unsigned int, 42>("histogram_sca", {});
    env.newProperty<unsigned int, 42>("histogram_scn", {});
}
void data_layer_primage(flamegpu::ModelDescription& model) {
    auto& env = model.Environment();
    env.newProperty<int>("orchestrator_time", 0);
    env.newProperty<float>("nb_telomere_length_mean", 0);
    env.newProperty<float>("nb_telomere_length_sd", 0);
    env.newProperty<float>("sc_telomere_length_mean", 0);
    env.newProperty<float>("sc_telomere_length_sd", 0);
    env.newProperty<float>("extent_of_differentiation_mean", 0);
    env.newProperty<float>("extent_of_differentiation_sd", 0);
    env.newProperty<float>("nb_necro_signal_mean", 0);
    env.newProperty<float>("nb_necro_signal_sd", 0);
    env.newProperty<float>("nb_apop_signal_mean", 0);
    env.newProperty<float>("nb_apop_signal_sd", 0);
    env.newProperty<float>("sc_necro_signal_mean", 0);
    env.newProperty<float>("sc_necro_signal_sd", 0);
    env.newProperty<float>("sc_apop_signal_mean", 0);
    env.newProperty<float>("sc_apop_signal_sd", 0);
}
FLAMEGPU_INIT_FUNCTION(InitDerivedEnvironment) {
    /**
     * Data layer 0 (integration with imaging biomarkers).
     */
    const int histology_init = FLAMEGPU->environment.getProperty<int>("histology_init");
    int histology;
    const int gradiff = FLAMEGPU->environment.getProperty<int>("gradiff");
    float theta_sc = 0;
    // const float O2 = (2/72.0f) + (FLAMEGPU->random.uniform<float>() * (30 / 72.0f));  // rng in range [2/72, 32/72]
    if (histology_init == 1) {
        if (FLAMEGPU->random.uniform<float>() < 0.5f) {
            histology = 2;
        } else {
            histology = 3;
        }
    } else if (histology_init == 4) {
        if (FLAMEGPU->random.uniform<float>() < 0.5f) {
            histology = 5;
        } else {
            histology = 6;
        }
    } else {
        histology = histology_init;
    }
    if (histology == 0) {
        if (gradiff == 0) {
            theta_sc = 0.05f + (FLAMEGPU->random.uniform<float>() * 0.12f);  // [0.05, 0.17)
        } else if (gradiff == 1) {
            theta_sc = 0.17f + (FLAMEGPU->random.uniform<float>() * 0.16f);  // [0.17, 0.33)
        } else if (gradiff == 2) {
            theta_sc = 0.33f + (FLAMEGPU->random.uniform<float>() * 0.17f);  // [0.33, 0.5)
        }
    } else if (histology == 2) {
        float dummy = FLAMEGPU->random.uniform<float>();
        if (dummy < 0.33f) {
            if (gradiff == 0) {
                theta_sc =  0.05f + (FLAMEGPU->random.uniform<float>() * 0.12f);  // [0.05, 0.17)
            } else if (gradiff == 1) {
                theta_sc = 0.17f + (FLAMEGPU->random.uniform<float>() * 0.16f);  // [0.17, 0.33)
            } else if (gradiff == 2) {
                theta_sc = 0.33f + (FLAMEGPU->random.uniform<float>() * 0.17f);  // [0.33, 0.5)
            }
        } else if (dummy < 0.66f) {
            theta_sc = 0.5f + (FLAMEGPU->random.uniform<float>() * 0.17f);  // [0.5, 0.67)
        } else {
            theta_sc = 0.67f + (FLAMEGPU->random.uniform<float>() * 0.16f);  // [0.67, 0.83)
        }
    } else if (histology == 3) {
        theta_sc = 0.5f + (FLAMEGPU->random.uniform<float>() * 0.17f);  // [0.5, 0.67)
    } else if (histology == 5) {
        theta_sc = 0.67f + (FLAMEGPU->random.uniform<float>() * 0.16f);  // [0.67, 0.83)
    } else if (histology == 6) {
        theta_sc = 0.83f + (FLAMEGPU->random.uniform<float>() * 0.12f);  // [0.83, 0.95)
    }
    FLAMEGPU->environment.setProperty<int>("histology", histology);
    FLAMEGPU->environment.setProperty<int>("gradiff", gradiff);
    if (FLAMEGPU->environment.getProperty<int>("orchestrator_time") == 0) {
        // Must update cellularity if orchestrator_time == 0
        const float cellularity_total = FLAMEGPU->environment.getProperty<float>("cellularity", 0);
        FLAMEGPU->environment.setProperty<float>("cellularity", 0, cellularity_total * (1.0f - theta_sc));
        FLAMEGPU->environment.setProperty<float>("cellularity", 3, cellularity_total * theta_sc);
    }
    FLAMEGPU->environment.setProperty<float>("theta_sc", theta_sc);
    // FLAMEGPU->environment.setProperty<float>("O2", O2);
    /**
     * Data Layer 1 (integration with genetic/molecular biomarkers of neuroblasts).
     */
    // FLAMEGPU->environment.setProperty<int>("MYCN_amp", FLAMEGPU->random.uniform<int>(0, 1));
    // FLAMEGPU->environment.setProperty<int>("TERT_rarngm", FLAMEGPU->random.uniform<int>(0, 1));
    // FLAMEGPU->environment.setProperty<int>("ATRX_inact", FLAMEGPU->random.uniform<int>(0, 1));
    // FLAMEGPU->environment.setProperty<int>("ALT", FLAMEGPU->random.uniform<int>(0, 1));
    // FLAMEGPU->environment.setProperty<int>("ALK", FLAMEGPU->random.uniform<int>(0, 2));
    /**
     * Data Layer 2 (integration with genetic/molecular biomarkers of neuroblasts).
     */
    // n/a
    /**
     * Data Layer 3 (integration with genetic/molecular biomarkers of neuroblasts).
     * Activity levels of various species/pathways (continuous, 0 to 1), default (-1) means unknown.
     */
    // n/a
    /**
     * Cell cycle parameters (neuroblasts and Schwann cells).
     */
    // n/a
    /**
     * Cell death parameters (neuroblasts and Schwann cells).
     */
    // n/a
    /**
     * Schwann cell parameters.
     */
    // n/a
    /**
     * NB-SC crosstalk parameters.
     */
    // n/a
    /**
     * immune system parameters.
     */
    // n/a
    /**
     * Extracellular environment parameters.
     */
    // n/a
    /**
     * Initial conditions (neuroblasts).
     */
    const int ALT = FLAMEGPU->environment.getProperty<int>("ALT");
    const int ATRX_inact = FLAMEGPU->environment.getProperty<int>("ATRX_inact");
    const int MYCN_amp = FLAMEGPU->environment.getProperty<int>("MYCN_amp");
    const int TERT_rarngm = FLAMEGPU->environment.getProperty<int>("TERT_rarngm");
    int telo_count;
    if ((ALT == 1 || ATRX_inact == 1) && (MYCN_amp == 1 || TERT_rarngm == 1)) {
        telo_count = FLAMEGPU->random.uniform<int>(1, 2);
    } else if (ALT == 1 || ATRX_inact == 1) {
        telo_count = 1;
    } else if (MYCN_amp == 1 || TERT_rarngm == 1) {
        telo_count = 2;
    } else {
        telo_count = 3;
    }
    FLAMEGPU->environment.setProperty<int>("telo_count", telo_count);
    /**
     * Initial conditions (Schwann cells).
     */
    // n/a
    /**
     * Internal/Derived
     * These might not need to be init, but better safe than sorry
     */
    const float boundary_max = FLAMEGPU->environment.getProperty<float>("boundary_max");
    const float V_tumour = FLAMEGPU->environment.getProperty<float>("V_tumour");
    const float R_voxel = FLAMEGPU->environment.getProperty<float>("R_voxel");
    const float P_O2v = FLAMEGPU->environment.getProperty<float>("P_O2v");
    const int P_O2v_OFF = FLAMEGPU->environment.getProperty<int>("P_O2v_OFF");
     // Initial tumour radius (microns)
    const float R_tumour = static_cast<float>(0.5 * pow(V_tumour, (1.0 / 3.0)));
    FLAMEGPU->environment.setProperty<float>("R_tumour", R_tumour);
    // Tumour boundary's (microns)
    FLAMEGPU->environment.setProperty<glm::vec3>("bc_minus", glm::vec3(-boundary_max * R_tumour));
    FLAMEGPU->environment.setProperty<glm::vec3>("bc_plus", glm::vec3(boundary_max * R_tumour));
    // Grid element or voxel volume (cubic microns).
    FLAMEGPU->environment.setProperty<float>("V_grid", static_cast<float>(pow(2.0 * R_voxel, 3.0f)));
    // Grid element or voxel side area (square microns).
    FLAMEGPU->environment.setProperty<float>("A_grid", static_cast<float>(pow(2.0 * R_voxel, 2.0f)));
    // This variable represents vasculature()
    FLAMEGPU->environment.setProperty<float>("P_O2v", P_O2v_OFF ? 0.0f : P_O2v);
    std::array<float, 6> h_env_cellularity = FLAMEGPU->environment.getProperty<float, 6>("cellularity");
    float h_matrix_dummy = (1.0f - (h_env_cellularity[0] + h_env_cellularity[1] + h_env_cellularity[2] + h_env_cellularity[3] + h_env_cellularity[4] + h_env_cellularity[5]));
    FLAMEGPU->environment.setProperty<float>("matrix_dummy", h_matrix_dummy);
}
// #define _USE_MATH_DEFINES
// #include <math.h>
void defineEnvironment(flamegpu::ModelDescription& model) {
    data_layer_0(model);
    data_layer_1(model);
    data_layer_2(model);
    data_layer_3(model);
    physical_parameters(model);
    cell_cycle_parameters(model);
    stress_related_parameters(model);
    cell_death_parameters(model);
    nb_sc_crosstalk_parameters(model);
    mechanical_model_parameters(model);
    microenvironment_parameters(model);
    nb_initial_conditions(model);
    sc_initial_conditions(model);
    data_layer_primage(model);
    internal_derived(model);
    model.addInitFunction(InitDerivedEnvironment);
}
