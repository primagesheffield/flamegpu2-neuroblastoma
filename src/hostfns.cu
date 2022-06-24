#include <glm/gtx/component_wise.hpp>

#include "header.h"

__device__ float mean2_mean;

FLAMEGPU_CUSTOM_TRANSFORM(mean2_transform, a) {

    if (a != 0)

        return static_cast<double>((a - mean2_mean) * (a - mean2_mean));

    return 0;

}

FLAMEGPU_CUSTOM_REDUCTION(mean2_sum, a, b) {

    return a + b;

}

FLAMEGPU_CUSTOM_REDUCTION(glm_min, a, b) {
    return glm::min(a, b);
}
FLAMEGPU_CUSTOM_REDUCTION(glm_max, a, b) {
    return glm::max(a, b);
}
FLAMEGPU_HOST_FUNCTION(CAexpand) {
    // Calculate average matrix value incase the grid increases
    // Initially matrix_dummy is init correct, but grid_dims isn't, so don't calc until CAexpand() has been called once.
    static bool first_time = true;
    if (!first_time) {
        const glm::uvec3 grid_dims = FLAMEGPU->environment.getProperty<glm::uvec3>("grid_dims");
        const float matrix_dummy = FLAMEGPU->agent("GridCell").sum<float>("matrix_value") / compMul(grid_dims);
        FLAMEGPU->environment.setProperty<float>("matrix_dummy", matrix_dummy);
        first_time = false;
    }
    // Expand the grid to match tumour growth.
    // 1. Check if the tumour has expanded in each direction.
    // 2. If so, expand the matrix grid in that direction, using the average quantity in the existing grid.
    // 3. Expand the grids for various cell types.
    const glm::ivec3 oldspan = FLAMEGPU->environment.getProperty<glm::uvec3>("grid_span");
    FLAMEGPU->environment.setProperty<glm::uvec3>("grid_span_old", oldspan);

    glm::ivec3 newspan;
    {
        const glm::vec3 min_pos = min(
            FLAMEGPU->agent("Neuroblastoma").reduce<glm::vec3>("xyz", glm_min, glm::vec3(std::numeric_limits<float>().max())),
            FLAMEGPU->agent("Schwann").reduce<glm::vec3>("xyz", glm_min, glm::vec3(std::numeric_limits<float>().max())));
        const glm::vec3 max_pos = max(
            FLAMEGPU->agent("Neuroblastoma").reduce<glm::vec3>("xyz", glm_max, glm::vec3(-std::numeric_limits<float>().max())),
            FLAMEGPU->agent("Schwann").reduce<glm::vec3>("xyz", glm_max, glm::vec3(-std::numeric_limits<float>().max())));
        const float R_voxel = FLAMEGPU->environment.getProperty<float>("R_voxel");
        newspan = glm::uvec3(glm::ceil((glm::max(glm::abs(min_pos), glm::abs(max_pos)) / R_voxel / 2.0f) + 0.5f));
    }
    // clamp span (i don't think the python algorithm lets o2 grid shrink)
    const glm::uvec3 newspan_u = max(oldspan, newspan);
    FLAMEGPU->environment.setProperty<glm::uvec3>("grid_span", newspan_u);
    const glm::uvec3 new_grid_dims = (newspan_u * 2u) - glm::uvec3(1);
    FLAMEGPU->environment.setProperty<glm::uvec3>("grid_dims", new_grid_dims);
    if (new_grid_dims.x > GRID_MAX_DIMENSIONS || new_grid_dims.y > GRID_MAX_DIMENSIONS || new_grid_dims.z > GRID_MAX_DIMENSIONS) {
        fprintf(stderr, "grid has grown too large (%u, %u, %u), recompile with bigger!\n", new_grid_dims.x, new_grid_dims.y, new_grid_dims.z);
        const glm::vec3 min_pos = min(
            FLAMEGPU->agent("Neuroblastoma").reduce<glm::vec3>("xyz", glm_min, glm::vec3(std::numeric_limits<float>().max())),
            FLAMEGPU->agent("Schwann").reduce<glm::vec3>("xyz", glm_min, glm::vec3(std::numeric_limits<float>().max())));
        const glm::vec3 max_pos = max(
            FLAMEGPU->agent("Neuroblastoma").reduce<glm::vec3>("xyz", glm_max, glm::vec3(-std::numeric_limits<float>().max())),
            FLAMEGPU->agent("Schwann").reduce<glm::vec3>("xyz", glm_max, glm::vec3(-std::numeric_limits<float>().max())));
        fprintf(stderr, "step %u\n", FLAMEGPU->getStepCounter());
        fprintf(stderr, "Min Pos (%f, %f, %f), MaxPos (%f, %f, %f)!\n", min_pos.x, min_pos.y, min_pos.z, max_pos.x, max_pos.y, max_pos.z);
        throw std::runtime_error("grid has grown too large, recompile with bigger\n");
    }
    glm::uvec3 grid_origin = (glm::uvec3(GRID_MAX_DIMENSIONS) - new_grid_dims) / 2u;
    FLAMEGPU->environment.setProperty<glm::uvec3>("grid_origin", grid_origin);
}
FLAMEGPU_HOST_FUNCTION(alter2) {
    const unsigned int Nnbl_count = FLAMEGPU->agent("GridCell").sum<unsigned int>("Nnbl_grid");
    const unsigned int Nscl_count = FLAMEGPU->agent("GridCell").sum<unsigned int>("Nscl_grid");
    FLAMEGPU->environment.setProperty<unsigned int>("Nnbl_count", Nnbl_count);
    FLAMEGPU->environment.setProperty<unsigned int>("Nscl_count", Nscl_count);

    // Don't update O2/matrix if it's INIT pass
    if (FLAMEGPU->getStepCounter() == 0) {
        FLAMEGPU->environment.setProperty<int>("SKIP_ALTER", 0);
        return;
    }

    if (FLAMEGPU->environment.getProperty<int>("staticO2") == 0) {
        const glm::uvec3 grid_dims = FLAMEGPU->environment.getProperty<glm::uvec3>("grid_dims");
        const float N_voxel = static_cast<float>(compMul(grid_dims));
        float total_O2 = FLAMEGPU->environment.getProperty<float>("O2");
        // O2 += (np.sum(Nnbl_grid)+np.sum(Nscl_grid))*envn.P_O20*step_size/(N_voxel*envn.V_grid()*1e-15)/envn.Cs_O2
        const float P_O20 = FLAMEGPU->environment.getProperty<float>("P_O20");
        const unsigned int step_size = FLAMEGPU->environment.getProperty<unsigned int>("step_size");
        const float V_grid = FLAMEGPU->environment.getProperty<float>("V_grid");
        const float Cs_O2 = FLAMEGPU->environment.getProperty<float>("Cs_O2");
        const float P_O2v = FLAMEGPU->environment.getProperty<float>("P_O2v");
        total_O2 += FLAMEGPU->agent("GridCell").sum<unsigned int>("N_l_grid") * P_O20 * step_size / static_cast<float>(N_voxel * V_grid * 1e-15) / Cs_O2;
        total_O2 += P_O2v;
        total_O2 = glm::clamp<float>(total_O2, 0, 1);
        FLAMEGPU->environment.setProperty<float>("O2", total_O2);
    }
}
FLAMEGPU_HOST_FUNCTION(vasculature) {
    // Case 1:
    // Set up the initial vasculature in terms of the amount of oxygen it supplies in one time step.
    // The assumption is that it can supply the amount consumed by the initial population of living neuroblasts and Schwann cells in one time step.
    // Case 2 :
    // When there are more living VEGF - producing neuroblasts than living Schwann cells, an angiogenic signal is produced.
    // When there are enough angiogenic signals, calculate the amount of oxygen consumed by the current population of living cells.
    // Take this as the new oxygen supply rate if it exceeds the old rate.
    const unsigned int t = FLAMEGPU->getStepCounter();  // This should be step_count + 1, however step count is incremented early by calculate_convergence
    const float P_O20 = FLAMEGPU->environment.getProperty<float>("P_O20");
    const unsigned int step_size = FLAMEGPU->environment.getProperty<unsigned int>("step_size");
    const float V_grid = FLAMEGPU->environment.getProperty<float>("V_grid");
    const float Cs_O2 = FLAMEGPU->environment.getProperty<float>("Cs_O2");
    const glm::uvec3 grid_dims = FLAMEGPU->environment.getProperty<glm::uvec3>("grid_dims");
    const unsigned int N_voxel = glm::compMul(grid_dims);
    const int ang_critical = FLAMEGPU->environment.getProperty<int>("ang_critical");
    float P_O2v = FLAMEGPU->environment.getProperty<float>("P_O2v");
    int ang_signal = FLAMEGPU->environment.getProperty<int>("ang_signal");
    unsigned int N_vf = FLAMEGPU->agent("Neuroblastoma").count<int>("VEGF", 1);
    const int NB_COUNT = FLAMEGPU->agent("Neuroblastoma").count();
    const int SC_COUNT = FLAMEGPU->agent("Schwann").count();
    const unsigned int Nnbl_grid = t == 0 ? NB_COUNT : FLAMEGPU->agent("GridCell").sum<unsigned int>("Nnbl_grid");
    const unsigned int Nscl_grid = t == 0 ? SC_COUNT : FLAMEGPU->agent("GridCell").sum<unsigned int>("Nscl_grid");
    if (t == 0) {
        P_O2v = -1.0f * (NB_COUNT + SC_COUNT) * P_O20 * step_size / static_cast<float>(N_voxel * V_grid * 1e-15) / Cs_O2;
        ang_signal = 0;
    } else if (N_vf > Nscl_grid) {
        ang_signal += 1 * step_size;
    }
    if (ang_signal == ang_critical) {
        const float dummy_P_O2v = -1.0f * (Nnbl_grid + Nscl_grid) * P_O20 * step_size / static_cast<float>(N_voxel * V_grid * 1e-15) / Cs_O2;
        if (dummy_P_O2v > P_O2v) {
            P_O2v = dummy_P_O2v;
        }
        ang_signal = 0;
    }
    // Check whether P_O2v is disabled
    const int P_O2v_OFF = FLAMEGPU->environment.getProperty<int>("P_O2v_OFF");
    FLAMEGPU->environment.setProperty<float>("P_O2v", P_O2v_OFF ? 0 : P_O2v);
    FLAMEGPU->environment.setProperty<int>("ang_signal", ang_signal);
}
FLAMEGPU_HOST_FUNCTION(reset_grids) {
    // Reset all grid counters
    // This is mostly useful after cell_cycle, which further manips the grid counters
    FLAMEGPU->environment.getMacroProperty<unsigned int, GMD, GMD, GMD>("Nnbn_grid").zero();
    FLAMEGPU->environment.getMacroProperty<unsigned int, GMD, GMD, GMD>("Nscn_grid").zero();
    FLAMEGPU->environment.getMacroProperty<unsigned int, GMD, GMD, GMD>("Nnbl_grid").zero();
    FLAMEGPU->environment.getMacroProperty<unsigned int, GMD, GMD, GMD>("Nscl_grid").zero();
    FLAMEGPU->environment.getMacroProperty<unsigned int, GMD, GMD, GMD>("Nscl_col_grid").zero();
    // Histograms
    FLAMEGPU->environment.getMacroProperty<unsigned int, GMD, GMD, GMD>("Nnba_grid").zero();
    FLAMEGPU->environment.getMacroProperty<unsigned int, GMD, GMD, GMD>("Nsca_grid").zero();
    FLAMEGPU->environment.getMacroProperty<unsigned int, 42>("histogram_nbl").zero();
    FLAMEGPU->environment.getMacroProperty<unsigned int, 42>("histogram_nba").zero();
    FLAMEGPU->environment.getMacroProperty<unsigned int, 42>("histogram_nbn").zero();
    FLAMEGPU->environment.getMacroProperty<unsigned int, 42>("histogram_scl").zero();
    FLAMEGPU->environment.getMacroProperty<unsigned int, 42>("histogram_sca").zero();
    FLAMEGPU->environment.getMacroProperty<unsigned int, 42>("histogram_scn").zero();
}
FLAMEGPU_HOST_FUNCTION(host_validation) {
    auto GridCell = FLAMEGPU->agent("GridCell");
    auto Neuroblastoma = FLAMEGPU->agent("Neuroblastoma");
    auto Schwann = FLAMEGPU->agent("Schwann");
    const unsigned int validation_Nnbl = FLAMEGPU->environment.getMacroProperty<unsigned int>("validation_Nnbl");
    const unsigned int validation_Nscl = FLAMEGPU->environment.getMacroProperty<unsigned int>("validation_Nscl");
    FLAMEGPU->environment.setProperty<unsigned int>("validation_Nnbl", validation_Nnbl);
    FLAMEGPU->environment.setProperty<unsigned int>("validation_Nscl", validation_Nscl);
    FLAMEGPU->environment.getMacroProperty<unsigned int>("validation_Nnbl").zero();
    FLAMEGPU->environment.getMacroProperty<unsigned int>("validation_Nscl").zero();
    // Cellularity
    const unsigned int NB_living_count = validation_Nnbl;
    const unsigned int NB_apop_count = Neuroblastoma.sum<int>("apop");
    const unsigned int NB_necro_count = Neuroblastoma.sum<int>("necro");
    const unsigned int SC_living_count = validation_Nscl;
    const unsigned int SC_apop_count = Schwann.sum<int>("apop");
    const unsigned int SC_necro_count = Schwann.sum<int>("necro");
    const unsigned int TOTAL_CELL_COUNT = NB_living_count + NB_apop_count + NB_necro_count + SC_living_count + SC_apop_count + SC_necro_count;
    // Calculate each fraction (e.g. number of living SCs/number of all cells) and multiply it by (1-matrix).
    const float ecm = GridCell.sum<float>("matrix_value") / glm::compMul(FLAMEGPU->environment.getProperty<glm::uvec3>("grid_dims"));
    std::array<float, 6> cellularity = {};
    if (TOTAL_CELL_COUNT) {
        cellularity[0] = NB_living_count * (1.0f - ecm) / TOTAL_CELL_COUNT;
        cellularity[1] = NB_apop_count * (1.0f - ecm) / TOTAL_CELL_COUNT;
        cellularity[2] = NB_necro_count * (1.0f - ecm) / TOTAL_CELL_COUNT;
        cellularity[3] = SC_living_count * (1.0f - ecm) / TOTAL_CELL_COUNT;
        cellularity[4] = SC_apop_count * (1.0f - ecm) / TOTAL_CELL_COUNT;
        cellularity[5] = SC_necro_count * (1.0f - ecm) / TOTAL_CELL_COUNT;
    }
    FLAMEGPU->environment.setProperty<float, 6>("validation_cellularity", cellularity);
    // Tumour volume
    float tumour_volume;
    {
        const int total_cell_count = Neuroblastoma.count() + Schwann.count();
        if (total_cell_count) {
            const float rho_tumour = FLAMEGPU->environment.getProperty<float>("rho_tumour");
            const float matrix_dummy = FLAMEGPU->environment.getProperty<float>("matrix_dummy");
            tumour_volume = total_cell_count / rho_tumour / (1 - matrix_dummy);
        } else {
            tumour_volume = FLAMEGPU->environment.getProperty<float>("V_tumour");  // initial tumour volume
        }
        // Convert tumour volume to mm3
        tumour_volume /= 1e+9;
    }
    FLAMEGPU->environment.setProperty<float>("validation_tumour_volume", tumour_volume);
    // Histograms
    std::array<unsigned int, 42> histogram_nbl = {};
    std::array<unsigned int, 42> histogram_nba = {};
    std::array<unsigned int, 42> histogram_nbn = {};
    std::array<unsigned int, 42> histogram_scl = {};
    std::array<unsigned int, 42> histogram_sca = {};
    std::array<unsigned int, 42> histogram_scn = {};

    const auto h_nbl = FLAMEGPU->environment.getMacroProperty<unsigned int, 42>("histogram_nbl");
    const auto h_nba = FLAMEGPU->environment.getMacroProperty<unsigned int, 42>("histogram_nba");
    const auto h_nbn = FLAMEGPU->environment.getMacroProperty<unsigned int, 42>("histogram_nbn");
    const auto h_scl = FLAMEGPU->environment.getMacroProperty<unsigned int, 42>("histogram_scl");
    const auto h_sca = FLAMEGPU->environment.getMacroProperty<unsigned int, 42>("histogram_sca");
    const auto h_scn = FLAMEGPU->environment.getMacroProperty<unsigned int, 42>("histogram_scn");

    for (int i = 0; i < 42; ++i) {
        histogram_nbl[i] = h_nbl[i];
        histogram_nba[i] = h_nba[i];
        histogram_nbn[i] = h_nbn[i];
        histogram_scl[i] = h_scl[i];
        histogram_sca[i] = h_sca[i];
        histogram_scn[i] = h_scn[i];
    }

    FLAMEGPU->environment.setProperty<unsigned int, 42>("histogram_nbl", histogram_nbl);
    FLAMEGPU->environment.setProperty<unsigned int, 42>("histogram_nba", histogram_nba);
    FLAMEGPU->environment.setProperty<unsigned int, 42>("histogram_nbn", histogram_nbn);
    FLAMEGPU->environment.setProperty<unsigned int, 42>("histogram_scl", histogram_scl);
    FLAMEGPU->environment.setProperty<unsigned int, 42>("histogram_sca", histogram_sca);
    FLAMEGPU->environment.setProperty<unsigned int, 42>("histogram_scn", histogram_scn);

    //Output intracellular details to initialise the next loop.
    if (NB_living_count) {
        //sim_out.ratio_VEGF_NB_SC = Schwann.count() ? Neuroblastoma.sum<int>("VEGF") / static_cast<float>(Schwann.count()) : 0;
        // Calc mean (living cells only)
        const float nb_telomere_length_mean = Neuroblastoma.sum<int>("telo_count") / static_cast<float>(NB_living_count);
        const float nb_necro_signal_mean = Neuroblastoma.sum<int>("necro_signal") / static_cast<float>(NB_living_count);
        const float nb_apop_signal_mean = Neuroblastoma.sum<int>("apop_signal") / static_cast<float>(NB_living_count);
        const float extent_of_differentiation_mean = Neuroblastoma.sum<float>("degdiff") / static_cast<float>(NB_living_count);
        FLAMEGPU->environment.setProperty<float>("nb_telomere_length_mean", nb_telomere_length_mean);
        FLAMEGPU->environment.setProperty<float>("nb_necro_signal_mean", nb_necro_signal_mean);
        FLAMEGPU->environment.setProperty<float>("nb_apop_signal_mean", nb_apop_signal_mean);
        FLAMEGPU->environment.setProperty<float>("extent_of_differentiation_mean", extent_of_differentiation_mean);
        // Calc the numerator of the sd equation (refered to as mean2 here)
        // The custom transform/reduce, only accounts for variables with non zero values
        // Dead cells are all zero, but some living cells too
        gpuErrchk(cudaMemcpyToSymbol(mean2_mean, &nb_telomere_length_mean, sizeof(float)));
        double nb_telomere_length_mean2 = Neuroblastoma.transformReduce<int, double>("telo_count", mean2_transform, mean2_sum, 0);
        gpuErrchk(cudaMemcpyToSymbol(mean2_mean, &nb_necro_signal_mean, sizeof(float)));
        double nb_necro_signal_mean2 = Neuroblastoma.transformReduce<int, double>("necro_signal", mean2_transform, mean2_sum, 0);
        gpuErrchk(cudaMemcpyToSymbol(mean2_mean, &nb_apop_signal_mean, sizeof(float)));
        double nb_apop_signal_mean2 = Neuroblastoma.transformReduce<int, double>("apop_signal", mean2_transform, mean2_sum, 0);
        gpuErrchk(cudaMemcpyToSymbol(mean2_mean, &extent_of_differentiation_mean, sizeof(float)));
        double extent_of_differentiation_mean2 = Neuroblastoma.transformReduce<float, double>("degdiff", mean2_transform, mean2_sum, 0);
        // Therefore, we add living cells with zero value to the sum too
        const auto nb_telomere_length_count0 = Neuroblastoma.count<int>("telo_count", 0);
        nb_telomere_length_mean2 += (nb_telomere_length_count0 - (NB_apop_count + NB_necro_count)) * pow(nb_telomere_length_mean, 2);
        const auto nb_necro_signal_count0 = Neuroblastoma.count<int>("necro_signal", 0);
        nb_necro_signal_mean2 += (nb_necro_signal_count0 - (NB_apop_count + NB_necro_count)) * pow(nb_necro_signal_mean, 2);
        const auto nb_apop_signal_count0 = Neuroblastoma.count<int>("apop_signal", 0);
        nb_apop_signal_mean2 += (nb_apop_signal_count0 - (NB_apop_count + NB_necro_count)) * pow(nb_apop_signal_mean, 2);
        const auto extent_of_differentiation_count0 = Neuroblastoma.count<float>("degdiff", 0);
        extent_of_differentiation_mean2 += (extent_of_differentiation_count0 - (NB_apop_count + NB_necro_count)) * pow(extent_of_differentiation_mean, 2);
        // Divide and sqrt for the sd
        FLAMEGPU->environment.setProperty<float>("nb_telomere_length_sd", static_cast<float>(sqrt(nb_telomere_length_mean2 / NB_living_count)));
        FLAMEGPU->environment.setProperty<float>("nb_necro_signal_sd", static_cast<float>(sqrt(nb_necro_signal_mean2 / NB_living_count)));
        FLAMEGPU->environment.setProperty<float>("nb_apop_signal_sd", static_cast<float>(sqrt(nb_apop_signal_mean2 / NB_living_count)));
        FLAMEGPU->environment.setProperty<float>("extent_of_differentiation_sd", static_cast<float>(sqrt(extent_of_differentiation_mean2 / NB_living_count)));
    }
    if (SC_living_count) {
        // Calc mean (living cells only)
        const float sc_telomere_length_mean = Schwann.sum<int>("telo_count") / static_cast<float>(SC_living_count);
        const float sc_necro_signal_mean = Schwann.sum<int>("necro_signal") / static_cast<float>(SC_living_count);
        const float sc_apop_signal_mean = Schwann.sum<int>("apop_signal") / static_cast<float>(SC_living_count);
        FLAMEGPU->environment.setProperty<float>("sc_telomere_length_mean", sc_telomere_length_mean);
        FLAMEGPU->environment.setProperty<float>("sc_necro_signal_mean", sc_necro_signal_mean);
        FLAMEGPU->environment.setProperty<float>("sc_apop_signal_mean", sc_apop_signal_mean);
        // Calc the numerator of the sd equation (refered to as mean2 here)
        // The custom transform/reduce, only accounts for variables with non zero values
        // Dead cells are all zero, but some living cells too
        double sc_telomere_length_mean2 = Schwann.transformReduce<int, double>("telo_count", mean2_transform, mean2_sum, 0);
        double sc_necro_signal_mean2 = Schwann.transformReduce<int, double>("necro_signal", mean2_transform, mean2_sum, 0);
        double sc_apop_signal_mean2 = Schwann.transformReduce<int, double>("apop_signal", mean2_transform, mean2_sum, 0);
        // Therefore, we add living cells with zero value to the sum too
        const auto sc_telomere_length_count0 = Schwann.count<int>("telo_count", 0);
        sc_telomere_length_mean2 += (sc_telomere_length_count0 - (SC_apop_count + SC_necro_count)) * pow(sc_telomere_length_mean, 2);
        const auto sc_necro_signal_count0 = Schwann.count<int>("necro_signal", 0);
        sc_necro_signal_mean2 += (sc_necro_signal_count0 - (SC_apop_count + SC_necro_count)) * pow(sc_necro_signal_mean, 2);
        const auto sc_apop_signal_count0 = Schwann.count<int>("apop_signal", 0);
        sc_apop_signal_mean2 += (sc_telomere_length_count0 - (SC_apop_count + SC_necro_count)) * pow(sc_apop_signal_mean, 2);
        // Divide and sqrt for the sd
        FLAMEGPU->environment.setProperty<float>("sc_telomere_length_sd", static_cast<float>(sqrt(sc_telomere_length_mean2 / SC_living_count)));
        FLAMEGPU->environment.setProperty<float>("sc_necro_signal_sd", static_cast<float>(sqrt(sc_necro_signal_mean2 / SC_living_count)));
        FLAMEGPU->environment.setProperty<float>("sc_apop_signal_sd", static_cast<float>(sqrt(sc_apop_signal_mean2 / SC_living_count)));
    }
}

FLAMEGPU_HOST_FUNCTION(toggle_chemo) {
    int chemo_state = 0;
    int chemo_index = -1;
    const std::array<unsigned int, CHEMO_LEN> h_env_chemo_start = FLAMEGPU->environment.getProperty<unsigned int, CHEMO_LEN>("chemo_start");
    const std::array<unsigned int, CHEMO_LEN> h_env_chemo_end = FLAMEGPU->environment.getProperty<unsigned int, CHEMO_LEN>("chemo_end");
    for (unsigned int i = 0; i < sizeof(h_env_chemo_start) / sizeof(h_env_chemo_start[0]); ++i) {
        if (FLAMEGPU->getStepCounter() >= h_env_chemo_start[i]) {
            if (FLAMEGPU->getStepCounter() < h_env_chemo_end[i]) {
                chemo_state = 1;
                chemo_index = i;
                break;
            }
        }
    }
    FLAMEGPU->environment.setProperty<int>("CHEMO_ACTIVE", chemo_state);
    chemo_index *= 6;  // 6 effects per start/end time
    FLAMEGPU->environment.setProperty<int>("CHEMO_OFFSET", chemo_index);
}
