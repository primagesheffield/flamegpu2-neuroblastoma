#include <numeric>
#include <algorithm>
#include <functional>
#include <glm/gtx/component_wise.hpp>

#include "header.h"
#include "main.h"
#include "json.h"

OrchestratorOutput sim_out;
FLAMEGPU_EXIT_FUNCTION(ConstructPrimageOutput) {
    auto GridCell = FLAMEGPU->agent("GridCell");
    auto Neuroblastoma = FLAMEGPU->agent("Neuroblastoma");
    auto Schwann = FLAMEGPU->agent("Schwann");
    memset(&sim_out, 0, sizeof(OrchestratorOutput));
    // @todo float delta_O2;
    sim_out.O2 = FLAMEGPU->environment.getProperty<float>("O2");
    // @todo float delta_ecm;
    sim_out.ecm = GridCell.sum<float>("matrix_value") / glm::compMul(FLAMEGPU->environment.getProperty<glm::uvec3>("grid_dims"));
    sim_out.material_properties = 0;  // Unused
    sim_out.diffusion_coefficient = 0;  // Unused
    sim_out.total_volume_ratio_updated = 0;  // Unused
    // Cellularity
    const unsigned int NB_living_count = FLAMEGPU->environment.getProperty<unsigned int>("validation_Nnbl");
    const unsigned int NB_apop_count = Neuroblastoma.sum<int>("apop");
    const unsigned int NB_necro_count = Neuroblastoma.sum<int>("necro");
    const unsigned int SC_living_count = FLAMEGPU->environment.getProperty<unsigned int>("validation_Nscl");
    const unsigned int SC_apop_count = Schwann.sum<int>("apop");
    const unsigned int SC_necro_count = Schwann.sum<int>("necro");
    const unsigned int TOTAL_CELL_COUNT = NB_living_count + NB_apop_count + NB_necro_count + SC_living_count + SC_apop_count + SC_necro_count;
    // Calculate each fraction (e.g. number of living SCs/number of all cells) and multiply it by (1-matrix).
    if (TOTAL_CELL_COUNT) {
        sim_out.cellularity[0] = NB_living_count * (1.0f - sim_out.ecm) / TOTAL_CELL_COUNT;
        sim_out.cellularity[1] = NB_apop_count * (1.0f - sim_out.ecm) / TOTAL_CELL_COUNT;
        sim_out.cellularity[2] = NB_necro_count * (1.0f - sim_out.ecm) / TOTAL_CELL_COUNT;
        sim_out.cellularity[3] = SC_living_count * (1.0f - sim_out.ecm) / TOTAL_CELL_COUNT;
        sim_out.cellularity[4] = SC_apop_count * (1.0f - sim_out.ecm) / TOTAL_CELL_COUNT;
        sim_out.cellularity[5] = SC_necro_count * (1.0f - sim_out.ecm) / TOTAL_CELL_COUNT;
    } else {
        memset(sim_out.cellularity.data(), 0, sizeof(float) * 6);
    }
    // Tumour volume
    {
        const int total_cell_count = Neuroblastoma.count() + Schwann.count();
        if (total_cell_count) {
            const float rho_tumour = FLAMEGPU->environment.getProperty<float>("rho_tumour");
            const float matrix_dummy = FLAMEGPU->environment.getProperty<float>("matrix_dummy");
            sim_out.tumour_volume = total_cell_count / rho_tumour / (1 - matrix_dummy);
        } else {
            sim_out.tumour_volume = FLAMEGPU->environment.getProperty<float>("V_tumour");  // initial tumour volume
        }
        // Convert tumour volume to mm3
        sim_out.tumour_volume /= 1e+9;
    }
    if (NB_living_count) {
        sim_out.ratio_VEGF_NB_SC = Schwann.count() ? Neuroblastoma.sum<int>("VEGF") / static_cast<float>(Schwann.count()) : 0;
        const auto telomere_length = Neuroblastoma.meanStandardDeviation<int>("telo_count");
        sim_out.nb_telomere_length_mean = static_cast<float>(telomere_length.first);
        sim_out.nb_telomere_length_sd = static_cast<float>(telomere_length.second);
        const auto necro_signal = Neuroblastoma.meanStandardDeviation<int>("necro_signal");
        sim_out.nb_necro_signal_mean = static_cast<float>(necro_signal.first);
        sim_out.nb_necro_signal_sd = static_cast<float>(necro_signal.second);
        const auto apop_signal = Neuroblastoma.meanStandardDeviation<int>("apop_signal");
        sim_out.nb_apop_signal_mean = static_cast<float>(apop_signal.first);
        sim_out.nb_apop_signal_sd = static_cast<float>(apop_signal.second);
        const auto degdiff = Neuroblastoma.meanStandardDeviation<float>("degdiff");
        sim_out.extent_of_differentiation_mean = static_cast<float>(degdiff.first);
        sim_out.extent_of_differentiation_sd = static_cast<float>(degdiff.second);
    }
    if (SC_living_count) {
        const auto telomere_length = Schwann.meanStandardDeviation<int>("telo_count");
        sim_out.sc_telomere_length_mean = static_cast<float>(telomere_length.first);
        sim_out.sc_telomere_length_sd = static_cast<float>(telomere_length.second);
        const auto necro_signal = Schwann.meanStandardDeviation<int>("necro_signal");
        sim_out.sc_necro_signal_mean = static_cast<float>(necro_signal.first);
        sim_out.sc_necro_signal_sd = static_cast<float>(necro_signal.second);
        const auto apop_signal = Schwann.meanStandardDeviation<int>("apop_signal");
        sim_out.sc_apop_signal_mean = static_cast<float>(apop_signal.first);
        sim_out.sc_apop_signal_sd = static_cast<float>(apop_signal.second);
    }
}
int main(int argc, const char** argv) {
    // Parse commandline
    RunConfig cfg = parseArgs(argc, argv);
    // Parse input file
    OrchestratorInput input = readOrchestratorInput(cfg.inFile);
    // Setup model
    flamegpu::ModelDescription model("PRIMAGE: Neuroblastoma");
    defineModel(model);
    // Add function to construct orchestrator output
    model.addExitFunction(ConstructPrimageOutput);
    // Construct fgpu2 inputs
    flamegpu::CUDASimulation sim(model);
    sim.SimulationConfig().steps = input.steps;
    sim.SimulationConfig().random_seed = input.seed;
    sim.setEnvironmentProperty<int>("TERT_rarngm", input.TERT_rarngm);
    sim.setEnvironmentProperty<int>("ATRX_inact", input.ATRX_inact);
    sim.setEnvironmentProperty<float>("V_tumour", input.V_tumour);
    sim.setEnvironmentProperty<float>("O2", input.O2);
    sim.setEnvironmentProperty<float, 6>("cellularity", input.cellularity);
    // @todo orchestrator_time
    sim.setEnvironmentProperty<int>("MYCN_amp", input.MYCN_amp);
    sim.setEnvironmentProperty<int>("ALT", input.ALT);
    sim.setEnvironmentProperty<int>("ALK", input.ALK);
    sim.setEnvironmentProperty<int>("gradiff", input.gradiff);
    sim.setEnvironmentProperty<int>("histology_init", input.histology_init);
    // @todo nb_telomere_length_mean
    // @todo nb_telomere_length_sd
    // @todo sc_telomere_length_mean
    // @todo sc_telomere_length_sd
    // @todo extent_of_differentiation_mean
    // @todo extent_of_differentiation_sd
    // @todo nb_necro_signal_mean
    // @todo nb_necro_signal_sd
    // @todo nb_apop_signal_mean
    // @todo nb_apop_signal_sd
    // @todo sc_necro_signal_mean
    // @todo sc_necro_signal_sd
    // @todo sc_apop_signal_mean
    // @todo sc_apop_signal_sd
    // @todo drug_effects
    // @todo start_effects
    // @todo end_effects

    // Run FGPU2
    sim.simulate();
    // Update delta outputs
    sim_out.delta_O2 = sim_out.O2 - input.O2;
    sim_out.delta_ecm = sim_out.ecm - (1 - std::reduce(input.cellularity.begin(), input.cellularity.end(), 0, std::plus<>()));
    // Write orchestrator output to disk
    writeOrchestratorOutput(sim_out, cfg.primageOutputFile);
    return 0;
}
RunConfig parseArgs(int argc, const char** argv) {
    if (argc == 1) {
        printHelp(argv[0]);
    }
    RunConfig cfg;
    int i = 1;
    for (; i < argc; i++) {
        // Get arg as lowercase
        std::string arg(argv[i]);
        std::transform(arg.begin(), arg.end(), arg.begin(), ::tolower);

        // -device <uint>, Uses the specified cuda device, defaults to 0
        if (arg.compare("--device") == 0 || arg.compare("-d") == 0) {
            cfg.device = (unsigned int)strtoul(argv[++i], nullptr, 0);
            continue;
        }
        // -in <string>, Specifies the input state file
        if (arg.compare("--in") == 0 || arg.compare("-i") == 0) {
            cfg.inFile = std::string(argv[++i]);
            continue;
        }
        if (arg.compare("--primage") == 0) {
            cfg.primageOutputFile = std::string(argv[++i]);
            continue;
        }
        fprintf(stderr, "Unexpected argument: %s\n", arg.c_str());
        printHelp(argv[0]);
    }
    if (cfg.inFile.empty()) {
        fprintf(stderr, "Input file not specified!\n");
        printHelp(argv[0]);
    }
    if (cfg.primageOutputFile.empty()) {
        fprintf(stderr, "Primage output file not specified!\n");
        printHelp(argv[0]);
    }
    return cfg;
}

void printHelp(const char* executable) {
    printf("Usage: %s <options>\n\n", executable);
    const char* line_fmt = "%-18s %s\n";
    printf("Mandatory Arguments:\n");
    printf(line_fmt, "-i, --in", "JSON initial state input filepath");
    printf(line_fmt, "--primage", "Primage exit data output file");
    printf("Optional Arguments:\n");
    printf(line_fmt, "-d, --device", "GPU index");
    printf("Note: The orchestrator interface to the FGPU2 model has a reduced subset of command line arguments.\n");

    exit(EXIT_FAILURE);
}
