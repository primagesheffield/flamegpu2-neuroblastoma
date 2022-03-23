#include <algorithm>
#include <glm/gtx/component_wise.hpp>

#include "header.h"
#include "main.h"
#include "json.h"

__device__ float mean2_mean;
FLAMEGPU_CUSTOM_TRANSFORM(mean2_transform, a) {
    if (a != 0)
        return static_cast<double>((a - mean2_mean) * (a - mean2_mean));
    return 0;
}
FLAMEGPU_CUSTOM_REDUCTION(mean2_sum, a, b) {
    return a + b;
}

OrchestratorOutput sim_out;
FLAMEGPU_STEP_FUNCTION(ConstructPrimageOutput) {
    auto GridCell = FLAMEGPU->agent("GridCell");
    auto Neuroblastoma = FLAMEGPU->agent("Neuroblastoma");
    auto Schwann = FLAMEGPU->agent("Schwann");
    sim_out = {};
    sim_out.O2 = FLAMEGPU->environment.getProperty<float>("O2");
    sim_out.ecm = GridCell.sum<float>("matrix_value") / glm::compMul(FLAMEGPU->environment.getProperty<glm::uvec3>("grid_dims"));
    sim_out.material_properties = 0;  // Unused
    sim_out.diffusion_coefficient = 0;  // Unused
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
        sim_out.cellularity = {};
    }
    FLAMEGPU->environment.setProperty<float, 6>("validation_cellularity", sim_out.cellularity);
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
        sim_out.total_volume_ratio_updated = FLAMEGPU->environment.getProperty<float>("V_tumour") / sim_out.tumour_volume;
        // Convert tumour volume to mm3
        sim_out.tumour_volume /= 1e+9;
    }
    if (NB_living_count) {
        sim_out.ratio_VEGF_NB_SC = Schwann.count() ? Neuroblastoma.sum<int>("VEGF") / static_cast<float>(Schwann.count()) : 0;
        // Calc mean (living cells only)
        sim_out.nb_telomere_length_mean = Neuroblastoma.sum<int>("telo_count") / static_cast<float>(NB_living_count);
        sim_out.nb_necro_signal_mean = Neuroblastoma.sum<int>("necro_signal") / static_cast<float>(NB_living_count);
        sim_out.nb_apop_signal_mean = Neuroblastoma.sum<int>("apop_signal") / static_cast<float>(NB_living_count);
        sim_out.extent_of_differentiation_mean = Neuroblastoma.sum<float>("degdiff") / static_cast<float>(NB_living_count);
        // Calc the numerator of the sd equation (refered to as mean2 here)
        // The custom transform/reduce, only accounts for variables with non zero values
        // Dead cells are all zero, but some living cells too
        gpuErrchk(cudaMemcpyToSymbol(mean2_mean, &sim_out.nb_telomere_length_mean, sizeof(float)));
        double nb_telomere_length_mean2 = Neuroblastoma.transformReduce<int, double>("telo_count", mean2_transform, mean2_sum, 0);
        gpuErrchk(cudaMemcpyToSymbol(mean2_mean, &sim_out.nb_necro_signal_mean, sizeof(float)));
        double nb_necro_signal_mean2 = Neuroblastoma.transformReduce<int, double>("necro_signal", mean2_transform, mean2_sum, 0);
        gpuErrchk(cudaMemcpyToSymbol(mean2_mean, &sim_out.nb_apop_signal_mean, sizeof(float)));
        double nb_apop_signal_mean2 = Neuroblastoma.transformReduce<int, double>("apop_signal", mean2_transform, mean2_sum, 0);
        gpuErrchk(cudaMemcpyToSymbol(mean2_mean, &sim_out.extent_of_differentiation_mean, sizeof(float)));
        double extent_of_differentiation_mean2 = Neuroblastoma.transformReduce<float, double>("degdiff", mean2_transform, mean2_sum, 0);
        // Therefore, we add living cells with zero value to the sum too
        const auto nb_telomere_length_count0 = Neuroblastoma.count<int>("telo_count", 0);
        nb_telomere_length_mean2 += (nb_telomere_length_count0 - (NB_apop_count + NB_necro_count)) * pow(sim_out.nb_telomere_length_mean, 2);
        const auto nb_necro_signal_count0 = Neuroblastoma.count<int>("necro_signal", 0);
        nb_necro_signal_mean2 += (nb_necro_signal_count0 - (NB_apop_count + NB_necro_count)) * pow(sim_out.nb_necro_signal_mean, 2);
        const auto nb_apop_signal_count0 = Neuroblastoma.count<int>("apop_signal", 0);
        nb_apop_signal_mean2 += (nb_telomere_length_count0 - (NB_apop_count + NB_necro_count)) * pow(sim_out.nb_apop_signal_mean, 2);
        const auto extent_of_differentiation_count0 = Neuroblastoma.count<float>("degdiff", 0);
        extent_of_differentiation_mean2 += (extent_of_differentiation_count0 - (NB_apop_count + NB_necro_count)) * pow(sim_out.extent_of_differentiation_mean, 2);
        // Divide and sqrt for the sd
        sim_out.nb_telomere_length_sd = static_cast<float>(sqrt(nb_telomere_length_mean2 / NB_living_count));
        sim_out.nb_necro_signal_sd = static_cast<float>(sqrt(nb_necro_signal_mean2 / NB_living_count));
        sim_out.nb_apop_signal_sd = static_cast<float>(sqrt(nb_apop_signal_mean2 / NB_living_count));
        sim_out.extent_of_differentiation_sd = static_cast<float>(sqrt(extent_of_differentiation_mean2 / NB_living_count));
    }
    if (SC_living_count) {
        // Calc mean (living cells only)
        sim_out.sc_telomere_length_mean = Schwann.sum<int>("telo_count") / static_cast<float>(SC_living_count);
        sim_out.sc_necro_signal_mean = Schwann.sum<int>("necro_signal") / static_cast<float>(SC_living_count);
        sim_out.sc_apop_signal_mean = Schwann.sum<int>("apop_signal") / static_cast<float>(SC_living_count);
        // Calc the numerator of the sd equation (refered to as mean2 here)
        // The custom transform/reduce, only accounts for variables with non zero values
        // Dead cells are all zero, but some living cells too
        gpuErrchk(cudaMemcpyToSymbol(mean2_mean, &sim_out.sc_telomere_length_mean, sizeof(float)));
        double sc_telomere_length_mean2 = Schwann.transformReduce<int, double>("telo_count", mean2_transform, mean2_sum, 0);
        gpuErrchk(cudaMemcpyToSymbol(mean2_mean, &sim_out.sc_necro_signal_mean, sizeof(float)));
        double sc_necro_signal_mean2 = Schwann.transformReduce<int, double>("necro_signal", mean2_transform, mean2_sum, 0);
        gpuErrchk(cudaMemcpyToSymbol(mean2_mean, &sim_out.sc_apop_signal_mean, sizeof(float)));
        double sc_apop_signal_mean2 = Schwann.transformReduce<int, double>("apop_signal", mean2_transform, mean2_sum, 0);
        // Therefore, we add living cells with zero value to the sum too
        const auto sc_telomere_length_count0 = Schwann.count<int>("telo_count", 0);
        sc_telomere_length_mean2 += (sc_telomere_length_count0 - (SC_apop_count + SC_necro_count)) * pow(sim_out.sc_telomere_length_mean, 2);
        const auto sc_necro_signal_count0 = Schwann.count<int>("necro_signal", 0);
        sc_necro_signal_mean2 += (sc_necro_signal_count0 - (SC_apop_count + SC_necro_count)) * pow(sim_out.sc_necro_signal_mean, 2);
        const auto sc_apop_signal_count0 = Schwann.count<int>("apop_signal", 0);
        sc_apop_signal_mean2 += (sc_telomere_length_count0 - (SC_apop_count + SC_necro_count)) * pow(sim_out.sc_apop_signal_mean, 2);
        // Divide and sqrt for the sd
        sim_out.sc_telomere_length_sd = static_cast<float>(sqrt(sc_telomere_length_mean2 / SC_living_count));
        sim_out.sc_necro_signal_sd = static_cast<float>(sqrt(sc_necro_signal_mean2 / SC_living_count));
        sim_out.sc_apop_signal_sd = static_cast<float>(sqrt(sc_apop_signal_mean2 / SC_living_count));
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
    model.addStepFunction(ConstructPrimageOutput);
    // Construct fgpu2 inputs
    flamegpu::CUDASimulation sim(model);
    sim.SimulationConfig().step_log_file = cfg.validation_path;
    sim.SimulationConfig().steps = input.steps;
    sim.SimulationConfig().random_seed = input.seed;
    sim.applyConfig();
    sim.setEnvironmentProperty<int>("TERT_rarngm", input.TERT_rarngm);
    sim.setEnvironmentProperty<int>("ATRX_inact", input.ATRX_inact);
    sim.setEnvironmentProperty<float>("V_tumour", static_cast<float>(input.V_tumour * 1e+9));  // Convert from primage mm^3 to micron ^3
    sim.setEnvironmentProperty<float>("O2", input.O2);
    sim.setEnvironmentProperty<float, 6>("cellularity", input.cellularity);
    sim.setEnvironmentProperty<int>("orchestrator_time", input.orchestrator_time);
    sim.setEnvironmentProperty<int>("MYCN_amp", input.MYCN_amp);
    sim.setEnvironmentProperty<int>("ALT", input.ALT);
    sim.setEnvironmentProperty<int>("ALK", input.ALK);
    sim.setEnvironmentProperty<int>("gradiff", input.gradiff);
    sim.setEnvironmentProperty<int>("histology_init", input.histology_init);
    sim.setEnvironmentProperty<float>("nb_telomere_length_mean", input.nb_telomere_length_mean);
    sim.setEnvironmentProperty<float>("nb_telomere_length_sd", input.nb_telomere_length_sd);
    sim.setEnvironmentProperty<float>("sc_telomere_length_mean", input.sc_telomere_length_mean);
    sim.setEnvironmentProperty<float>("sc_telomere_length_sd", input.sc_telomere_length_sd);
    sim.setEnvironmentProperty<float>("extent_of_differentiation_mean", input.extent_of_differentiation_mean);
    sim.setEnvironmentProperty<float>("extent_of_differentiation_sd", input.extent_of_differentiation_sd);
    sim.setEnvironmentProperty<float>("nb_necro_signal_mean", input.nb_necro_signal_mean);
    sim.setEnvironmentProperty<float>("nb_necro_signal_sd", input.nb_necro_signal_sd);
    sim.setEnvironmentProperty<float>("nb_apop_signal_mean", input.nb_apop_signal_mean);
    sim.setEnvironmentProperty<float>("nb_apop_signal_sd", input.nb_apop_signal_sd);
    sim.setEnvironmentProperty<float>("sc_necro_signal_mean", input.sc_necro_signal_mean);
    sim.setEnvironmentProperty<float>("sc_necro_signal_sd", input.sc_necro_signal_sd);
    sim.setEnvironmentProperty<float>("sc_apop_signal_mean", input.sc_apop_signal_mean);
    sim.setEnvironmentProperty<float>("sc_apop_signal_sd", input.sc_apop_signal_sd);
    std::array<float, 6 * CHEMO_LEN> chemo_effects = { };
    memcpy(chemo_effects.data(), input.drug_effects.data(), min(input.drug_effects.size(), chemo_effects.size()) * sizeof(float));
    sim.setEnvironmentProperty<float, 6 * CHEMO_LEN>("chemo_effects", chemo_effects);
    std::array<unsigned int, CHEMO_LEN> chemo_start = { };
    memcpy(chemo_start.data(), input.start_effects.data(), min(input.start_effects.size(), chemo_start.size()) * sizeof(unsigned int));
    sim.setEnvironmentProperty<unsigned int, CHEMO_LEN>("chemo_start", chemo_start);
    std::array<unsigned int, CHEMO_LEN> chemo_end = { };
    memcpy(chemo_end.data(), input.end_effects.data(), min(input.end_effects.size(), chemo_end.size()) * sizeof(unsigned int));
    sim.setEnvironmentProperty<unsigned int, CHEMO_LEN>("chemo_end", chemo_end);


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
        step_log_cfg.logEnvironment("Nscl_count");  //?
        step_log_cfg.logEnvironment("Nnbl_count");  //?
        step_log_cfg.logEnvironment("MYCN_amp");
        step_log_cfg.logEnvironment("TERT_rarngm");
        step_log_cfg.logEnvironment("ATRX_inact");
        step_log_cfg.logEnvironment("ALT");
        step_log_cfg.logEnvironment("ALK");
        step_log_cfg.logEnvironment("telo_count");
        step_log_cfg.logEnvironment("P_necrorp");
        step_log_cfg.logEnvironment("P_necroIS");
        step_log_cfg.logEnvironment("force_resolution_steps");
        step_log_cfg.logEnvironment("validation_cellularity");
        step_log_cfg.agent("Neuroblastoma").logCount();
        step_log_cfg.agent("Neuroblastoma").logMean<int>("apop");
        step_log_cfg.agent("Neuroblastoma").logMean<int>("necro");
        step_log_cfg.agent("Neuroblastoma").logSum<int>("necro");
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
        step_log_cfg.agent("Schwann").logSum<int>("necro");
        step_log_cfg.agent("Schwann").logMean<int>("necro_signal");
        step_log_cfg.agent("Schwann").logMean<int>("apop_signal");
        step_log_cfg.agent("Schwann").logMean<unsigned int>("cycle");
        step_log_cfg.agent("Schwann").logMean<int>("hypoxia");
        step_log_cfg.agent("Schwann").logMean<int>("DNA_damage");
        step_log_cfg.agent("Schwann").logMean<int>("ATP");
        step_log_cfg.agent("Schwann").logMean<int>("nutrient");
        step_log_cfg.agent("Schwann").logMean<int>("telo_count");
        step_log_cfg.agent("Schwann").logMean<int>("neighbours");
        step_log_cfg.logEnvironment("grid_origin");
        step_log_cfg.logEnvironment("grid_dims");
    }
    sim.setStepLog(step_log_cfg);
    
    // Run FGPU2
    sim.simulate();
    // Update delta outputs
    sim_out.delta_O2 = sim_out.O2 - input.O2;
    float cellularity_sum = 0;
    for (const float &c : input.cellularity)
        cellularity_sum += c;
    sim_out.delta_ecm = sim_out.ecm - (1 - cellularity_sum);
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
        std::transform(arg.begin(), arg.end(), arg.begin(), [](unsigned char c) { return std::use_facet< std::ctype<char>>(std::locale()).tolower(c); });

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
        if (arg.compare("--vd") == 0) {
            cfg.validation_path = std::string(argv[++i]);
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
    if (cfg.validation_path.empty()) {
        fprintf(stderr, "Pvalidation_path not specified!\n");
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
    printf(line_fmt, "--vd", "Validation output directory");
    printf("Optional Arguments:\n");
    printf(line_fmt, "-d, --device", "GPU index");
    printf("Note: The orchestrator interface to the FGPU2 model has a reduced subset of command line arguments.\n");

    exit(EXIT_FAILURE);
}
