#include <../rapidjson-src/include/rapidjson/stream.h>
#include <../rapidjson-src/include/rapidjson/reader.h>
#include <../rapidjson-src/include/rapidjson/error/en.h>

#include <string>
#include <iostream>
#include <fstream>
#include <stack>
#include <set>

#include "json.h"
#include "header.h"
/**
 * This is the main sax style parser for the json state
 * It stores it's current position within the hierarchy with mode, lastKey and current_variable_array_index
 */
class JSONStateReader_impl : public rapidjson::BaseReaderHandler<rapidjson::UTF8<>, JSONStateReader_impl>  {
    enum Mode{ Nop, Root, Config, Environment, Version, VariableArray };
    std::stack<Mode> mode;
    std::set<std::string> found_keys;
    std::string lastKey;
    std::string filename;
    OrchestratorInput input;
    /**
     * Tracks current position reading variable array
     */
    unsigned int current_variable_array_index = 0;
    /**
     * Set when we enter an agent
     */
    std::string current_agent;
    /**
     * Set when we enter a state
     */
    std::string current_state;

 public:
    explicit JSONStateReader_impl(const std::string &_filename)
        : filename(_filename) {
        input = {};
    }
    template<typename T>
    bool processValue(const T&val) {
        Mode isArray = Nop;
        if (mode.top() == VariableArray) {
            isArray = mode.top();
            mode.pop();
        }
        if (mode.top() == Environment) {
            if (lastKey == "TERT_rarngm") {
                input.TERT_rarngm = static_cast<int32_t>(val);
            } else if (lastKey == "ATRX_inact") {
                input.ATRX_inact = static_cast<int32_t>(val);
            } else if (lastKey == "V_tumour") {
                input.V_tumour = static_cast<float>(val);
            } else if (lastKey == "O2") {
                input.O2 = static_cast<float>(val);
            } else if (lastKey == "cellularity") {
                if (current_variable_array_index < std::size(input.cellularity)) {
                    input.cellularity[current_variable_array_index++] = static_cast<float>(val);
                } else {
                    fprintf(stderr, "Index %d out of bounds for cellularity array in environment block, "
                        "in JSONStateReader::parse()\n", current_variable_array_index);
                    throw std::exception();
                }
            } else if (lastKey == "orchestrator_time") {
                input.orchestrator_time = static_cast<int32_t>(val);
            } else if (lastKey == "MYCN_amp") {
                input.MYCN_amp = static_cast<int32_t>(val);
            } else if (lastKey == "ALT") {
                input.ALT = static_cast<int32_t>(val);
            } else if (lastKey == "ALK") {
                input.ALK = static_cast<int32_t>(val);
            } else if (lastKey == "gradiff") {
                input.gradiff = static_cast<int32_t>(val);
            } else if (lastKey == "histology_init") {
                input.histology_init = static_cast<int32_t>(val);
            } else if (lastKey == "nb_telomere_length_mean") {
                input.nb_telomere_length_mean = static_cast<float>(val);
            } else if (lastKey == "nb_telomere_length_sd") {
                input.nb_telomere_length_sd = static_cast<float>(val);
            } else if (lastKey == "sc_telomere_length_mean") {
                input.sc_telomere_length_mean = static_cast<float>(val);
            } else if (lastKey == "sc_telomere_length_sd") {
                input.sc_telomere_length_sd = static_cast<float>(val);
            } else if (lastKey == "extent_of_differentiation_mean") {
                input.extent_of_differentiation_mean = static_cast<float>(val);
            } else if (lastKey == "extent_of_differentiation_sd") {
                input.extent_of_differentiation_sd = static_cast<float>(val);
            } else if (lastKey == "nb_necro_signal_mean") {
                input.nb_necro_signal_mean = static_cast<float>(val);
            } else if (lastKey == "nb_necro_signal_sd") {
                input.nb_necro_signal_sd = static_cast<float>(val);
            } else if (lastKey == "nb_apop_signal_mean") {
                input.nb_apop_signal_mean = static_cast<float>(val);
            } else if (lastKey == "nb_apop_signal_sd") {
                input.nb_apop_signal_sd = static_cast<float>(val);
            } else if (lastKey == "sc_necro_signal_mean") {
                input.sc_necro_signal_mean = static_cast<float>(val);
            } else if (lastKey == "sc_necro_signal_sd") {
                input.sc_necro_signal_sd = static_cast<float>(val);
            } else if (lastKey == "sc_apop_signal_mean") {
                input.sc_apop_signal_mean = static_cast<float>(val);
            } else if (lastKey == "sc_apop_signal_sd") {
                input.sc_apop_signal_sd = static_cast<float>(val);
            } else if (lastKey == "drug_effects") {
                input.drug_effects.push_back(static_cast<float>(val));
            } else if (lastKey == "start_effects") {
                input.start_effects.push_back(static_cast<int32_t>(val));
            } else if (lastKey == "end_effects") {
                input.end_effects.push_back(static_cast<int32_t>(val));
            } else {
                fprintf(stderr, "Unexpected key '%s' in environment block, "
                    "in JSONStateReader::parse()\n", lastKey.c_str());
                throw std::exception();
            }
        } else if (mode.top() == Config) {
            if (lastKey == "seed") {
                input.seed = static_cast<uint32_t>(val);
            } else if (lastKey == "steps") {
                input.steps = static_cast<uint32_t>(val);
            } else {
                fprintf(stderr, "Unexpected key '%s' in config block, "
                    "in JSONStateReader::parse()\n", lastKey.c_str());
                throw std::exception();
            }
        } else if (mode.top() == Root && lastKey == "version") {
            // Do nothing, we currently don't handle the version input
        } else {
            fprintf(stderr, "Unexpected value whilst parsing input file '%s'.\n", filename.c_str());
            throw std::exception();
        }
        if (isArray == VariableArray) {
            mode.push(isArray);
        } else {
            current_variable_array_index = 0;  // Didn't actually want to increment it above, because not in an array
        }
        return true;
    }
    bool Null() { return true; }
    bool Bool(bool b) { return processValue<bool>(b); }
    bool Int(int i) { return processValue<int32_t>(i); }
    bool Uint(unsigned u) { return processValue<uint32_t>(u); }
    bool Int64(int64_t i) { return processValue<int64_t>(i); }
    bool Uint64(uint64_t u) { return processValue<uint64_t>(u); }
    bool Double(double d) { return processValue<double>(d); }
    bool String(const char*, rapidjson::SizeType, bool) {
        // String is not expected
        fprintf(stderr, "Unexpected string whilst parsing input file '%s'.\n", filename.c_str());
        throw std::exception();
    }
    bool StartObject() {
        if (mode.empty()) {
            mode.push(Root);
        } else if (mode.top() == Root) {
            if (lastKey == "config") {
                mode.push(Config);
            } else if (lastKey == "version") {
                mode.push(Version);
            } else {
                fprintf(stderr, "Unexpected object start whilst parsing input file '%s'.\n", filename.c_str());
                throw std::exception();
            }
        } else if (mode.top() == Config) {
            if (lastKey == "environment") {
                mode.push(Environment);
            }
        } else {
            fprintf(stderr, "Unexpected object start whilst parsing input file '%s'.\n", filename.c_str());
            throw std::exception();
        }
        return true;
    }
    bool Key(const char* str, rapidjson::SizeType, bool) {
        lastKey = str;
        if (!found_keys.insert(lastKey).second) {
            fprintf(stderr, "Unexpected duplicate of key '%s' found.\n", lastKey.c_str());
            throw std::exception();
        }
        return true;
    }
    bool EndObject(rapidjson::SizeType) {
        mode.pop();
        return true;
    }
    bool StartArray() {
        if (current_variable_array_index != 0) {
            fprintf(stderr, "Array start when current_variable_array_index !=0, in file '%s'. This should never happen.\n", filename.c_str());
            throw std::exception();
        }
        if (mode.top() == Environment || mode.top() == Root) {
            mode.push(VariableArray);
        } else {
            fprintf(stderr, "Unexpected array start whilst parsing input file '%s'.\n", filename.c_str());
            throw std::exception();
        }
        return true;
    }
    bool EndArray(rapidjson::SizeType) {
        if (mode.top() == VariableArray) {
            if (lastKey == "cellularity" && current_variable_array_index !=6) {
                fprintf(stderr, "Cellularity array should have length 6 (%u != 6).\n", current_variable_array_index);
                throw std::exception();
            }
            current_variable_array_index = 0;
        }
        mode.pop();
        return true;
    }
    void validateInput() {
        if (found_keys.size() == 33) {
            if (input.start_effects.size() == input.end_effects.size()) {
                if (input.drug_effects.size() != 6 * input.end_effects.size()) {
                    fprintf(stderr, "Input validation failed.\n'drug_effects' should be 6x the length of 'end_effects' should have the same length (%u != %u == 6 x %u).\n",
                        static_cast<uint32_t>(input.drug_effects.size()), static_cast<uint32_t>(6*input.end_effects.size()), static_cast<uint32_t>(input.end_effects.size()));
                    throw std::exception();
                } else if (input.start_effects.size() > CHEMO_LEN) {
                    fprintf(stderr, "Input validation failed.\nA maximum of %d chemo events can be specified, %u were specified.\n",
                    CHEMO_LEN,
                    static_cast<uint32_t>(input.start_effects.size()));
                    throw std::exception();
                }
            } else {
                fprintf(stderr, "Input validation failed.\n'start_effects' and 'end_effects' should have the same length (%lu != %lu).\n",
                    static_cast<uint32_t>(input.start_effects.size()), static_cast<uint32_t>(input.end_effects.size()));
                throw std::exception();
            }
        } else {
            // Report missing keys
            fprintf(stderr, "Input validation failed.\nThe following keys were missing from input:\n");
            if (!found_keys.count("version")) printf("version\n");
            if (!found_keys.count("config")) printf("config\n");
            if (!found_keys.count("seed")) printf("seed\n");
            if (!found_keys.count("steps")) printf("steps\n");
            if (!found_keys.count("environment")) printf("environment\n");
            if (!found_keys.count("TERT_rarngm")) printf("TERT_rarngm\n");
            if (!found_keys.count("ATRX_inact")) printf("ATRX_inact\n");
            if (!found_keys.count("V_tumour")) printf("V_tumour\n");
            if (!found_keys.count("O2")) printf("O2\n");
            if (!found_keys.count("cellularity")) printf("cellularity\n");
            if (!found_keys.count("orchestrator_time")) printf("orchestrator_time\n");
            if (!found_keys.count("MYCN_amp")) printf("MYCN_amp\n");
            if (!found_keys.count("ALT")) printf("ALT\n");
            if (!found_keys.count("ALK")) printf("ALK\n");
            if (!found_keys.count("gradiff")) printf("gradiff\n");
            if (!found_keys.count("histology_init")) printf("histology_init\n");
            if (!found_keys.count("nb_telomere_length_mean")) printf("nb_telomere_length_mean\n");
            if (!found_keys.count("nb_telomere_length_sd")) printf("nb_telomere_length_sd\n");
            if (!found_keys.count("sc_telomere_length_mean")) printf("sc_telomere_length_mean\n");
            if (!found_keys.count("sc_telomere_length_sd")) printf("sc_telomere_length_sd\n");
            if (!found_keys.count("extent_of_differentiation_mean")) printf("extent_of_differentiation_mean\n");
            if (!found_keys.count("extent_of_differentiation_sd")) printf("extent_of_differentiation_sd\n");
            if (!found_keys.count("nb_necro_signal_mean")) printf("nb_necro_signal_mean\n");
            if (!found_keys.count("nb_necro_signal_sd")) printf("nb_necro_signal_sd\n");
            if (!found_keys.count("nb_apop_signal_mean")) printf("nb_apop_signal_mean\n");
            if (!found_keys.count("nb_apop_signal_sd")) printf("nb_apop_signal_sd\n");
            if (!found_keys.count("sc_necro_signal_mean")) printf("sc_necro_signal_mean\n");
            if (!found_keys.count("sc_necro_signal_sd")) printf("sc_necro_signal_sd\n");
            if (!found_keys.count("sc_apop_signal_mean")) printf("sc_apop_signal_mean\n");
            if (!found_keys.count("sc_apop_signal_sd")) printf("sc_apop_signal_sd\n");
            if (!found_keys.count("drug_effects")) printf("drug_effects\n");
            if (!found_keys.count("start_effects")) printf("start_effects\n");
            if (!found_keys.count("end_effects")) printf("end_effects\n");
            throw std::exception();
        }
    }
    OrchestratorInput getInput() { return input; }
};
OrchestratorInput readOrchestratorInput(const std::string& inputFile) {
    std::ifstream in(inputFile, std::ios::in | std::ios::binary);
    if (!in) {
        fprintf(stderr, "Unable to open file '%s' for reading.\n", inputFile.c_str());
        throw std::exception();
    }
    JSONStateReader_impl handler(inputFile);
    const std::string filestring = std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    rapidjson::StringStream file_ss(filestring.c_str());
    rapidjson::Reader reader;
    rapidjson::ParseResult pr1 = reader.Parse(file_ss, handler);
    if (pr1.Code() != rapidjson::ParseErrorCode::kParseErrorNone) {
        fprintf(stderr, "Whilst parsing input file '%s', RapidJSON returned error: %s\n", inputFile.c_str(), rapidjson::GetParseError_En(pr1.Code()));
        throw std::exception();
    }
    handler.validateInput();
    return handler.getInput();
}
