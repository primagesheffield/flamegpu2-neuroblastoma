#include <../rapidjson-src/include/rapidjson/stream.h>
#include <../rapidjson-src/include/rapidjson/reader.h>
#include <../rapidjson-src/include/rapidjson/error/en.h>

#include <string>
#include <iostream>
#include <fstream>
#include <stack>
#include <set>
#include <cassert>

#include "json.h"
/**
 * This is the main sax style parser for the json state
 * It stores it's current position within the hierarchy with mode, lastKey and current_variable_array_index
 */
class JSONStateReader2_impl : public rapidjson::BaseReaderHandler<rapidjson::UTF8<>, JSONStateReader2_impl>  {
    std::string lastKey;
    std::string filename;
    CalibrationInput input;
    int depth;

 public:
    explicit JSONStateReader2_impl(const std::string &_filename, const CalibrationInput &init)
        : filename(_filename)
        , input(init)
        , depth(-1) {
    }
    template<typename T>
    bool processValue(const T&val) {
        if (lastKey == "MYCN_fn11") {
            input.MYCN_fn11 = static_cast<float>(val);
        } else if (lastKey == "MAPK_RAS_fn11") {
            input.MAPK_RAS_fn11 = static_cast<float>(val);
        } else if (lastKey == "MAPK_RAS_fn01") {
            input.MAPK_RAS_fn01 = static_cast<float>(val);
        } else if (lastKey == "p53_fn") {
            input.p53_fn = static_cast<float>(val);
        } else if (lastKey == "p73_fn") {
            input.p73_fn = static_cast<float>(val);
        } else if (lastKey == "HIF_fn") {
            input.HIF_fn = static_cast<float>(val);
        } else if (lastKey == "P_cycle_nb") {
            input.P_cycle_nb = static_cast<float>(val);
        } else if (lastKey == "P_cycle_sc") {
            input.P_cycle_sc = static_cast<float>(val);
        } else if (lastKey == "P_DNA_damageHypo") {
            input.P_DNA_damageHypo = static_cast<float>(val);
        } else if (lastKey == "P_DNA_damagerp") {
            input.P_DNA_damagerp = static_cast<float>(val);
        } else if (lastKey == "P_unrepDNAHypo") {
            input.P_unrepDNAHypo = static_cast<float>(val);
        } else if (lastKey == "P_unrepDNArp") {
            input.P_unrepDNArp = static_cast<float>(val);
        } else if (lastKey == "P_necroIS") {
            input.P_necroIS = static_cast<float>(val);
        } else if (lastKey == "P_telorp") {
            input.P_telorp = static_cast<float>(val);
        } else if (lastKey == "P_apopChemo") {
            input.P_apopChemo = static_cast<float>(val);
        } else if (lastKey == "P_DNA_damage_pathways") {
            input.P_DNA_damage_pathways = static_cast<float>(val);
        } else if (lastKey == "P_apoprp") {
            input.P_apoprp = static_cast<float>(val);
        } else if (lastKey == "P_necrorp") {
            input.P_necrorp = static_cast<float>(val);
        } else if (lastKey == "scpro_jux") {
            input.scpro_jux = static_cast<float>(val);
        } else if (lastKey == "nbdiff_jux") {
            input.nbdiff_jux = static_cast<float>(val);
        } else if (lastKey == "nbdiff_amount") {
            input.nbdiff_amount = static_cast<float>(val);
        } else if (lastKey == "nbapop_jux") {
            input.nbapop_jux = static_cast<float>(val);
        } else if (lastKey == "mig_sc") {
            input.mig_sc = static_cast<float>(val);
        } else {
            fprintf(stderr, "Unexpected key '%s' whilst parsing calibration file, "
                "in JSONStateReader2::parse()\n", lastKey.c_str());
            throw std::exception();
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
    bool String(const char*s, rapidjson::SizeType, bool) {
        // String is not expected
        fprintf(stderr, "Unexpected string whilst parsing input file '%s'.\n", filename.c_str());
        throw std::exception();
    }
    bool StartObject() {
        if (++depth > 0) {
            fprintf(stderr, "Unexpected object start whilst parsing input file '%s'.\n", filename.c_str());
            throw std::exception();
        }
        return true;
    }
    bool Key(const char* str, rapidjson::SizeType, bool) {
        lastKey = str;
        return true;
    }
    bool EndObject(rapidjson::SizeType) {
        --depth;
        return true;
    }
    bool StartArray() {
        // Array is not expected
        fprintf(stderr, "Unexpected array whilst parsing input file '%s'.\n", filename.c_str());
        return true;
    }
    bool EndArray(rapidjson::SizeType) {
        // Array is not expected
        fprintf(stderr, "Unexpected array whilst parsing input file '%s'.\n", filename.c_str());
        return true;
    }

    CalibrationInput getInput() { return input; }
};
CalibrationInput readCalibrationInput(const std::string& inputFile, const CalibrationInput& init) {
    std::ifstream in(inputFile, std::ios::in | std::ios::binary);
    if (!in) {
        fprintf(stderr, "Unable to open file '%s' for reading.\n", inputFile.c_str());
        throw std::exception();
    }
    JSONStateReader2_impl handler(inputFile, init);
    const std::string filestring = std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    rapidjson::StringStream file_ss(filestring.c_str());
    rapidjson::Reader reader;
    rapidjson::ParseResult pr1 = reader.Parse(file_ss, handler);
    if (pr1.Code() != rapidjson::ParseErrorCode::kParseErrorNone) {
        fprintf(stderr, "Whilst parsing input file '%s', RapidJSON returned error: %s\n", inputFile.c_str(), rapidjson::GetParseError_En(pr1.Code()));
        throw std::exception();
    }
    return handler.getInput();
}
