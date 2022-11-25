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
        if (lastKey == "TERT_rarngm") {
            input.TERT_rarngm = static_cast<int32_t>(val);
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
