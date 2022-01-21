#ifndef ORCHESTRATOR_SRC_MAIN_H_
#define ORCHESTRATOR_SRC_MAIN_H_

#include <string>

/**
 * Commandline execution configuration options and default values
 */
struct RunConfig {
    std::string inFile;
    std::string primageOutputFile;
    unsigned int device = 0;
};
RunConfig parseArgs(int argc, const char** argv);

#endif  // ORCHESTRATOR_SRC_MAIN_H_
