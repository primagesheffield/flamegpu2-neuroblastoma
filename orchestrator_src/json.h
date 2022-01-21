#ifndef ORCHESTRATOR_SRC_JSON_H_
#define ORCHESTRATOR_SRC_JSON_H_

#include "structures.h"

OrchestratorInput readOrchestratorInput(const std::string& inputFile);
void writeOrchestratorOutput(const OrchestratorOutput& out, const std::string& outputFile);
void printHelp(const char* executable);

#endif  // ORCHESTRATOR_SRC_JSON_H_
