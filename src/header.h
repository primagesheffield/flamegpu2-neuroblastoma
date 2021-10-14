#ifndef SRC_HEADER_H
#define SRC_HEADER_H

#include "flamegpu/flamegpu.h"
/**
 * This shared header defines the methods used to define the model
 */
#define GRID_MAX_DIMENSIONS 151
#define GMD GRID_MAX_DIMENSIONS

/**
 * Define all environment properties
 */
void defineEnvironment(flamegpu::ModelDescription &model, unsigned int CELL_COUNT);

/**
 * Define the Neuroblastoma agent properties
 */
flamegpu::AgentDescription& defineNeuroblastoma(flamegpu::ModelDescription& model);
/**
 * Allocate and init all Neuroblastoma agents
 * @note This must follow derived environment initialisation
 */
void initNeuroblastoma(flamegpu::HostAPI& FLAMEGPU);

/**
 * Define the Schwann agent properties
 */
flamegpu::AgentDescription& defineSchwann(flamegpu::ModelDescription& model);
/**
 * Allocate and init all Schwann agents
 * @note This must follow derived environment initialisation
 */
void initSchwann(flamegpu::HostAPI& FLAMEGPU);

/**
 * Define the Grid agent/macro property
 */
flamegpu::AgentDescription& defineGrid(flamegpu::ModelDescription& model);
/**
 * Allocate and init all Grid agents/macro property elements
 * @note This must follow derived environment initialisation
 */
void initGrid(flamegpu::HostAPI& FLAMEGPU);

/**
 * Define the force resolution submodel
 */
flamegpu::SubModelDescription& defineForceResolution(flamegpu::ModelDescription& model);

#endif  // SRC_HEADER_H
