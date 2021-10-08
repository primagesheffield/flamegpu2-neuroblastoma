#ifndef SRC_HEADER_H
#define SRC_HEADER_H

#include "flamegpu/flamegpu.h"
/**
 * This shared header defines the methods used to define the model
 */

/**
 * Define all environment properties
 */
void defineEnvironment(flamegpu::ModelDescription &model, unsigned int CELL_COUNT);

/**
 * Define the Neuroblastoma agent properties
 */
flamegpu::AgentDescription& defineNeuroblastoma(flamegpu::ModelDescription& model);
/**
 * Allocate and init all neuroblastoma agents
 * @note This must follow derived environment initialisation
 */
void initNeuroblastoma(flamegpu::HostAPI& FLAMEGPU);

/**
 * Define the Schwann agent properties
 */
flamegpu::AgentDescription& defineSchwann(flamegpu::ModelDescription& model);
/**
 * Allocate and init all neuroblastoma agents
 * @note This must follow derived environment initialisation
 */
void initSchwann(flamegpu::HostAPI& FLAMEGPU);

/**
 * Define the force resolution submodel
 */
flamegpu::SubModelDescription& defineForceResolution(flamegpu::ModelDescription& model);

#endif  // SRC_HEADER_H
