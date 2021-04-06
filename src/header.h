#ifndef SRC_HEADER_H
#define SRC_HEADER_H

#include "flamegpu/flame_api.h"
/**
 * This shared header defines the methods used to define the model
 */

/**
 * Define all environment properties
 */
void defineEnvironment(ModelDescription &model, unsigned int CELL_COUNT);

/**
 * Define the Neuroblastoma agent properties
 */
AgentDescription& defineNeuroblastoma(ModelDescription& model);

/**
 * Define the Schwann agent properties
 */
AgentDescription& defineSchwann(ModelDescription& model);

/**
 * Define the force resolution submodel
 */
SubModelDescription& defineForceResolution(ModelDescription& model);

#endif  // SRC_HEADER_H
