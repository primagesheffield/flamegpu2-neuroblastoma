#ifndef SRC_HEADER_H_
#define SRC_HEADER_H_

#include "flamegpu/flamegpu.h"
/**
 * This shared header defines the methods used to define the model
 */
#define GRID_MAX_DIMENSIONS 151
#define GMD GRID_MAX_DIMENSIONS

/**
 * Define model
 */
void defineModel(flamegpu::ModelDescription &model);
#ifdef VISUALISATION
/**
 * Define visualisation
 */
flamegpu::visualiser::ModelVis& defineVisualisation(flamegpu::ModelDescription& model, flamegpu::CUDASimulation& sim);
#endif

/**
 * Define all environment properties
 */
void defineEnvironment(flamegpu::ModelDescription &model);

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


/**
 * Converts a continuous location to a discrete grid position
 */
template<typename Mi, typename Mo>
__device__ __forceinline__ glm::ivec3 toGrid(flamegpu::DeviceAPI<Mi, Mo>* FLAMEGPU, const glm::vec3& location) {
    const flamegpu::DeviceEnvironment &env = FLAMEGPU->environment;  // Have to split this out otherwise template keyword required before getProperty
    const float R_voxel = env.getProperty<float>("R_voxel");
    const glm::uvec3 grid_dims = env.getProperty<glm::uvec3>("grid_dims");
    const glm::vec3 span = glm::vec3(grid_dims) * R_voxel * 2.0f;
    const glm::uvec3 grid_origin = env.getProperty<glm::uvec3>("grid_origin");
    return glm::ivec3(
        grid_origin.x + floor((location.x + span.x / 2.0f) / R_voxel / 2.0f),
        grid_origin.y + floor((location.y + span.y / 2.0f) / R_voxel / 2.0f),
        grid_origin.z + floor((location.z + span.z / 2.0f) / R_voxel / 2.0f));
}
/**
 * Notify the NB tracking grid counters of the passed cell's state
 */
template<typename Mi, typename Mo>
__device__ __forceinline__ void increment_grid_nb(flamegpu::DeviceAPI<Mi, Mo>* FLAMEGPU, const glm::ivec3& gid) {
    const flamegpu::DeviceEnvironment& env = FLAMEGPU->environment;  // Have to split this out otherwise template keyword required before getProperty
    // Notify that we are present
    ++env.getMacroProperty<unsigned int, GMD, GMD, GMD>("Nnb_grid")[gid.x][gid.y][gid.z];
    if (FLAMEGPU->template getVariable<int>("apop") == 1) {
        // ++env.getMacroProperty<unsigned int, GMD, GMD, GMD>("Nnba_grid")[gid.x][gid.y][gid.z];
    } else if (FLAMEGPU->template getVariable<int>("necro") == 1) {
        ++env.getMacroProperty<unsigned int, GMD, GMD, GMD>("Nnbn_grid")[gid.x][gid.y][gid.z];
    } else {
        ++env.getMacroProperty<unsigned int, GMD, GMD, GMD>("Nnbl_grid")[gid.x][gid.y][gid.z];
    }
}
/**
 * Notify the SC tracking grid counters of the passed cell's state
 */
template<typename Mi, typename Mo>
__device__ void increment_grid_sc(flamegpu::DeviceAPI<Mi, Mo>* FLAMEGPU, const glm::ivec3& gid) {
    const flamegpu::DeviceEnvironment& env = FLAMEGPU->environment;  // Have to split this out otherwise template keyword required before getProperty
    // Notify that we are present
    ++env.getMacroProperty<unsigned int, GMD, GMD, GMD>("Nsc_grid")[gid.x][gid.y][gid.z];
    if (FLAMEGPU->template getVariable<int>("apop") == 1) {
        // ++env.getMacroProperty<unsigned int, GMD, GMD, GMD>("Nsca_grid")[gid.x][gid.y][gid.z];
    } else if (FLAMEGPU->template getVariable<int>("necro") == 1) {
        ++env.getMacroProperty<unsigned int, GMD, GMD, GMD>("Nscn_grid")[gid.x][gid.y][gid.z];
    } else {
        ++env.getMacroProperty<unsigned int, GMD, GMD, GMD>("Nscl_grid")[gid.x][gid.y][gid.z];
        if (FLAMEGPU->template getVariable<int>("neighbours") < env.getProperty<int>("N_neighbours")) {
            ++env.getMacroProperty<unsigned int, GMD, GMD, GMD>("Nscl_col_grid")[gid.x][gid.y][gid.z];
        }
    }
}
/**
 * Common function used by nb and sc cell lifecycle to shift cells which divide
 */
__device__ __forceinline__ glm::vec3 drift(flamegpu::DeviceAPI<flamegpu::MessageNone, flamegpu::MessageNone>* FLAMEGPU) {
    // Randomly and slightly move a generic cell.
    glm::vec3 location = FLAMEGPU->template getVariable<glm::vec3>("xyz");
    const float R_cell = FLAMEGPU->environment.getProperty<float>("R_cell");
    const glm::vec3 dummy_dir = glm::vec3(
        (FLAMEGPU->random.uniform<float>() * 2) - 1,
        (FLAMEGPU->random.uniform<float>() * 2) - 1,
        (FLAMEGPU->random.uniform<float>() * 2) - 1);
    const glm::vec3 norm_dir = normalize(dummy_dir);
    location.x += 2 * R_cell * norm_dir.x;
    location.y += 2 * R_cell * norm_dir.y;
    location.z += 2 * R_cell * norm_dir.z;
    return location;
}
/**
 * Common function used by nb and sc cell lifecycle to decide whether a cell is dead
 */
__device__ __forceinline__ bool remove(flamegpu::DeviceAPI<flamegpu::MessageNone, flamegpu::MessageNone>* FLAMEGPU) {
    // Remove an apoptotic or necrotic cell if it is engulfed by an immune cell.
    if (FLAMEGPU->getVariable<int>("apop") == 1 || FLAMEGPU->getVariable<int>("necro") == 1) {
        const float P_lysis = FLAMEGPU->environment.getProperty<float>("P_lysis");
        const unsigned int step_size = FLAMEGPU->environment.getProperty<unsigned int>("step_size");
        if (FLAMEGPU->random.uniform<float>() < P_lysis * step_size)
            return true;
    }
    return false;
}
template<typename Mi, typename Mo>
__device__ __forceinline__ bool getChemoState(flamegpu::DeviceAPI<Mi, Mo>* FLAMEGPU) {
    int chemo = 0;
    const int chemo_number = FLAMEGPU->environment.getProperty<int>("chemo_number");
    const std::array<int, 336> chemo_start = FLAMEGPU.environment.getProperty<int, 336>("chemo_start");
    const std::array<int, 336> chemo_end = FLAMEGPU.environment.getProperty<int, 336>("chemo_end");
    for(int i=0;i<chemo_number;i++)
	{
		if(FLAMEGPU->getStepCounter()>=chemo_start[i] && FLAMEGPU->getStepCounter()<=chemo_end[i])
			{
                                                chemo = 1;
                                                break;
                        }
	}
    if(chemo==1)
	{
		return true;
	}
    else
	{
		return false;
	}
    // return (FLAMEGPU->getStepCounter() % 504) < 24;
}

// Hostfn prototypes
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER vasculature;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER reset_grids;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER alter2;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER CAexpand;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER toggle_chemo;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER host_validation;

#endif  // SRC_HEADER_H_
