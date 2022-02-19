#define GRID_MAX_DIMENSIONS 151
#define GMD GRID_MAX_DIMENSIONS
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

FLAMEGPU_AGENT_FUNCTION(output_oxygen_cell, flamegpu::MessageNone, flamegpu::MessageNone) {
    const glm::ivec3 gid = toGrid(FLAMEGPU, FLAMEGPU->getVariable<glm::vec3>("xyz"));
    increment_grid_nb(FLAMEGPU, gid);
    return flamegpu::ALIVE;
}