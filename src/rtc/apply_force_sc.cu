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
 * Updates the passed location by applying boundary forces to it
 */
__device__ __forceinline__ void boundary_conditions(flamegpu::DeviceAPI<flamegpu::MessageNone, flamegpu::MessageNone>* FLAMEGPU, glm::vec3& location) {
    const glm::vec3 bc_minus = FLAMEGPU->environment.getProperty<glm::vec3>("bc_minus");
    const glm::vec3 bc_plus = FLAMEGPU->environment.getProperty<glm::vec3>("bc_plus");
    const glm::vec3 displace = FLAMEGPU->environment.getProperty<glm::vec3>("displace");
    if (location.x < bc_minus.x) {
        location.x += displace.x * (bc_minus.x - location.x);
    } else if (location.x > bc_plus.x) {
        location.x -= displace.x * (location.x - bc_plus.x);
    }
    if (location.y < bc_minus.y) {
        location.y += displace.y * (bc_minus.y - location.y);
    } else if (location.y > bc_plus.y) {
        location.y -= displace.y * (location.y - bc_plus.y);
    }
    if (location.z < bc_minus.z) {
        location.z += displace.z * (bc_minus.z - location.z);
    } else if (location.z > bc_plus.z) {
        location.z -= displace.z * (location.z - bc_plus.z);
    }
}
FLAMEGPU_AGENT_FUNCTION(apply_force_sc, flamegpu::MessageNone, flamegpu::MessageNone) {
    // Access vectors in self, as vectors
    glm::vec3 location = FLAMEGPU->getVariable<glm::vec3>("xyz");
    const glm::vec3 force = FLAMEGPU->getVariable<glm::vec3>("Fxyz");
    // Decide our voxel
    const glm::ivec3 gid = toGrid(FLAMEGPU, location);
    // Apply Force
    const float mu = FLAMEGPU->environment.getProperty<float>("mu");
    float matrix = FLAMEGPU->environment.getMacroProperty<float, GMD, GMD, GMD>("matrix_grid")[gid.x][gid.y][gid.z];
    matrix = matrix == 0.0f ? FLAMEGPU->environment.getProperty<float>("matrix_dummy") : matrix;
    const float mu_eff = (1.0f + matrix) * mu;
    const float dt = FLAMEGPU->environment.getProperty<float>("dt");
    location += force * dt / mu_eff;
    // Apply boundary conditions
    boundary_conditions(FLAMEGPU, location);

    // Set updated agent variables
    FLAMEGPU->setVariable<glm::vec3>("xyz", location);

    return flamegpu::ALIVE;
}