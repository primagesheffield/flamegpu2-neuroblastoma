template<typename M1, typename M2>
__forceinline__ __device__ float calc_R(flamegpu::DeviceAPI<M1, M2>*FLAMEGPU) {
    // The math here isn't as dynamic as in the python model
    // Cycle is tracked differently in each version, which makes tracking the dynamic maths more expensive here
    // Cycle stages haven't changed since the model was first created, so we're probably safe.
    const unsigned int cycle = FLAMEGPU->template getVariable<unsigned int>("cycle");
    const glm::uvec4 cycle_stages = FLAMEGPU->environment.template getProperty<glm::uvec4>("cycle_stages");
    float Ri;
    if (cycle < cycle_stages[0])
        // Ri = <0-1> * 12 / (12 + 4) = <0-0.75>
        Ri = cycle * 0.75f / 12.0f;
    else if (cycle < cycle_stages[1])
        // Ri = 12 / (12 + 4) = 0.75
        Ri = 0.75f;
    else if (cycle < cycle_stages[2])
        // Ri = (12 + (<2-3> - 2) * 4) / (12 + 4) = <12-16> / 16  = <0.75-0.1>
        Ri = 0.75f + (0.25f * (cycle - 18) / (22.0f - 18.0f));
    else
        // Ri = (12 + 4) / (12 + 4) = 1
        Ri = 1.0f;
    return Ri;
}
FLAMEGPU_AGENT_FUNCTION(calculate_force, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
    // Load location
    const glm::vec3 i_xyz = FLAMEGPU->getVariable<glm::vec3>("xyz");
    const flamegpu::id_t i_id = FLAMEGPU->getID();
    const float Ri = calc_R(FLAMEGPU);
    const float R_cell = FLAMEGPU->environment.getProperty<float>("R_cell");
    // Init force to 0 (old force doesn't matter)
    glm::vec3 i_Fxyz = glm::vec3(0);
    float i_overlap = 0;
    int i_neighbours = 1;  // We wouldn't normally count ourself, so begin at 1
    // Load env once
    const float MIN_OVERLAP = FLAMEGPU->environment.getProperty<float>("min_overlap");
    const float K1 = FLAMEGPU->environment.getProperty<float>("k1");
    const float R_NEIGHBOURS = FLAMEGPU->environment.getProperty<float>("R_neighbours");
    for (auto j : FLAMEGPU->message_in(i_xyz.x, i_xyz.y, i_xyz.z)) {
        if (j.getVariable<flamegpu::id_t>("id") != i_id) {
            glm::vec3 j_xyz = glm::vec3(
                j.getVariable<float>("x"),
                j.getVariable<float>("y"),
                j.getVariable<float>("z"));
            const float Rj = j.getVariable<float>("Rj");
            // Displacement
            const glm::vec3 ij_xyz = i_xyz - j_xyz;
            const float distance_ij = glm::length(ij_xyz);
            if (distance_ij < R_NEIGHBOURS)
                i_neighbours++;
            if (distance_ij <= FLAMEGPU->message_in.radius()) {
                const glm::vec3 direction_ij = ij_xyz / distance_ij;
                const float overlap_ij = 2 * R_cell + (Ri + Rj) * R_cell - distance_ij;
                if (overlap_ij == 2 * R_cell + (Ri + Rj) * R_cell) {
                    // overlap_ij = 0;
                    // force_i = glm::vec3(0);
                } else if (overlap_ij < MIN_OVERLAP) {
                    // float Fij = 0;
                    // force_i += Fij*direction_ij;
                } else {
                    const float Fij = K1 * overlap_ij;
                    i_Fxyz += Fij * direction_ij;
                    i_overlap += overlap_ij;
                }
            }
        }
    }
    if (i_neighbours > FLAMEGPU->environment.getProperty<int>("N_neighbours") && FLAMEGPU->getVariable<int>("mobile") == 1) {
        i_Fxyz *= FLAMEGPU->environment.getProperty<float>("k_locom");
    }
    // Set outputs
    FLAMEGPU->setVariable<glm::vec3>("Fxyz", i_Fxyz);
    FLAMEGPU->setVariable<float>("overlap", i_overlap);
    FLAMEGPU->setVariable<float>("force_magnitude", length(i_Fxyz));
    FLAMEGPU->setVariable<int>("neighbours", i_neighbours);
    return flamegpu::ALIVE;
}