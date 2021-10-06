FLAMEGPU_DEVICE_FUNCTION float length(const float &x, const float &y, const float &z) {
    const float rtn = sqrt(x*x + y*y + z*z);

    return rtn;
}
FLAMEGPU_AGENT_FUNCTION(calculate_force, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
    // Load location
    const float i_x = FLAMEGPU->getVariable<float>("x");
    const float i_y = FLAMEGPU->getVariable<float>("y");
    const float i_z = FLAMEGPU->getVariable<float>("z");
    const float i_radius = FLAMEGPU->environment.getProperty<float>("R_cell");  // Eventually this might vary based on the cell
    // unsigned int i_id = FLAMEGPU->getVariable<unsigned int>("id");  // Id mechanics are not currently setup
    // Init force to 0 (old force doesn't matter)
    float i_fx = 0;
    float i_fy = 0;
    float i_fz = 0;
    float i_overlap = 0;
    // Load env once
    const float MIN_OVERLAP = FLAMEGPU->environment.getProperty<float>("min_overlap");
    const float CRIT_OVERLAP = FLAMEGPU->environment.getProperty<float>("crit_overlap");
    const float K1 = FLAMEGPU->environment.getProperty<float>("k1");
    const float ALPHA = FLAMEGPU->environment.getProperty<float>("alpha");
    for (auto j : FLAMEGPU->message_in(i_x, i_y, i_z)) {
        // if (j->getVariable<unsigned int>("id") != i_id)  // Id mechanics are not currently setup
        {
            const float j_x = j.getVariable<float>("x");
            const float j_y = j.getVariable<float>("y");
            const float j_z = j.getVariable<float>("z");
            const float j_radius = FLAMEGPU->environment.getProperty<float>("R_cell");  // Eventually this might vary based on the cell
            // Displacement
            const float ij_x = i_x - j_x;
            const float ij_y = i_y - j_y;
            const float ij_z = i_z - j_z;
            const float distance_ij = length(ij_x, ij_y, ij_z);
            if (distance_ij != 0) {  // This may need to be replaced by ID check later?
                if (distance_ij <= FLAMEGPU->message_in.radius()) {
                    // Displacement
                    const float ij_dx = ij_x / distance_ij;
                    const float ij_dy = ij_y / distance_ij;
                    const float ij_dz = ij_z / distance_ij;
                    float overlap_ij = i_radius + j_radius - distance_ij;
                    float Fij = 0;
                    if (overlap_ij == i_radius + j_radius) {
                        // This case is redundant, should be covered by j != i
                        overlap_ij = 0;
                    } else if (overlap_ij < MIN_OVERLAP) {
                        //float Fij = 0;
                        //force_i += Fij*direction_ij;
                    } else if (overlap_ij < CRIT_OVERLAP) {
                        Fij = K1 * overlap_ij;
                    } else {
                        Fij = K1 * CRIT_OVERLAP * exp(ALPHA * (overlap_ij / CRIT_OVERLAP - 1));
                    }
                    i_fx += Fij * ij_dx;
                    i_fy += Fij * ij_dy;
                    i_fz += Fij * ij_dz;
                    i_overlap += overlap_ij;
                }
            }
        }
    }
    // Set outputs
    FLAMEGPU->setVariable<float>("fx", i_fx);
    FLAMEGPU->setVariable<float>("fy", i_fy);
    FLAMEGPU->setVariable<float>("fz", i_fz);
    FLAMEGPU->setVariable<float>("overlap", i_overlap);
    FLAMEGPU->setVariable<float>("force_magnitude", length(i_fx, i_fy, i_fz));
    return flamegpu::ALIVE;
}