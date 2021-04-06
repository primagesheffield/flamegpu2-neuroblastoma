#include "header.h"

// Used by both Neuroblastoma and Schwann
FLAMEGPU_AGENT_FUNCTION(apply_force, MsgNone, MsgNone) {
    //Apply Force (don't bother with dummy, directly clamp inside location)
    const float force_mod = FLAMEGPU->environment.getProperty<float>("dt_computed") / FLAMEGPU->environment.getProperty<float>("mu_eff");
    FLAMEGPU->setVariable<float>("x", FLAMEGPU->getVariable<float>("x") + force_mod * FLAMEGPU->getVariable<float>("fx"));
    FLAMEGPU->setVariable<float>("y", FLAMEGPU->getVariable<float>("y") + force_mod * FLAMEGPU->getVariable<float>("fy"));
    FLAMEGPU->setVariable<float>("z", FLAMEGPU->getVariable<float>("z") + force_mod * FLAMEGPU->getVariable<float>("fz"));
    return ALIVE;
}
// Used by both Neuroblastoma and Schwann
FLAMEGPU_AGENT_FUNCTION(output_location, MsgNone, MsgSpatial3D) {
    FLAMEGPU->message_out.setLocation(
        FLAMEGPU->getVariable<float>("x"),
        FLAMEGPU->getVariable<float>("y"),
        FLAMEGPU->getVariable<float>("z"));
    FLAMEGPU->message_out.setVariable<unsigned int>("id", 0);  // Currently unused, FGPU2 will have auto agent ids in future
    return ALIVE;
}
FLAMEGPU_DEVICE_FUNCTION float length(const float &x, const float &y, const float &z) {
    const float rtn = sqrt(x*x + y*y + z*z);
#if !defined(SEATBELTS) || SEATBELTS
    if (isnan(rtn)) {
        DTHROW("length(%f, %f, %f) == NaN\n", x, y, z);
    }
#endif
    return rtn;
}
FLAMEGPU_AGENT_FUNCTION(calculate_force, MsgSpatial3D, MsgNone) {
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
    return ALIVE;
}
FLAMEGPU_CUSTOM_REDUCTION(Max, a, b) {
    return a > b ? a : b;
}
FLAMEGPU_EXIT_CONDITION(calculate_convergence) {
    //Local static, so value is maintained between function calls
    static float total_force = 0;

    //Reduce force and overlap
    const float dummy_force = FLAMEGPU->agent("Neuroblastoma").sum<float>("force_magnitude") + FLAMEGPU->agent("Schwann").sum<float>("force_magnitude");
    const float max_force = std::max<float>(FLAMEGPU->agent("Neuroblastoma").reduce<float>("force_magnitude", Max, 0), FLAMEGPU->agent("Schwann").reduce<float>("force_magnitude", Max, 0));
    const float R_CELL = FLAMEGPU->environment.getProperty<float>("R_cell");
    const float MU_EFF = FLAMEGPU->environment.getProperty<float>("mu_eff");
    const float DT_MAX = FLAMEGPU->environment.getProperty<float>("dt_max");
    const float new_dt = std::min<float>((R_CELL / max_force) * MU_EFF, DT_MAX);
    FLAMEGPU->environment.setProperty<float>("dt_computed", new_dt);
    //Unused currently
    //float dummy_overlap = FLAMEGPU->agent("Neuroblastoma").sum<float>("overlap");

    if (FLAMEGPU->getStepCounter() + 1 < FLAMEGPU->environment.getProperty<unsigned int>("min_force_resolution_steps")) {
        // Force resolution must always run at least 2 steps, maybe more
        // First pass step counter == 0, 2nd == 1 etc
        total_force = dummy_force;
        return CONTINUE;
    } else if(FLAMEGPU->agent("Neuroblastoma").count() + FLAMEGPU->agent("Schwann").count() < 2) {
        return EXIT;
    } else if (abs(dummy_force) < 2e-07) {
        return EXIT;
    } else if (100 * abs((total_force - dummy_force) / total_force) < FLAMEGPU->environment.getProperty<float>("thres_converge")) {
        return EXIT;
    } else {
        total_force = dummy_force;
        return CONTINUE;
    }
}

/**
 * Define the force resolution submodel
 */
SubModelDescription& defineForceResolution(ModelDescription& model) {
    ModelDescription force_resolution("force resolution");

    auto &env = force_resolution.Environment();
    env.newProperty<float>("R_cell", 11);
    env.newProperty<float>("min_overlap", (float)(-4e-6 * 1e6 * 0));
    env.newProperty<float>("crit_overlap", (float)(2e-6 * 1e6));
    env.newProperty<float>("k1", (float)(2.2e-3));
    env.newProperty<float>("alpha", 1);
    env.newProperty<float>("dt_max", 36);
    env.newProperty<float>("thres_converge", 10);
    // Internal/derived
    env.newProperty<float>("mu_eff", 0.4f);
    env.newProperty<float>("dt_computed", 36);
    env.newProperty<unsigned int>("min_force_resolution_steps", 2);

    auto &loc = force_resolution.newMessage<MsgSpatial3D>("Location");
    loc.setMin(-1000, -1000, -1000);
    loc.setMax(1000, 1000, 1000);
    loc.setRadius(20);
    // loc.newVariable<float>("x");
    // loc.newVariable<float>("y");
    // loc.newVariable<float>("z");
    loc.newVariable<unsigned int>("id");
    auto &nb = force_resolution.newAgent("Neuroblastoma");
    nb.newVariable<float>("x");
    nb.newVariable<float>("y");
    nb.newVariable<float>("z");
    nb.newVariable<float>("fx", 0);
    nb.newVariable<float>("fy", 0);
    nb.newVariable<float>("fz", 0);
    nb.newVariable<float>("overlap");
    nb.newVariable<float>("force_magnitude");
    auto &nb1 = nb.newFunction("nb_apply_force", apply_force);
    auto &nb2 = nb.newFunction("nb_output_location", output_location);
    auto &nb3 = nb.newFunction("nb_calculate_force", calculate_force);
    nb2.setMessageOutput(loc);
    nb3.setMessageInput(loc);
    auto &sc = force_resolution.newAgent("Schwann");
    sc.newVariable<float>("x");
    sc.newVariable<float>("y");
    sc.newVariable<float>("z");
    sc.newVariable<float>("fx", 0);
    sc.newVariable<float>("fy", 0);
    sc.newVariable<float>("fz", 0);
    sc.newVariable<float>("overlap");
    sc.newVariable<float>("force_magnitude");
    auto &sc1 = sc.newFunction("sc_apply_force", apply_force);
    auto &sc2 = sc.newFunction("sc_output_location", output_location);
    auto &sc3 = sc.newFunction("sc_calculate_force", calculate_force);
    sc2.setMessageOutput(loc);
    sc3.setMessageInput(loc);

    // Apply force
    auto &l1 = force_resolution.newLayer();
    l1.addAgentFunction(nb1);
    l1.addAgentFunction(sc1);
    // Output location nb
    auto &l2 = force_resolution.newLayer();
    l2.addAgentFunction(nb2);
    // Output location sc
    auto &l3 = force_resolution.newLayer();
    l3.addAgentFunction(sc2);
    // Calculate force
    auto &l4 = force_resolution.newLayer();
    l4.addAgentFunction(nb3);
    l4.addAgentFunction(sc3);
    force_resolution.addExitCondition(calculate_convergence);

    SubModelDescription& smd = model.newSubModel("force resolution", force_resolution);
    smd.bindAgent("Neuroblastoma", "Neuroblastoma", true);
    smd.bindAgent("Schwann", "Schwann", true);
    smd.SubEnvironment(true);

    return smd;
}
