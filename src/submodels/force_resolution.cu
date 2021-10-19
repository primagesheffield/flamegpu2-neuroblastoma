#include "header.h"


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
FLAMEGPU_AGENT_FUNCTION(apply_force_nb, flamegpu::MessageNone, flamegpu::MessageNone) {
    // Access vectors in self, as vectors
    glm::vec3 location = FLAMEGPU->getVariable<glm::vec3>("xyz");
    const glm::vec3 force = FLAMEGPU->getVariable<glm::vec3>("Fxyz");
    // Decide our voxel
    const glm::ivec3 gid = toGrid(FLAMEGPU, location);
    // Apply Force
    const float mu = FLAMEGPU->environment.getProperty<float>("mu");
    const float matrix = FLAMEGPU->environment.getMacroProperty<float, GMD, GMD, GMD>("matrix_grid")[gid.x][gid.y][gid.z];
    const float mu_eff = (1.0f + matrix) * mu;
    const float dt = FLAMEGPU->environment.getProperty<float>("dt");
    location += force * dt / mu_eff;
    // Apply boundary conditions
    boundary_conditions(FLAMEGPU, location);
    // Begin CAexpand with new data
    increment_grid_nb(FLAMEGPU, gid);

    // Set updated agent variables
    FLAMEGPU->setVariable<glm::vec3>("xyz", location);

    // VALIDATION: Calculate distance moved
    const glm::vec3 old_location = FLAMEGPU->getVariable<glm::vec3>("old_xyz");
    FLAMEGPU->setVariable("move_dist", glm::distance(location, old_location));

    return flamegpu::ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(apply_force_sc, flamegpu::MessageNone, flamegpu::MessageNone) {
    // Access vectors in self, as vectors
    glm::vec3 location = FLAMEGPU->getVariable<glm::vec3>("xyz");
    const glm::vec3 force = FLAMEGPU->getVariable<glm::vec3>("Fxyz");
    // Decide our voxel
    const glm::ivec3 gid = toGrid(FLAMEGPU, location);
    // Apply Force
    const float mu = FLAMEGPU->environment.getProperty<float>("mu");
    const float matrix = FLAMEGPU->environment.getMacroProperty<float, GMD, GMD, GMD>("matrix_grid")[gid.x][gid.y][gid.z];
    const float mu_eff = (1.0f + matrix) * mu;
    const float dt = FLAMEGPU->environment.getProperty<float>("dt");
    location += force * dt / mu_eff;
    // Apply boundary conditions
    boundary_conditions(FLAMEGPU, location);
    // Begin CAexpand with new data
    increment_grid_sc(FLAMEGPU, gid);

    // Set updated agent variables
    FLAMEGPU->setVariable<glm::vec3>("xyz", location);

    return flamegpu::ALIVE;
}
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
FLAMEGPU_AGENT_FUNCTION(output_location_nb, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
   const glm::vec3 loc = FLAMEGPU->getVariable<glm::vec3>("xyz");
    FLAMEGPU->message_out.setLocation(loc.x, loc.y, loc.z);
    FLAMEGPU->message_out.setVariable<flamegpu::id_t>("id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setVariable<float>("Rj", calc_R(FLAMEGPU));

    // VALIDATION: Set old location on it 0
    if (FLAMEGPU->environment.getProperty<unsigned int>("force_resolution_steps") == 0) {
        FLAMEGPU->setVariable<glm::vec3>("old_xyz", loc);
    }
    return flamegpu::ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(output_location, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
    const glm::vec3 loc = FLAMEGPU->getVariable<glm::vec3>("xyz");
    FLAMEGPU->message_out.setLocation(loc.x, loc.y, loc.z);
    FLAMEGPU->message_out.setVariable<flamegpu::id_t>("id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setVariable<float>("Rj", calc_R(FLAMEGPU));

    return flamegpu::ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(calculate_force, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
    // Load location
    const glm::vec3 i_xyz = FLAMEGPU->getVariable<glm::vec3>("xyz");
    const flamegpu::id_t i_id = FLAMEGPU->getID();
    const float Ri = calc_R(FLAMEGPU);
    const float R_cell = FLAMEGPU->environment.getProperty<float>("R_cell");  // Eventually this might vary based on the cell
    // unsigned int i_id = FLAMEGPU->getVariable<unsigned int>("id");  // Id mechanics are not currently setup
    // Init force to 0 (old force doesn't matter)
    glm::vec3 i_Fxyz = glm::vec3(0);
    float i_overlap = 0;
    int i_neighbours = 1;  // We wouldn't normally count ourself, so begin at 1
    // Load env once
    const float MIN_OVERLAP = FLAMEGPU->environment.getProperty<float>("min_overlap");
    const float K1 = FLAMEGPU->environment.getProperty<float>("k1");
    for (auto j : FLAMEGPU->message_in(i_xyz.x, i_xyz.y, i_xyz.z)) {
        if (j.getVariable<unsigned int>("id") != i_id) {  // Id mechanics are not currently setup
            glm::vec3 j_xyz = glm::vec3(
                j.getVariable<float>("x"),
                j.getVariable<float>("y"),
                j.getVariable<float>("z"));
            const float Rj = j.getVariable<float>("Rj");
            // Displacement
            const glm::vec3 ij_xyz = i_xyz - j_xyz;
            const float distance_ij = glm::length(ij_xyz);
            if (distance_ij < FLAMEGPU->environment.getProperty<unsigned int>("R_neighbours"))
                i_neighbours++;
            if (distance_ij <= FLAMEGPU->message_in.radius()) {
                const glm::vec3 direction_ij = ij_xyz / distance_ij;
                const float overlap_ij = 2 * R_cell + (Ri + Rj) * R_cell - distance_ij;
                if (overlap_ij == 2 * R_cell + (Ri + Rj) * R_cell) {
                    // overlap_ij = 0;
                    // force_i = glm::vec3(0);
                } else if (overlap_ij < MIN_OVERLAP) {
                    //float Fij = 0;
                    //force_i += Fij*direction_ij;
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
FLAMEGPU_CUSTOM_REDUCTION(Max, a, b) {
    return a > b ? a : b;
}
FLAMEGPU_EXIT_CONDITION(calculate_convergence) {
    //Reduce overlap
    const int max_neighbours = glm::max(FLAMEGPU->agent("Neuroblastoma").max<int>("neighbours"), FLAMEGPU->agent("Schwann").max<int>("neighbours"));
    const float max_overlap = glm::max(FLAMEGPU->agent("Neuroblastoma").max<float>("overlap"), FLAMEGPU->agent("Schwann").max<float>("overlap"));
    //Unused currently
    //float dummy_overlap = reduce_Neuroblastoma_default_overlap_variable();

    const int N_neighbours = FLAMEGPU->environment.getProperty<int>("N_neighbours");
    const float R_cell = FLAMEGPU->environment.getProperty<float>("R_cell");
    const float dt = FLAMEGPU->environment.getProperty<float>("dt");
    const unsigned int step_size = FLAMEGPU->environment.getProperty<unsigned int>("step_size");

    if (FLAMEGPU->getStepCounter() + 1 < FLAMEGPU->environment.getProperty<unsigned int>("min_force_resolution_steps")) {
        // Force resolution must always run at least 2 steps, maybe more
        // First pass step counter == 0, 2nd == 1 etc
        return flamegpu::CONTINUE;
    } else if ((FLAMEGPU->agent("Neuroblastoma").count() + FLAMEGPU->agent("Schwann").count() < 2) ||
        (max_neighbours <= N_neighbours) ||
        (max_overlap < 0.15f * R_cell) ||
        (static_cast<float>(FLAMEGPU->getStepCounter()) > static_cast<float>(step_size) * 3600.0f / dt)) {
        return flamegpu::EXIT;
    } else {
        // Force resolution is stuck, log details to stderr
        if ((FLAMEGPU->getStepCounter() >= 1000 && FLAMEGPU->getStepCounter() < 1100) ||
            (FLAMEGPU->getStepCounter() >= 2000 && FLAMEGPU->getStepCounter() < 2200)) {
            fprintf(stderr, "Force Resolution Stuck@FR Step %u\n", FLAMEGPU->getStepCounter());
            fprintf(stderr, "(%u + %u < 2) == %s\n", FLAMEGPU->agent("Neuroblastoma").count(), FLAMEGPU->agent("Schwann").count(), (FLAMEGPU->agent("Neuroblastoma").count() + FLAMEGPU->agent("Schwann").count() < 2) ? "true" : "false");
            fprintf(stderr, "(%d < %d) == %s\n", max_neighbours, N_neighbours, (max_neighbours < N_neighbours) ? "true" : "false");
            fprintf(stderr, "(%f < %f) == %s\n", max_overlap, 0.15f * R_cell, (max_overlap < 0.15f *R_cell) ? "true" : "false");
            fprintf(stderr, "(%u > %u * 3600 /  %f) == %s\n", FLAMEGPU->getStepCounter(), step_size, dt, (static_cast<float>(FLAMEGPU->getStepCounter()) > static_cast<float>(step_size) * 3600.0f / dt) ? "true" : "false");
        }
        return flamegpu::CONTINUE;
    }
}

/**
 * Define the force resolution submodel
 */
flamegpu::SubModelDescription& defineForceResolution(flamegpu::ModelDescription& model) {
    flamegpu::ModelDescription force_resolution("force resolution");

    auto &env = force_resolution.Environment();
    env.newProperty<float>("R_voxel", 0);
    env.newProperty<unsigned int, 3>("grid_dims", {0, 0, 0});
    env.newProperty<float>("dt_computed", 0);
    env.newProperty<unsigned int, 3>("grid_origin", { 0, 0, 0 });
    env.newProperty<float, 3>("bc_minus", { 0, 0, 0 });
    env.newProperty<float, 3>("bc_plus", { 0, 0, 0 });
    env.newProperty<float, 3>("displace", { 0, 0, 0 });
    env.newProperty<float>("mu", 0.0f);
    env.newProperty<float>("dt", 0.0f);
    env.newProperty<unsigned int, 4>("cycle_stages", { 0, 0, 0, 0 });
    env.newProperty<unsigned int>("force_resolution_steps", 0);
    env.newProperty<float>("R_cell", 0);
    env.newProperty<float>("min_overlap", 0);
    env.newProperty<float>("k1", 0);
    env.newProperty<float>("alpha", 0);
    env.newProperty<unsigned int>("R_neighbours", 0);
    env.newProperty<int>("N_neighbours", 0);
    env.newProperty<float>("k_locom", 0);
    env.newProperty<unsigned int>("step_size", 0);
    env.newProperty<unsigned int>("min_force_resolution_steps", 0);

    model.Environment().newMacroProperty<unsigned int, GMD, GMD, GMD>("Nnb_grid");
    model.Environment().newMacroProperty<unsigned int, GMD, GMD, GMD>("Nsc_grid");
    // model.Environment().newMacroProperty<unsigned int, GMD, GMD, GMD>("Nsca_grid");  // Location of all SC cells with apop == 1
    // model.Environment().newMacroProperty<unsigned int, GMD, GMD, GMD>("Nnba_grid");  // Location of all NB cells with apop == 1
    model.Environment().newMacroProperty<unsigned int, GMD, GMD, GMD>("Nnbn_grid");
    model.Environment().newMacroProperty<unsigned int, GMD, GMD, GMD>("Nscn_grid");
    model.Environment().newMacroProperty<unsigned int, GMD, GMD, GMD>("Nnbl_grid");
    model.Environment().newMacroProperty<unsigned int, GMD, GMD, GMD>("Nscl_grid");
    model.Environment().newMacroProperty<unsigned int, GMD, GMD, GMD>("Nscl_col_grid");
    model.Environment().newMacroProperty<float, GMD, GMD, GMD>("matrix_grid");

    auto &loc = force_resolution.newMessage<flamegpu::MessageSpatial3D>("Location");
    loc.setMin(-2000, -2000, -2000);
    loc.setMax(2000, 2000, 2000);
    loc.setRadius(25);
    // loc.newVariable<float>("x");
    // loc.newVariable<float>("y");
    // loc.newVariable<float>("z");
    loc.newVariable<float>("Rj");
    loc.newVariable<unsigned int>("id");
    auto &nb = force_resolution.newAgent("Neuroblastoma");
    nb.newVariable<float, 3>("xyz");
    nb.newVariable<float, 3>("Fxyz");
    nb.newVariable<int>("neighbours");
    nb.newVariable<float>("overlap");
    nb.newVariable<float>("force_magnitude");
    nb.newVariable<float>("move_dist");
    nb.newVariable<float, 3>("old_xyz");
    auto &nb1 = nb.newFunction("apply_force_nb", apply_force_nb);  // Version with added validation
    auto &nb2 = nb.newFunction("output_location_nb", output_location_nb);  // Version with added validation
    auto &nb3 = nb.newFunction("calculate_force_nb", calculate_force);
    nb2.setMessageOutput(loc);
    nb3.setMessageInput(loc);
    auto &sc = force_resolution.newAgent("Schwann");
    sc.newVariable<float, 3>("xyz");
    sc.newVariable<float, 3>("Fxyz");
    sc.newVariable<int>("neighbours");
    sc.newVariable<float>("overlap");
    sc.newVariable<float>("force_magnitude");
    auto &sc1 = sc.newFunction("apply_force_sc", apply_force_sc);
    auto &sc2 = sc.newFunction("output_location_sc", output_location);
    auto &sc3 = sc.newFunction("calculate_force_sc", calculate_force);
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

    flamegpu::SubModelDescription& smd = model.newSubModel("force resolution", force_resolution);
    smd.bindAgent("Neuroblastoma", "Neuroblastoma", true);
    smd.bindAgent("Schwann", "Schwann", true);
    smd.SubEnvironment(true);

    return smd;
}
