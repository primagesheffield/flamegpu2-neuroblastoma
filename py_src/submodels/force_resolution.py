from pyflamegpu import *
from environment import *

apply_force = """
// Used by both Neuroblastoma and Schwann
FLAMEGPU_AGENT_FUNCTION(apply_force, flamegpu::MessageNone, flamegpu::MessageNone) {
    //Apply Force (don't bother with dummy, directly clamp inside location)
    const float force_mod = FLAMEGPU->environment.getProperty<float>("dt_computed") / FLAMEGPU->environment.getProperty<float>("mu_eff");
    FLAMEGPU->setVariable<float>("x", FLAMEGPU->getVariable<float>("x") + force_mod * FLAMEGPU->getVariable<float>("fx"));
    FLAMEGPU->setVariable<float>("y", FLAMEGPU->getVariable<float>("y") + force_mod * FLAMEGPU->getVariable<float>("fy"));
    FLAMEGPU->setVariable<float>("z", FLAMEGPU->getVariable<float>("z") + force_mod * FLAMEGPU->getVariable<float>("fz"));
    return flamegpu::ALIVE;
}
"""
output_location = """
// Used by both Neuroblastoma and Schwann
FLAMEGPU_AGENT_FUNCTION(output_location, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
    FLAMEGPU->message_out.setLocation(
        FLAMEGPU->getVariable<float>("x"),
        FLAMEGPU->getVariable<float>("y"),
        FLAMEGPU->getVariable<float>("z"));
    FLAMEGPU->message_out.setVariable<unsigned int>("id", 0);  // Currently unused, FGPU2 will have auto agent ids in future
    return flamegpu::ALIVE;
}
"""
calculate_force = """
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
"""

class calculate_convergence(pyflamegpu.HostFunctionConditionCallback):
    def __init__(self):
        super().__init__();  # Mandatory if we are defining __init__ ourselves
        # Local, so value is maintained between calls to calculate_convergence::run
        self.total_force = 0;

    def run(self, FLAMEGPU):
        # Reduce force and overlap
        dummy_force = FLAMEGPU.agent("Neuroblastoma").sumFloat("force_magnitude") + FLAMEGPU.agent("Schwann").sumFloat("force_magnitude");
        max_force = max(FLAMEGPU.agent("Neuroblastoma").maxFloat("force_magnitude"), FLAMEGPU.agent("Schwann").maxFloat("force_magnitude"));
        R_CELL = FLAMEGPU.environment.getPropertyFloat("R_cell");
        MU_EFF = FLAMEGPU.environment.getPropertyFloat("mu_eff");
        DT_MAX = FLAMEGPU.environment.getPropertyFloat("dt_max");
        if max_force > 0:  # Div 0 error here
            new_dt = min((R_CELL / max_force) * MU_EFF, DT_MAX);
            FLAMEGPU.environment.setPropertyFloat("dt_computed", new_dt);
        # Unused currently
        # float dummy_overlap = FLAMEGPU->agent("Neuroblastoma").sum<float>("overlap");

        if (FLAMEGPU.getStepCounter() + 1 < FLAMEGPU.environment.getPropertyUInt("min_force_resolution_steps")):
            # Force resolution must always run at least 2 steps, maybe more
            # First pass step counter == 0, 2nd == 1 etc
            self.total_force = dummy_force;
            return pyflamegpu.CONTINUE;
        elif(FLAMEGPU.agent("Neuroblastoma").count() + FLAMEGPU.agent("Schwann").count() < 2):
            return pyflamegpu.EXIT;
        elif(abs(dummy_force) < 2e-07):
            return pyflamegpu.EXIT;
        elif(100 * abs((self.total_force - dummy_force) / self.total_force) < FLAMEGPU.environment.getPropertyFloat("thres_converge")):
            return pyflamegpu.EXIT;
        else:
            self.total_force = dummy_force;
            return pyflamegpu.CONTINUE;

"""
   Define the force resolution submodel
"""
def defineForceResolution(model):
    force_resolution = pyflamegpu.ModelDescription("force resolution");
    # Setup the mechanical model env properties
    mechanical_model_parameters(force_resolution);
    env = force_resolution.Environment();
    # Defined elsewhere    
    env.newPropertyFloat("R_cell", 11);
    # Internal/derived
    env.newPropertyFloat("mu_eff", 0.4);
    env.newPropertyFloat("dt_computed", 36);
    env.newPropertyUInt("min_force_resolution_steps", 2);

    loc = force_resolution.newMessageSpatial3D("Location");
    loc.setMin(-1000, -1000, -1000);
    loc.setMax(1000, 1000, 1000);
    loc.setRadius(20);
    # These are auto defined for Spatial3D messages
    # loc.newVariableFloat("x");
    # loc.newVariableFloat("y");
    # loc.newVariableFloat("z");
    loc.newVariableUInt("id");
    nb = force_resolution.newAgent("Neuroblastoma");
    nb.newVariableFloat("x");
    nb.newVariableFloat("y");
    nb.newVariableFloat("z");
    nb.newVariableFloat("fx", 0);
    nb.newVariableFloat("fy", 0);
    nb.newVariableFloat("fz", 0);
    nb.newVariableFloat("overlap");
    nb.newVariableFloat("force_magnitude");
    nb1 = nb.newRTCFunction("nb_apply_force", apply_force);
    nb2 = nb.newRTCFunction("nb_output_location", output_location);
    nb3 = nb.newRTCFunction("nb_calculate_force", calculate_force);
    nb2.setMessageOutput(loc);
    nb3.setMessageInput(loc);
    sc = force_resolution.newAgent("Schwann");
    sc.newVariableFloat("x");
    sc.newVariableFloat("y");
    sc.newVariableFloat("z");
    sc.newVariableFloat("fx", 0);
    sc.newVariableFloat("fy", 0);
    sc.newVariableFloat("fz", 0);
    sc.newVariableFloat("overlap");
    sc.newVariableFloat("force_magnitude");
    sc1 = sc.newRTCFunction("sc_apply_force", apply_force);
    sc2 = sc.newRTCFunction("sc_output_location", output_location);
    sc3 = sc.newRTCFunction("sc_calculate_force", calculate_force);
    sc2.setMessageOutput(loc);
    sc3.setMessageInput(loc);

    # Apply force
    l1 = force_resolution.newLayer();
    l1.addAgentFunction(nb1);
    l1.addAgentFunction(sc1);
    # Output location nb
    l2 = force_resolution.newLayer();
    l2.addAgentFunction(nb2);
    # Output location sc
    l3 = force_resolution.newLayer();
    l3.addAgentFunction(sc2);
    # Calculate force
    l4 = force_resolution.newLayer();
    l4.addAgentFunction(nb3);
    l4.addAgentFunction(sc3);
    force_resolution.addExitConditionCallback(calculate_convergence().__disown__());

    smd = model.newSubModel("force resolution", force_resolution);
    smd.bindAgent("Neuroblastoma", "Neuroblastoma", True);
    smd.bindAgent("Schwann", "Schwann", True);
    smd.SubEnvironment(True);

    return smd;

