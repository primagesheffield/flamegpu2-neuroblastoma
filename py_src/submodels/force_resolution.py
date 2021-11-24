from pyflamegpu import *
from environment import *

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
    nb1 = nb.newRTCFunctionFile("nb_apply_force", "submodels/fr_apply_force.cu");
    nb2 = nb.newRTCFunctionFile("nb_output_location", "submodels/fr_output_location.cu");
    nb3 = nb.newRTCFunctionFile("nb_calculate_force", "submodels/fr_calculate_force.cu");
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
    sc1 = sc.newRTCFunctionFile("sc_apply_force", "submodels/fr_apply_force.cu");
    sc2 = sc.newRTCFunctionFile("sc_output_location", "submodels/fr_output_location.cu");
    sc3 = sc.newRTCFunctionFile("sc_calculate_force", "submodels/fr_calculate_force.cu");
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

