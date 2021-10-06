// Used by both Neuroblastoma and Schwann
FLAMEGPU_AGENT_FUNCTION(apply_force, flamegpu::MessageNone, flamegpu::MessageNone) {
    //Apply Force (don't bother with dummy, directly clamp inside location)
    const float force_mod = FLAMEGPU->environment.getProperty<float>("dt_computed") / FLAMEGPU->environment.getProperty<float>("mu_eff");
    FLAMEGPU->setVariable<float>("x", FLAMEGPU->getVariable<float>("x") + force_mod * FLAMEGPU->getVariable<float>("fx"));
    FLAMEGPU->setVariable<float>("y", FLAMEGPU->getVariable<float>("y") + force_mod * FLAMEGPU->getVariable<float>("fy"));
    FLAMEGPU->setVariable<float>("z", FLAMEGPU->getVariable<float>("z") + force_mod * FLAMEGPU->getVariable<float>("fz"));
    return flamegpu::ALIVE;
}