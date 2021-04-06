#include "header.h"

AgentDescription &defineSchwann(ModelDescription& model) {
    auto& sc = model.newAgent("Schwann");
    sc.newVariable<float>("x");
    sc.newVariable<float>("y");
    sc.newVariable<float>("z");
    return sc;
}
