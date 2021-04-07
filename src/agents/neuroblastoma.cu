#include "header.h"

AgentDescription& defineNeuroblastoma(ModelDescription& model) {
    auto& nb = model.newAgent("Neuroblastoma");
    nb.newVariable<float>("x");
    nb.newVariable<float>("y");
    nb.newVariable<float>("z");
    // Temporary replacement for cell cycle
    nb.newVariable<float>("age");
    return nb;
}
