from pyflamegpu import *

def defineNeuroblastoma(model):
    nb = model.newAgent("Neuroblastoma");
    nb.newVariableFloat("x");
    nb.newVariableFloat("y");
    nb.newVariableFloat("z");
    # Temporary replacement for cell cycle
    nb.newVariableFloat("age");
    return nb;
