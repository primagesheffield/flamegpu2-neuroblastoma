import json
from json import JSONEncoder
import numpy

class EnvMini:
    def __init__(self):
        pass

'''
Basic class to allow desired objects 'EnvMini' to be serialised
'''
class NBEncoder(JSONEncoder):
    def default(self, object):
        if isinstance(object, EnvMini):
            return object.__dict__
        elif isinstance(object, numpy.generic): 
            return object.item()
        else:
            # call base class implementation which takes care of
            # raising exceptions for unsupported types
            return json.JSONEncoder.default(self, object)

def save(filePath, environment, seed = None, steps = None):
    #Construct the export structure
    exportState = {}
    exportState['version'] = (13, 9, 0, False)
    exportState['config'] = {}
    
    if environment!=None:
        if not isinstance(environment, EnvMini):
            raise ValueError("environment must be type EnvMini")
        exportState['config']['environment'] = environment
        
    if isinstance(seed, int):
        exportState['config']['seed'] = seed
    elif seed!=None:
        raise ValueError("seed must be type int")
        
    if steps!=None and isinstance(steps, int):
        exportState['config']['steps'] = steps
    elif steps!=None:
        raise ValueError("steps must be type int")
           
    #Perform export
    try:
        outFile = open(filePath,"w") 
        if pretty==True:
            json.dump(exportState, outFile, cls=NBEncoder, indent=4)
        else:
            json.dump(exportState, outFile, cls=NBEncoder)   
        outFile.close()
    except IOError:
        raise IOError("Could not open file '%s' for JSON export."%(filePath))