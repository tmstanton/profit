# imports
import os
from . import utils
import profit # type: ignore

# set up class for storing lines
class Spectral_Line(object):

    def __init__(self:object, id:str, type:str, wavelengths:list, title:str, components:list, emission:bool) -> None:
        
        self.id         = id
        self.title      = title
        self.type       = type
        self.wls        = wavelengths
        self.components = components
        self.emission   = emission

        # set nprofiles
        self.nprofiles = len(self.components)

# methods
def Generate_Features() -> dict:
    """ Generates dictionary of line objects based on the contents of the lines directory """

    # source features
    linepath = os.path.abspath(profit.__file__).strip('__init__.py') + 'lines'
    lines = {_file.strip('.pkl'): utils.PklLoad(f'{linepath}/{_file}') for _file in os.listdir(linepath)}

    # return dictionary of objects
    return {_: Spectral_Line(id=_line['id'], 
                             type=_line['type'], 
                             wavelengths=_line['wavelengths'], 
                             title=_line['title'], 
                             components=_line['components'],
                             emission=_line['emission']) 
                             for _, _line in lines.items()}
        
def Add_Line(id:str, wavelengths:list, title:str, type:str, components:list, emission:bool=True) -> None:
    """ Adds a feature to the lines directory """
    line = {l1: l2 for l1, l2 in zip(['wavelengths', 'id', 'title', 'type', 'components', 'emission'], [wavelengths, id, title, type, components, emission])}
    linepath = os.path.abspath(profit.__file__).strip('__init__.py') + 'lines'
    utils.PklSave(f'{linepath}/{line["id"]}.pkl', line)

