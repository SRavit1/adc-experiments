import yaml

# https://www.geeksforgeeks.org/how-to-change-a-dictionary-into-a-class/
class Dict2Class(object):
    def __init__(self, my_dict):
        for key in my_dict: 
            setattr(self, key, my_dict[key])

def load_config():
    with open("config.yaml") as stream:
        args = Dict2Class(yaml.safe_load(stream))
    return args