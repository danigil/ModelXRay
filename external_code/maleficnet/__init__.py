import os,sys

def add_module_path():
    module_path = os.path.dirname(os.path.realpath(__file__))
    print('added path:', module_path)
    if module_path not in sys.path:
        sys.path.append(module_path)

add_module_path()