import os
import sys

def add_module_path():
    module_path = os.path.abspath(os.path.join(os.pardir, os.pardir))
    print('added path:', module_path)
    if module_path not in sys.path:
        sys.path.append(module_path)