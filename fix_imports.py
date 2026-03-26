import os
import re
import glob

def get_relative_import(file_path, abs_import_module):
    # Determine how many levels deep we are from the root
    # e.g., gpu_kernels/generation/integrator.py -> 2 levels deep
    # root is the directory containing gpu_kernels
    
    parts = os.path.normpath(file_path).split(os.sep)
    # The root is where the script is run from (the repo root)
    # We want to find the relative path from the current file's directory to the target module.
    # The absolute import is like 'gpu_kernels.schemas'.
    
    # Simple heuristic: count directories above the file up to the repo root.
    # We know the repo root contains 'bandit', 'gpu_kernels', etc.
    # So if file is 'gpu_kernels/generation/integrator.py', it's 2 levels deep.
    # Rel import for 'gpu_kernels.schemas' would be '..schemas'
    # Rel import for 'model_scientist.schemas' would be '...model_scientist.schemas'
    
    rel_depth = len(file_path.split('/')) - 1
    dots = '.' * (rel_depth + 1)
    
    # If the target is in the SAME top-level package, e.g., 'gpu_kernels.schemas' from 'gpu_kernels/generation/...'
    top_level_pkg = file_path.split('/')[0]
    target_top_level = abs_import_module.split('.')[0]
    
    if top_level_pkg == target_top_level:
        # Same top level package.
        # e.g., from gpu_kernels.schemas -> from ..schemas
        sub_path = abs_import_module[len(top_level_pkg)+1:]
        dots = '.' * rel_depth
        if sub_path:
            return f"{dots}{sub_path}"
        else:
            return dots
    else:
        # Cross package, e.g., from model_scientist...
        # Just return the absolute import? The spec says "Convert all to relative imports"
        return f"{dots}{abs_import_module}"

for root, _, files in os.walk('.'):
    if '.git' in root or '__pycache__' in root:
        continue
    for file in files:
        if file.endswith('.py'):
            path = os.path.join(root, file).replace('\\', '/').removeprefix('./')
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if 'sys.path.insert' in content:
                # Find the block:
                # import sys
                # sys.path.insert(...)
                # from X import Y
                
                # We'll use regex to find sys.path.insert and the following from import
                pattern = r'(?:import sys\n(?:import os\n)?)?sys\.path\.insert\(0,[^\)]+\)\n(?:from ([a-zA-Z0-9_.]+) import ([a-zA-Z0-9_, ]+))'
                
                def repl(match):
                    module = match.group(1)
                    names = match.group(2)
                    rel_mod = get_relative_import(path, module)
                    return f"from {rel_mod} import {names}"
                
                new_content = re.sub(pattern, repl, content)
                
                # Also handle cases where import sys is separate
                new_content = re.sub(r'import sys\n(?:import os\n)?sys\.path\.insert\(0,[^\)]+\)\n', '', new_content)
                new_content = re.sub(r'sys\.path\.insert\(0,[^\)]+\)\n', '', new_content)
                
                if content != new_content:
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print(f"Fixed imports in {path}")
