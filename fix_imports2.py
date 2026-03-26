import os

def get_relative_import(file_path, abs_import_module):
    rel_depth = len(file_path.split('/')) - 1
    
    top_level_pkg = file_path.split('/')[0]
    target_top_level = abs_import_module.split('.')[0]
    
    if top_level_pkg == target_top_level:
        sub_path = abs_import_module[len(top_level_pkg)+1:]
        dots = '.' * rel_depth
        if sub_path:
            return f"{dots}{sub_path}"
        else:
            return dots
    else:
        # Cross package, go up to root
        dots = '.' * (rel_depth + 1)
        return f"{dots}{abs_import_module}"

import glob
for root, _, files in os.walk('.'):
    if '.git' in root or '__pycache__' in root:
        continue
    for file in files:
        if file.endswith('.py'):
            path = os.path.join(root, file).replace('\\', '/').removeprefix('./')
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            new_lines = []
            skip_next = 0
            changed = False
            for i, line in enumerate(lines):
                if skip_next > 0:
                    skip_next -= 1
                    continue
                
                    changed = True
                    # Remove preceding 'import sys\n' and 'import os\n' if they exist and are just there for this
                    if len(new_lines) >= 1 and new_lines[-1].strip() == 'import os':
                        if len(new_lines) >= 2 and new_lines[-2].strip() == 'import sys':
                            new_lines.pop()
                            new_lines.pop()
                        elif len(new_lines) >= 2 and new_lines[-2].strip() == 'import sys, os':
                             new_lines.pop()
                             new_lines.pop()
                        else:
                            # Maybe sys was earlier?
                            pass
                    elif len(new_lines) >= 1 and new_lines[-1].strip() == 'import sys':
                        new_lines.pop()
                    
                    # Also look ahead for the next 'from X import Y'
                    for j in range(i+1, min(i+5, len(lines))):
                        if lines[j].startswith('from '):
                            parts = lines[j].split(' ')
                            if len(parts) >= 4 and parts[2] == 'import':
                                module = parts[1]
                                rel_mod = get_relative_import(path, module)
                                lines[j] = lines[j].replace(f"from {module} import", f"from {rel_mod} import")
                            break
                    continue
                
                new_lines.append(line)
            
            if changed:
                with open(path, 'w', encoding='utf-8') as f:
                    f.writelines(new_lines)
                print(f"Fixed {path}")
