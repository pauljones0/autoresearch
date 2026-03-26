import os
import re

for root, _, files in os.walk('.'):
    if '.git' in root or '__pycache__' in root:
        continue
    for file in files:
        if file.endswith('.py'):
            path = os.path.join(root, file).replace('\\', '/').removeprefix('./')
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Replace 'from ...<package>' with 'from <package>'
            # actually it could be 'from ..model_scientist' or 'from ...model_scientist'
            new_content = re.sub(r'from \.\.+((?:model_scientist|gpu_kernels|surrogate_triage|bandit|meta)[A-Za-z0-9_.]*) import', r'from \1 import', content)
            
            if content != new_content:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f"Fixed absolute imports in {path}")
