import os
import re

target_files = [
    'bandit/pipeline.py',
    'bandit/loop.py',
    'gpu_kernels/pipeline.py',
    'surrogate_triage/pipeline.py',
    'model_scientist/integration/loop_integrator.py',
    'meta/pipeline.py'
]

for path in target_files:
    if not os.path.exists(path):
        continue
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    changed = False
    
    # Check for logging import
    if 'import logging' not in content:
        # Find the first import and add it before
        content = re.sub(r'^(import [a-zA-Z_]+)', r'import logging\n\nlogger = logging.getLogger(__name__)\n\n\1', content, count=1, flags=re.MULTILINE)
        changed = True
    elif 'logger = ' not in content:
        content = re.sub(r'^(import logging.*?\n)', r'\1\nlogger = logging.getLogger(__name__)\n', content, count=1, flags=re.MULTILINE)
        changed = True
        
    # Replace except Exception: pass
    def repl(m):
        indent = m.group(1)
        return f"{indent}except Exception as e:\n{indent}    logger.exception(e)"
        
    new_content, num_subs = re.subn(r'([ \t]*)except Exception:\s+pass', repl, content)
    if num_subs > 0:
        content = new_content
        changed = True
        
    if changed:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated logging in {path} ({num_subs} replacements)")
