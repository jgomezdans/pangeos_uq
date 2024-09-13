import os
import ast
import sys
import importlib.util

def is_standard_lib(module_name):
    """Check if a module is part of the standard library."""
    if module_name in sys.builtin_module_names:
        return True
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.origin is None:
            return False
        return 'site-packages' not in spec.origin
    except ModuleNotFoundError:
        return False

def get_imported_modules(file_path):
    """Extract imported modules from a Python file."""
    with open(file_path, "r") as file:
        node = ast.parse(file.read(), filename=file_path)
    
    imports = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Import):
            for alias in n.names:
                imports.add(alias.name.split('.')[0])
        elif isinstance(n, ast.ImportFrom):
            imports.add(n.module.split('.')[0])
    
    return imports

def scan_directory_for_imports(directory):
    """Recursively scan a directory for Python files and extract imports."""
    all_imports = set()
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                imports = get_imported_modules(file_path)
                all_imports.update(imports)
    
    return all_imports

def main():
    # Set your source directory path here
    source_directory = "src/pangeos_uq"
    
    all_imports = scan_directory_for_imports(source_directory)
    third_party_imports = {imp for imp in all_imports if not is_standard_lib(imp)}
    
    print("Third-party packages used:")
    for package in sorted(third_party_imports):
        print(package)

if __name__ == "__main__":
    main()

