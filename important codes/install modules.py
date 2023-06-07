import importlib

# List of required modules
required_modules = ['pandas', 'tqdm']

# Check if all required modules are already installed
missing_modules = [module for module in required_modules if importlib.util.find_spec(module) is None]

# Install missing modules
if missing_modules:
    print("The following modules are missing: ", ", ".join(missing_modules))
    print("Installing required modules...")

    try:
        import pip
        pip.main(['install'] + missing_modules)
    except AttributeError:
        import subprocess
        subprocess.check_call(['pip', 'install'] + missing_modules)

    print("Required modules have been successfully installed.")
else:
    print("All required modules are already installed.")