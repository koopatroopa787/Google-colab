#!/usr/bin/env python3
"""
Automated Setup and Testing Script

This script will:
1. Create a virtual environment
2. Install all dependencies
3. Test each project to ensure it's working
4. Generate a report
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text):
    """Print a formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")


def print_success(text):
    """Print success message"""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_error(text):
    """Print error message"""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


def print_warning(text):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


def print_info(text):
    """Print info message"""
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")


def run_command(command, cwd=None, check=True, capture_output=False):
    """Run a shell command"""
    try:
        if capture_output:
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                check=check,
                capture_output=True,
                text=True
            )
            return result.returncode == 0, result.stdout, result.stderr
        else:
            result = subprocess.run(command, shell=True, cwd=cwd, check=check)
            return result.returncode == 0, "", ""
    except subprocess.CalledProcessError as e:
        return False, "", str(e)


def get_python_command():
    """Get the appropriate Python command for the system"""
    if platform.system() == "Windows":
        return "python"
    return "python3"


def get_venv_activate_command():
    """Get the virtual environment activation command"""
    if platform.system() == "Windows":
        return "venv\\Scripts\\activate"
    return "source venv/bin/activate"


def get_pip_command():
    """Get the pip command for virtual environment"""
    if platform.system() == "Windows":
        return "venv\\Scripts\\pip"
    return "venv/bin/pip"


def create_virtual_environment():
    """Create a virtual environment"""
    print_header("Creating Virtual Environment")

    python_cmd = get_python_command()

    # Check if venv already exists
    if os.path.exists("venv"):
        print_warning("Virtual environment already exists. Skipping creation.")
        return True

    print_info(f"Creating virtual environment using {python_cmd}...")
    success, _, error = run_command(f"{python_cmd} -m venv venv")

    if success:
        print_success("Virtual environment created successfully!")
        return True
    else:
        print_error(f"Failed to create virtual environment: {error}")
        return False


def install_dependencies():
    """Install all dependencies from requirements.txt"""
    print_header("Installing Dependencies")

    pip_cmd = get_pip_command()

    # Upgrade pip first
    print_info("Upgrading pip...")
    run_command(f"{pip_cmd} install --upgrade pip")

    # Install requirements
    print_info("Installing requirements from requirements.txt...")
    success, stdout, stderr = run_command(
        f"{pip_cmd} install -r requirements.txt",
        capture_output=True
    )

    if success:
        print_success("All dependencies installed successfully!")
        return True
    else:
        print_error(f"Failed to install dependencies: {stderr}")
        return False


def test_imports(project_name, imports):
    """Test if required modules can be imported"""
    print_info(f"Testing imports for {project_name}...")

    python_cmd = get_pip_command().replace("pip", "python")

    for module in imports:
        test_code = f"import {module}; print('{module} imported successfully')"
        success, stdout, stderr = run_command(
            f'{python_cmd} -c "{test_code}"',
            capture_output=True
        )

        if success:
            print_success(f"  {module} ✓")
        else:
            print_error(f"  {module} ✗ - {stderr[:100]}")
            return False

    return True


def test_transformers_project():
    """Test the Transformers from Scratch project"""
    print_header("Testing: Transformers from Scratch")

    imports = [
        "torch",
        "numpy",
        "pandas",
        "matplotlib",
        "plotly",
        "gradio"
    ]

    if not test_imports("Transformers", imports):
        return False

    # Test project imports
    print_info("Testing project modules...")
    python_cmd = get_pip_command().replace("pip", "python")

    test_code = """
import sys
sys.path.insert(0, '.')
try:
    from transformers_from_scratch.models.components import RMSNorm, SwiGLU
    from transformers_from_scratch.models.llama import Llama
    print('Project modules imported successfully')
    exit(0)
except Exception as e:
    print(f'Import error: {e}')
    exit(1)
"""

    with open("test_transformers_import.py", "w") as f:
        f.write(test_code)

    success, stdout, stderr = run_command(
        f"{python_cmd} test_transformers_import.py",
        capture_output=True
    )

    os.remove("test_transformers_import.py")

    if success:
        print_success("Transformers project imports work correctly!")
        return True
    else:
        print_error(f"Transformers project import failed: {stderr}")
        return False


def test_stable_diffusion_project():
    """Test the Stable Diffusion project"""
    print_header("Testing: Stable Diffusion")

    imports = [
        "torch",
        "diffusers",
        "transformers",
        "gradio"
    ]

    if not test_imports("Stable Diffusion", imports):
        return False

    # Test project imports
    print_info("Testing project modules...")
    python_cmd = get_pip_command().replace("pip", "python")

    test_code = """
import sys
sys.path.insert(0, '.')
try:
    from stable_diffusion.core.generator import StableDiffusionGenerator
    print('Project modules imported successfully')
    exit(0)
except Exception as e:
    print(f'Import error: {e}')
    exit(1)
"""

    with open("test_sd_import.py", "w") as f:
        f.write(test_code)

    success, stdout, stderr = run_command(
        f"{python_cmd} test_sd_import.py",
        capture_output=True
    )

    os.remove("test_sd_import.py")

    if success:
        print_success("Stable Diffusion project imports work correctly!")
        return True
    else:
        print_error(f"Stable Diffusion project import failed: {stderr}")
        return False


def test_mistral_rag_project():
    """Test the Mistral RAG project"""
    print_header("Testing: Mistral RAG")

    imports = [
        "torch",
        "transformers",
        "langchain",
        "langchain_community",
        "langchain_core",
        "sentence_transformers",
        "faiss",
        "gradio"
    ]

    if not test_imports("Mistral RAG", imports):
        return False

    # Test project imports
    print_info("Testing project modules...")
    python_cmd = get_pip_command().replace("pip", "python")

    test_code = """
import sys
sys.path.insert(0, '.')
try:
    from mistral_rag.core.rag_system import MistralRAGSystem
    print('Project modules imported successfully')
    exit(0)
except Exception as e:
    print(f'Import error: {e}')
    exit(1)
"""

    with open("test_rag_import.py", "w") as f:
        f.write(test_code)

    success, stdout, stderr = run_command(
        f"{python_cmd} test_rag_import.py",
        capture_output=True
    )

    os.remove("test_rag_import.py")

    if success:
        print_success("Mistral RAG project imports work correctly!")
        return True
    else:
        print_error(f"Mistral RAG project import failed: {stderr}")
        return False


def test_rl_project():
    """Test the Reinforcement Learning project"""
    print_header("Testing: BipedalWalker RL")

    imports = [
        "gymnasium",
        "stable_baselines3",
        "gradio"
    ]

    if not test_imports("BipedalWalker RL", imports):
        return False

    # Test project imports
    print_info("Testing project modules...")
    python_cmd = get_pip_command().replace("pip", "python")

    test_code = """
import sys
sys.path.insert(0, '.')
try:
    from rl_bipedal_walker.core.trainer import BipedalWalkerTrainer
    print('Project modules imported successfully')
    exit(0)
except Exception as e:
    print(f'Import error: {e}')
    exit(1)
"""

    with open("test_rl_import.py", "w") as f:
        f.write(test_code)

    success, stdout, stderr = run_command(
        f"{python_cmd} test_rl_import.py",
        capture_output=True
    )

    os.remove("test_rl_import.py")

    if success:
        print_success("BipedalWalker RL project imports work correctly!")
        return True
    else:
        print_error(f"BipedalWalker RL project import failed: {stderr}")
        return False


def generate_report(results):
    """Generate a final test report"""
    print_header("Test Report")

    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r)
    failed_tests = total_tests - passed_tests

    print(f"Total Projects: {total_tests}")
    print(f"Passed: {Colors.OKGREEN}{passed_tests}{Colors.ENDC}")
    print(f"Failed: {Colors.FAIL}{failed_tests}{Colors.ENDC}")
    print()

    for project, status in results.items():
        status_text = f"{Colors.OKGREEN}PASSED{Colors.ENDC}" if status else f"{Colors.FAIL}FAILED{Colors.ENDC}"
        print(f"  {project}: {status_text}")

    print()

    if failed_tests == 0:
        print_success("All projects are working correctly! 🎉")
        return True
    else:
        print_error(f"{failed_tests} project(s) failed. Please check the errors above.")
        return False


def main():
    """Main execution function"""
    print_header("AI/ML Projects - Setup and Testing")

    print_info(f"Platform: {platform.system()}")
    print_info(f"Python: {sys.version.split()[0]}")
    print_info(f"Working Directory: {os.getcwd()}")
    print()

    # Step 1: Create virtual environment
    if not create_virtual_environment():
        print_error("Setup failed at virtual environment creation!")
        return False

    # Step 2: Install dependencies
    if not install_dependencies():
        print_error("Setup failed at dependency installation!")
        return False

    # Step 3: Test each project
    results = {
        "Transformers from Scratch": test_transformers_project(),
        "Stable Diffusion": test_stable_diffusion_project(),
        "Mistral RAG": test_mistral_rag_project(),
        "BipedalWalker RL": test_rl_project()
    }

    # Step 4: Generate report
    success = generate_report(results)

    if success:
        print_header("Setup Complete!")
        print_info("You can now run any of the projects:")
        print("  • python transformers_from_scratch/app.py")
        print("  • python stable_diffusion/app.py")
        print("  • python mistral_rag/app.py")
        print("  • python rl_bipedal_walker/app.py")
        print()
        print_info("Or run the main hub:")
        print("  • python app.py")
        print()

        # Show activation command
        if platform.system() == "Windows":
            print_info("To activate the virtual environment:")
            print("  venv\\Scripts\\activate")
        else:
            print_info("To activate the virtual environment:")
            print("  source venv/bin/activate")

    return success


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n")
        print_warning("Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
