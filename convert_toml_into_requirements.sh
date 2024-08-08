#!/bin/bash

# Check if the pyproject.toml file exists
if [ ! -f "pyproject.toml" ]; then
    echo "pyproject.toml file not found!"
    exit 1
fi

# Create a temporary Python script to parse pyproject.toml
temp_script=$(mktemp)

cat << 'EOF' > "$temp_script"
import toml
import sys

def parse_dependencies(pyproject_file):
    dependencies = set()  # Use a set to avoid duplicates
    
    try:
        pyproject = toml.load(pyproject_file)
        
        def convert_version(version_str):
            # Convert caret (^) version to the format required by requirements.txt
            if version_str.startswith('^'):
                version_str = version_str[1:]  # Remove the caret
                major, minor = version_str.split('.', 2)[:2]
                return f">={major}.{minor}.0"
            return version_str

        # Parse main dependencies
        for dependency, version in pyproject.get('tool', {}).get('poetry', {}).get('dependencies', {}).items():
            if dependency == 'python':
                continue  # Skip Python version constraint
            
            if isinstance(version, str):
                version = convert_version(version)
                dependencies.add(f"{dependency}{version}")
            elif isinstance(version, dict) and 'git' in version:
                git_url = version['git']
                rev = version.get('rev', '')  # No default branch, use empty string
                if rev:
                    dependencies.add(f"git+{git_url}@{rev}")
                else:
                    dependencies.add(f"git+{git_url}")
            elif isinstance(version, dict):
                version_str = convert_version(version.get('version', '*'))
                dependencies.add(f"{dependency}{version_str}")

        # Parse dev dependencies
        for dependency, version in pyproject.get('tool', {}).get('poetry', {}).get('group', {}).get('dev', {}).get('dependencies', {}).items():
            if dependency == 'python':
                continue  # Skip Python version constraint
            
            if isinstance(version, str):
                version = convert_version(version)
                dependencies.add(f"{dependency}{version}")
            elif isinstance(version, dict) and 'git' in version:
                git_url = version['git']
                rev = version.get('rev', '')  # No default branch, use empty string
                if rev:
                    dependencies.add(f"git+{git_url}@{rev}")
                else:
                    dependencies.add(f"git+{git_url}")
            elif isinstance(version, dict):
                version_str = convert_version(version.get('version', '*'))
                dependencies.add(f"{dependency}{version_str}")

    except Exception as e:
        print(f"Error parsing pyproject.toml: {e}")
        sys.exit(1)

    return sorted(dependencies)  # Sort for consistent output

def main():
    dependencies = parse_dependencies('pyproject.toml')
    with open('requirements.txt', 'w') as f:
        for dep in dependencies:
            f.write(f"{dep}\n")

if __name__ == "__main__":
    main()
EOF

# Run the temporary Python script
python3 "$temp_script"

# Clean up the temporary script
rm "$temp_script"

# Notify the user
if [ $? -eq 0 ]; then
    echo "Successfully exported dependencies to requirements.txt"
else
    echo "Failed to export dependencies."
    exit 1
fi
