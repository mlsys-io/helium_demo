#!/bin/bash
# Python formatter and linter script
#
# Usage:
#   bash format.sh [--all | --files <file1> <file2> ...]

set -eo pipefail

# Ensure script is run from the project root
builtin cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT="$(git rev-parse --show-toplevel)"
builtin cd "$ROOT" || exit 1

# Check if a command is available
check_command() {
    if ! command -v "$1" &>/dev/null; then
        echo "❓❓$1 is not installed. Run 'bash install_tools.sh' to install the required tools."
        exit 1
    fi
}

# Check if required tools are installed
check_command black
check_command ruff
check_command mypy
check_command codespell
check_command isort

# Function to sort imports with isort
sort_imports() {
    isort "$@"
}

# Function to format files with Black
format_files() {
    black "$@"
}

# Function to lint files with Ruff
lint_files() {
    ruff check "$@"
}

# Function to type-check files with MyPy
type_check_files() {
    mypy "$@"
}

# Function to check spelling with Codespell
spell_check_files() {
    codespell "$@"
}

# Format, lint, type-check, and spell-check all files
process_all() {
    echo "Sorting imports with isort..."
    isort .
    echo "Formatting all Python files with Black..."
    black .
    echo "Linting all Python files with Ruff..."
    ruff check .
    echo "Type-checking with MyPy..."
    mypy
    echo "Checking spelling with Codespell..."
    codespell .
}

# Process only changed files
process_changed() {
    MERGEBASE="$(git merge-base origin/main HEAD)"

    # Get the list of changed files
    changed_files=$(git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.py')

    if [ -n "$changed_files" ]; then
        echo "Processing changed Python files..."
        echo "$changed_files" | xargs -P 5 -n 1 isort
        echo "$changed_files" | xargs -P 5 -n 1 black
        echo "$changed_files" | xargs -P 5 -n 1 ruff check
        echo "$changed_files" | xargs -P 5 -n 1 mypy
        echo "$changed_files" | xargs -P 5 -n 1 codespell
    else
        echo "No changed Python files to process."
    fi
}

# Main logic
if [[ "$1" == '--all' ]]; then
    process_all
elif [ $# -gt 0 ]; then
    echo "Processing specified files..."
    sort_imports "$@"
    format_files "$@"
    lint_files "$@"
    type_check_files "$@"
    spell_check_files "$@"
else
    process_changed
fi

echo "✨🎉 All checks passed! 🎉✨"