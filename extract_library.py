#!/usr/bin/env python3
import os
import shutil
import fnmatch
import sys

def find_files_recursive(root_dir, search_term):
    """Find all files/directories that contain the search term in their path."""
    matches = []
    for root, dirnames, filenames in os.walk(root_dir):
        # Check if search term is in the current directory path
        if search_term.lower() in root.lower():
            matches.append(root)
        
        # Check files in current directory
        for filename in filenames:
            path = os.path.join(root, filename)
            if search_term.lower() in path.lower():
                matches.append(path)
    
    return matches

def extract_files(source_dir, target_dir, search_terms, dry_run=False):
    """
    Extract files containing search terms from source_dir to target_dir,
    maintaining the original folder structure.
    
    Args:
        source_dir: Root directory to search in
        target_dir: Directory to extract files to
        search_terms: List of terms to search for in file/directory paths
        dry_run: If True, only print what would be done without copying files
    """
    # Make sure source directory exists
    if not os.path.isdir(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist")
        return
    
    # Create target directory if it doesn't exist
    if not dry_run and not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    all_matches = []
    
    # Find all files matching the search terms
    for term in search_terms:
        matches = find_files_recursive(source_dir, term)
        all_matches.extend(matches)
    
    # Remove duplicates
    all_matches = list(set(all_matches))
    
    if not all_matches:
        print(f"No files found containing terms {search_terms}")
        return
    
    print(f"Found {len(all_matches)} matches")
    
    # Copy files to target directory, maintaining folder structure
    for source_path in all_matches:
        # Get the relative path from the source directory
        relative_path = os.path.relpath(source_path, source_dir)
        target_path = os.path.join(target_dir, relative_path)
        target_dir_path = os.path.dirname(target_path)
        
        if os.path.isdir(source_path):
            print(f"Directory: {relative_path}")
            if not dry_run and not os.path.exists(target_path):
                os.makedirs(target_path)
        else:
            print(f"File: {relative_path}")
            if not dry_run:
                if not os.path.exists(target_dir_path):
                    os.makedirs(target_dir_path)
                shutil.copy2(source_path, target_path)
    
    print(f"{'Would extract' if dry_run else 'Extracted'} {len(all_matches)} files/directories to {target_dir}")

def get_user_input(prompt, default=None):
    """Get input from user with optional default value."""
    if default:
        response = input(f"{prompt} [{default}]: ")
        return response if response else default
    else:
        return input(f"{prompt}: ")

def main():
    # Default values
    default_source = './merged_only_library'
    default_target = './extracted_library'
    
    print("Library Extraction Tool")
    print("======================")
    
    source_dir = get_user_input("Enter source directory path", default_source)
    source_dir = os.path.abspath(source_dir)
    
    target_dir = get_user_input("Enter target directory path", default_target)
    target_dir = os.path.abspath(target_dir)
    
    print("\nEnter search terms (one per line, blank line to finish):")
    print("Example: Enter 'Makkot' to find all files with 'Makkot' in their path")
    terms = []
    while True:
        term = input("> ")
        if not term:
            break
        terms.append(term)
    
    if not terms:
        print("No search terms provided. Please enter at least one term.")
        return
    
    dry_run_input = get_user_input("Dry run (only show what would be done)? (y/n)", "n")
    dry_run = dry_run_input.lower() in ('y', 'yes')
    
    print(f"\nSummary:")
    print(f"Source directory: {source_dir}")
    print(f"Target directory: {target_dir}")
    print(f"Search terms: {terms}")
    print(f"Dry run: {dry_run}")
    
    confirm = get_user_input("Proceed? (y/n)", "y")
    if confirm.lower() not in ('y', 'yes'):
        print("Operation cancelled.")
        sys.exit(0)
    
    extract_files(source_dir, target_dir, terms, dry_run)

if __name__ == "__main__":
    main()