#!/usr/bin/env python3
import os
import json
import datetime
import time

def generate_code_report(root_dir='.', output_file='code_report.json'):
    """
    Generate a JSON report that appends all Python files with their metadata.
    Each new file is appended to the existing report file, making it easy to 
    add new .py files as they are created.
    """
    # Exclude virtual environment directories
    exclude_dirs = ['venv', 'env', '.env', '.venv', '__pycache__']
    
    # Initialize or load the existing report
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                report = json.load(f)
                # Get list of already processed files
                processed_files = {file_entry['metadata']['path'] for file_entry in report['files']}
        except (json.JSONDecodeError, KeyError):
            # If file is corrupted or has wrong structure, start fresh
            report = {
                'report_generated': datetime.datetime.now().isoformat(),
                'total_files': 0,
                'files': []
            }
            processed_files = set()
    else:
        # Create a new report
        report = {
            'report_generated': datetime.datetime.now().isoformat(),
            'total_files': 0,
            'files': []
        }
        processed_files = set()
    
    # Track new files added in this run
    new_files_added = 0
    
    # Walk through directory structure
    for root, dirs, files in os.walk(root_dir):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, root_dir)
                
                # Skip if file was already processed
                if rel_path in processed_files:
                    continue
                
                # Get file stats
                stats = os.stat(file_path)
                
                try:
                    # Read the full file content
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Create metadata for the file
                    metadata = {
                        'filename': file,
                        'path': rel_path,
                        'size_bytes': stats.st_size,
                        'line_count': content.count('\n') + 1,
                        'last_modified': datetime.datetime.fromtimestamp(stats.st_mtime).isoformat(),
                        'created': datetime.datetime.fromtimestamp(stats.st_ctime).isoformat(),
                    }
                    
                    # Add file entry to report
                    report['files'].append({
                        'metadata': metadata,
                        'content': content
                    })
                    
                    new_files_added += 1
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    # Update total files count and timestamp
    report['total_files'] = len(report['files'])
    report['last_updated'] = datetime.datetime.now().isoformat()
    
    # Save the updated report
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    return new_files_added, report['total_files']

if __name__ == '__main__':
    start_time = time.time()
    project_root = os.path.dirname(os.path.abspath(__file__))
    new_files, total_files = generate_code_report(project_root)
    
    print(f"Report saved to code_report.json")
    print(f"New Python files added: {new_files}")
    print(f"Total Python files in report: {total_files}")
    print(f"Execution time: {time.time() - start_time:.2f} seconds")
    
    if new_files == 0:
        print("No new Python files found to add to the report.")