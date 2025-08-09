#!/usr/bin/env python3
"""
Example script demonstrating the File Organizer functionality.

This script shows how to use the intelligent file organization system
for the Research File Manager MVP.
"""

import os
import tempfile
from pathlib import Path

# Mock the database imports for demonstration
class MockSession:
    def query(self, model): return self
    def filter(self, *args): return self
    def all(self): return []
    def first(self): return None
    def add(self, obj): pass
    def commit(self): pass
    def rollback(self): pass
    def close(self): pass

class MockDBSession:
    def __enter__(self): return MockSession()
    def __exit__(self, *args): pass

# Mock database functions
def db_session(): return MockDBSession()
def get_project_by_id(session, project_id): return None
def get_files_by_project(session, project_id): return []
def create_file(*args, **kwargs): pass
def get_file_by_id(session, file_id): return None

class File:
    def __init__(self):
        self.metadata = {}
        self.content = ""
        self.embedding = None
        self.type = ""
        self.path = ""

class Project:
    def __init__(self, name="Test Project", path="./test"):
        self.name = name
        self.path = path

# Now import our organizer with mocked dependencies
import sys
sys.modules['database'] = sys.modules[__name__]

from organizer import FileOrganizer, create_project_structure, OrganizationRule


def create_sample_files(base_dir: str):
    """Create sample files for testing the organizer."""
    sample_files = [
        ("experiment_data_2024.csv", "experiment,trial,result\n1,A,0.85\n2,B,0.92"),
        ("analysis_script.py", "import pandas as pd\ndef analyze_data():\n    pass"),
        ("results_figure.png", ""),  # Binary file, empty content
        ("research_paper.pdf", ""),  # Binary file, empty content
        ("config.yaml", "database:\n  host: localhost\n  port: 5432"),
        ("experiment_log.txt", "Experiment started at 2024-01-15\nResults: significant improvement"),
        ("model_v1.pkl", ""),  # Binary file
        ("draft_manuscript.md", "# Research Paper Draft\n## Introduction\nThis study examines..."),
        ("temp_calculations.tmp", "temporary calculations here"),
        ("backup.zip", ""),  # Archive file
        ("README.md", "# Project README\nThis is the main project file")
    ]
    
    created_files = []
    for filename, content in sample_files:
        file_path = os.path.join(base_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        created_files.append(file_path)
        print(f"📄 Created sample file: {filename}")
    
    return created_files


def demonstrate_file_classification():
    """Demonstrate file classification capabilities."""
    print("\n🔍 DEMONSTRATING FILE CLASSIFICATION")
    print("=" * 50)
    
    # Create organizer
    organizer = FileOrganizer()
    print(f"✅ Created organizer with {len(organizer.rules)} classification rules")
    
    # Create temporary directory and sample files
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📁 Working in temporary directory: {temp_dir}")
        
        # Create sample files
        sample_files = create_sample_files(temp_dir)
        
        print(f"\n📋 CLASSIFICATION RESULTS:")
        print("-" * 30)
        
        # Test classification for each file
        for file_path in sample_files:
            filename = os.path.basename(file_path)
            
            # Skip binary files that don't exist (created empty)
            if not os.path.exists(file_path):
                continue
                
            suggestion = organizer.suggest_organization(file_path)
            
            print(f"📄 {filename:<25} → {suggestion.suggested_category:<12} "
                  f"(confidence: {suggestion.confidence:.2f}) - {suggestion.reason}")
        
        print(f"\n🗂️  BULK ORGANIZATION TEST:")
        print("-" * 30)
        
        # Test bulk organization (dry run)
        result = organizer.organize_project(
            project_path=temp_dir,
            auto_move=False,  # Don't actually move files
            dry_run=True
        )
        
        print(f"Total files found: {result.total_files}")
        print(f"Files processed: {result.processed_files}")
        print(f"Suggestions generated: {len(result.suggestions)}")
        print(f"Execution time: {result.execution_time:.2f} seconds")
        
        # Show category breakdown
        category_counts = {}
        for suggestion in result.suggestions:
            category = suggestion.suggested_category
            category_counts[category] = category_counts.get(category, 0) + 1
        
        print(f"\nCategory breakdown:")
        for category, count in sorted(category_counts.items()):
            print(f"  {category}: {count} files")


def demonstrate_project_templates():
    """Demonstrate project template creation."""
    print("\n🏗️  DEMONSTRATING PROJECT TEMPLATES")
    print("=" * 50)
    
    templates = ['research', 'minimal', 'data_science', 'software_dev']
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for template in templates:
            project_path = os.path.join(temp_dir, f"project_{template}")
            
            print(f"\n📁 Creating {template} project structure...")
            result = create_project_structure(project_path, template)
            
            if 'error' not in result:
                print(f"✅ Created {result['folders_created']} folders")
                print(f"📋 Folders: {', '.join(result['folder_list'][:5])}...")
                
                # Check if README was created
                readme_path = os.path.join(project_path, 'README.md')
                if os.path.exists(readme_path):
                    print("📄 README.md generated successfully")
                    
                    # Show first few lines of README
                    with open(readme_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()[:3]
                        for line in lines:
                            print(f"   {line.strip()}")
            else:
                print(f"❌ Error: {result['error']}")


def demonstrate_custom_rules():
    """Demonstrate custom organization rules."""
    print("\n⚙️  DEMONSTRATING CUSTOM RULES")
    print("=" * 50)
    
    organizer = FileOrganizer()
    
    # Add custom rule for protocol files
    protocol_rule = OrganizationRule(
        category="protocols",
        extensions={'.protocol', '.method', '.sop'},
        name_patterns={'protocol', 'method', 'procedure', 'sop'},
        content_keywords={'step', 'procedure', 'protocol', 'method'},
        priority=15  # High priority
    )
    
    organizer.add_custom_rule(protocol_rule)
    print(f"✅ Added custom rule for 'protocols' category")
    print(f"📋 Organizer now has {len(organizer.rules)} rules")
    
    # Test with custom files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create protocol file
        protocol_file = os.path.join(temp_dir, "lab_protocol_pcr.txt")
        with open(protocol_file, 'w') as f:
            f.write("PCR Protocol\nStep 1: Prepare samples\nStep 2: Add reagents\nProcedure notes...")
        
        # Test classification
        suggestion = organizer.suggest_organization(protocol_file)
        print(f"📄 {os.path.basename(protocol_file)} → {suggestion.suggested_category} "
              f"(confidence: {suggestion.confidence:.2f})")
        print(f"   Reason: {suggestion.reason}")


def demonstrate_research_patterns():
    """Demonstrate research-specific pattern recognition."""
    print("\n🧬 DEMONSTRATING RESEARCH PATTERNS")
    print("=" * 50)
    
    organizer = FileOrganizer()
    
    # Test files with research patterns
    research_files = [
        "exp001_baseline.csv",
        "experiment_2024_01_15.py",
        "run_batch_003.txt", 
        "trial_v2.3_results.xlsx",
        "analysis_correlation_final.ipynb",
        "figure_1_draft.png",
        "model_regression_v1.pkl",
        "draft_preliminary_findings.docx"
    ]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for filename in research_files:
            # Create the file
            file_path = os.path.join(temp_dir, filename)
            with open(file_path, 'w') as f:
                f.write("Research file content for pattern matching test")
            
            # Test classification
            suggestion = organizer.suggest_organization(file_path)
            print(f"📄 {filename:<30} → {suggestion.suggested_category:<12} "
                  f"(confidence: {suggestion.confidence:.2f})")


def main():
    """Run all demonstrations."""
    print("🔬 RESEARCH FILE MANAGER - ORGANIZER DEMONSTRATION")
    print("=" * 60)
    
    try:
        demonstrate_file_classification()
        demonstrate_project_templates()
        demonstrate_custom_rules()
        demonstrate_research_patterns()
        
        print("\n" + "=" * 60)
        print("✅ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print("\n📝 SUMMARY:")
        print("- File classification system works with multiple criteria")
        print("- Project templates create structured directories") 
        print("- Custom rules can be added for domain-specific needs")
        print("- Research patterns are automatically recognized")
        print("- System handles various file types and naming conventions")
        print("- Comprehensive error handling and logging included")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()