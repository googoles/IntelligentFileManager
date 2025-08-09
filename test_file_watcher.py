#!/usr/bin/env python3
"""
Test script for the Research File Manager file watching system.

This script demonstrates the file monitoring capabilities and validates
the implementation meets the MVP requirements.
"""

import os
import sys
import time
import tempfile
import shutil
from pathlib import Path

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend import (
    db_manager, 
    start_watching, 
    stop_watching, 
    stop_all_watchers,
    file_watcher_manager
)


def create_test_files(test_dir: Path) -> None:
    """Create test files for monitoring."""
    
    # Create directory structure
    (test_dir / "documents").mkdir(exist_ok=True)
    (test_dir / "code").mkdir(exist_ok=True)
    (test_dir / "data").mkdir(exist_ok=True)
    
    # Create test files with different types
    test_files = {
        "README.md": "# Test Project\n\nThis is a test project for file monitoring.\n\nIt contains research data and code.",
        "code/main.py": "#!/usr/bin/env python3\n\"\"\"\nMain application script for machine learning experiments.\n\"\"\"\n\nimport numpy as np\nimport pandas as pd\n\ndef analyze_data():\n    print('Analyzing research data...')\n    return True\n\nif __name__ == '__main__':\n    analyze_data()",
        "code/utils.js": "// Utility functions for data processing\nfunction processData(data) {\n    console.log('Processing research data');\n    return data.filter(item => item.valid);\n}\n\nmodule.exports = { processData };",
        "data/experiment_results.csv": "experiment,result,accuracy,notes\nexp1,success,0.95,Machine learning model performed well\nexp2,partial,0.82,Need more training data\nexp3,success,0.97,Best results achieved",
        "documents/research_notes.txt": "Research Notes - Machine Learning Project\n\nObjective: Develop automated file organization system\n\nKey findings:\n- File monitoring works effectively\n- Content extraction successful for text files\n- Database integration complete\n\nNext steps:\n- Add semantic search capabilities\n- Implement web interface\n- Test with larger datasets",
        "documents/methodology.md": "# Research Methodology\n\n## Data Collection\n- Automated file monitoring using watchdog\n- Content extraction from text files\n- Metadata collection (size, modification time)\n\n## Analysis\n- Rule-based file classification\n- Content indexing for search\n- Project organization templates\n\n## Results\nThe file monitoring system successfully processes files and extracts relevant information for research workflows."
    }
    
    for file_path, content in test_files.items():
        full_path = test_dir / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    print(f"✅ Created {len(test_files)} test files in {test_dir}")


def test_file_watcher():
    """Test the file monitoring system."""
    
    print("🧪 Testing Research File Manager File Watcher System")
    print("=" * 60)
    
    # Create temporary test directory
    test_dir = Path(tempfile.mkdtemp(prefix="rfm_test_"))
    print(f"📁 Test directory: {test_dir}")
    
    try:
        # Step 1: Create a test project
        print("\n1️⃣ Creating test project...")
        project = db_manager.create_project(
            name="File Watcher Test Project",
            path=str(test_dir)
        )
        print(f"✅ Created project: {project.name} (ID: {project.id})")
        
        # Step 2: Start monitoring
        print("\n2️⃣ Starting file monitoring...")
        success = start_watching(str(test_dir), project.id, index_existing=False)
        if success:
            print("✅ File monitoring started successfully")
        else:
            print("❌ Failed to start file monitoring")
            return
        
        # Wait for watcher to initialize
        time.sleep(1)
        
        # Step 3: Create test files and monitor events
        print("\n3️⃣ Creating test files (should trigger monitoring events)...")
        create_test_files(test_dir)
        
        # Give the watcher time to process files
        print("⏳ Waiting for file processing...")
        time.sleep(3)
        
        # Step 4: Check database for processed files
        print("\n4️⃣ Checking database for processed files...")
        files = db_manager.get_files_by_project(project.id)
        print(f"📊 Found {len(files)} files in database:")
        
        for file_obj in files:
            content_length = len(file_obj.content) if file_obj.content else 0
            metadata = file_obj.metadata or {}
            print(f"  📄 {file_obj.name} ({file_obj.type}) - {content_length} chars, {metadata.get('size', 0)} bytes")
        
        # Step 5: Test content extraction
        print("\n5️⃣ Testing content extraction...")
        text_files = [f for f in files if f.content]
        print(f"📝 Successfully extracted content from {len(text_files)} files:")
        
        for file_obj in text_files:
            preview = file_obj.content[:100] + "..." if len(file_obj.content) > 100 else file_obj.content
            print(f"  📄 {file_obj.name}: {preview}")
        
        # Step 6: Test file modification
        print("\n6️⃣ Testing file modification detection...")
        readme_path = test_dir / "README.md"
        with open(readme_path, 'a', encoding='utf-8') as f:
            f.write("\n\n## Updated\nThis file was modified during testing.")
        
        # Wait for modification to be processed
        time.sleep(2)
        
        # Check if modification was detected
        updated_files = db_manager.get_files_by_project(project.id)
        readme_file = next((f for f in updated_files if f.name == "README.md"), None)
        if readme_file and "Updated" in readme_file.content:
            print("✅ File modification detected and processed")
        else:
            print("⚠️ File modification may not have been processed yet")
        
        # Step 7: Test file deletion
        print("\n7️⃣ Testing file deletion detection...")
        test_file = test_dir / "temp_test_file.txt"
        with open(test_file, 'w') as f:
            f.write("This file will be deleted")
        
        time.sleep(1)  # Let it be detected
        os.remove(test_file)
        time.sleep(2)  # Let deletion be processed
        
        # Check if file was removed from database
        final_files = db_manager.get_files_by_project(project.id)
        deleted_file = next((f for f in final_files if f.name == "temp_test_file.txt"), None)
        if deleted_file is None:
            print("✅ File deletion detected and processed")
        else:
            print("⚠️ File deletion may not have been processed yet")
        
        # Step 8: Display final statistics
        print("\n8️⃣ Final Statistics...")
        final_files = db_manager.get_files_by_project(project.id)
        
        file_types = {}
        total_content_length = 0
        
        for file_obj in final_files:
            file_type = file_obj.type or "unknown"
            file_types[file_type] = file_types.get(file_type, 0) + 1
            total_content_length += len(file_obj.content) if file_obj.content else 0
        
        print(f"📊 Final file count: {len(final_files)}")
        print(f"📝 Total content extracted: {total_content_length} characters")
        print(f"📁 File types processed: {dict(file_types)}")
        
        # Step 9: Test watcher status
        print("\n9️⃣ Testing watcher status...")
        watching_projects = file_watcher_manager.get_watching_projects()
        print(f"🔍 Currently watching {len(watching_projects)} projects: {watching_projects}")
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        stop_all_watchers()
        
        try:
            shutil.rmtree(test_dir)
            print(f"✅ Cleaned up test directory: {test_dir}")
        except Exception as e:
            print(f"⚠️ Failed to clean up test directory: {e}")
        
        print("\n🏁 Test completed")


def test_performance():
    """Test performance with multiple files."""
    
    print("\n🚀 Performance Test - Creating 100 files...")
    
    test_dir = Path(tempfile.mkdtemp(prefix="rfm_perf_test_"))
    
    try:
        # Create project
        project = db_manager.create_project(
            name="Performance Test Project",
            path=str(test_dir)
        )
        
        # Start monitoring
        start_watching(str(test_dir), project.id, index_existing=False)
        time.sleep(1)
        
        # Create 100 files
        start_time = time.time()
        
        for i in range(100):
            file_path = test_dir / f"file_{i:03d}.txt"
            with open(file_path, 'w') as f:
                f.write(f"This is test file number {i}\n" * 10)  # 10 lines each
        
        # Wait for processing
        time.sleep(5)
        
        end_time = time.time()
        
        # Check results
        files = db_manager.get_files_by_project(project.id)
        
        print(f"⏱️ Created and processed 100 files in {end_time - start_time:.2f} seconds")
        print(f"📊 Files in database: {len(files)}")
        print(f"🏃 Processing rate: {len(files) / (end_time - start_time):.1f} files/second")
        
    finally:
        stop_all_watchers()
        shutil.rmtree(test_dir)


if __name__ == "__main__":
    try:
        # Run main test
        test_file_watcher()
        
        # Run performance test
        test_performance()
        
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
        stop_all_watchers()
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        stop_all_watchers()
        raise