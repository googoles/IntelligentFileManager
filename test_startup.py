#!/usr/bin/env python3
"""
Test script to verify all components are working properly.
Run this to check if the system can start without errors.
"""

import sys
import os
sys.path.append('backend')

def test_imports():
    """Test if all modules can be imported"""
    print("Testing imports...")
    
    try:
        print("  - Importing database...", end=" ")
        from database import init_database, db_session, Project, File
        print("✅")
    except Exception as e:
        print(f"❌ {e}")
        return False
    
    try:
        print("  - Importing file_watcher...", end=" ")
        from file_watcher import FileWatcherManager
        print("✅")
    except Exception as e:
        print(f"❌ {e}")
        return False
    
    try:
        print("  - Importing organizer...", end=" ")
        from organizer import FileOrganizer
        print("✅")
    except Exception as e:
        print(f"❌ {e}")
        return False
    
    try:
        print("  - Importing search (ChromaDB)...", end=" ")
        from search import SemanticSearch
        print("✅ ChromaDB available")
    except Exception as e:
        print(f"⚠️  ChromaDB not available: {e}")
        try:
            print("  - Importing search_simple (fallback)...", end=" ")
            from search_simple import SemanticSearch
            print("✅ Simple search available")
        except Exception as e2:
            print(f"❌ {e2}")
            return False
    
    return True


def test_database():
    """Test database initialization"""
    print("\nTesting database...")
    
    try:
        from database import init_database
        init_database()
        print("  ✅ Database initialized successfully")
        return True
    except Exception as e:
        print(f"  ❌ Database initialization failed: {e}")
        return False


def test_directories():
    """Check if required directories exist"""
    print("\nChecking directories...")
    
    dirs = [
        "data/projects",
        "data/db",
        "logs",
        "backend",
        "frontend"
    ]
    
    all_good = True
    for dir_path in dirs:
        if os.path.exists(dir_path):
            print(f"  ✅ {dir_path}")
        else:
            print(f"  ❌ {dir_path} - missing")
            all_good = False
    
    return all_good


def test_semantic_search():
    """Test semantic search initialization"""
    print("\nTesting semantic search...")
    
    try:
        # Try ChromaDB first
        try:
            from search import SemanticSearch
            search = SemanticSearch()
            print("  ✅ ChromaDB semantic search initialized")
        except:
            # Fall back to simple search
            from search_simple import SemanticSearch
            search = SemanticSearch()
            print("  ✅ Simple semantic search initialized (Windows-compatible)")
        
        return True
    except Exception as e:
        print(f"  ❌ Semantic search failed: {e}")
        return False


def main():
    print("="*50)
    print("Research File Manager MVP - System Test")
    print("="*50)
    
    # Create required directories
    os.makedirs("data/projects", exist_ok=True)
    os.makedirs("data/db", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    tests = [
        ("Directories", test_directories),
        ("Imports", test_imports),
        ("Database", test_database),
        ("Semantic Search", test_semantic_search)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test {name} crashed: {e}")
            results.append((name, False))
    
    print("\n" + "="*50)
    print("Test Results:")
    print("="*50)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("✅ All tests passed! The system is ready to run.")
        print("\nYou can now start the application with:")
        print("  python backend/main.py")
        print("  or")
        print("  run_mvp_windows.bat")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        print("\nTry running:")
        print("  fix_windows_setup.bat")
        print("  or")
        print("  clean_start.bat")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)