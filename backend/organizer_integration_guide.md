# File Organizer Integration Guide

## Overview

The intelligent file organization and classification system provides comprehensive file management capabilities for the Research File Manager MVP. This system includes rule-based classification, project template creation, and intelligent pattern matching specifically designed for research workflows.

## Key Components

### 1. FileOrganizer Class

The main class that handles file classification and organization:

```python
from backend.organizer import FileOrganizer

# Initialize organizer
organizer = FileOrganizer()

# Get organization suggestion for a file
suggestion = organizer.suggest_organization("/path/to/file.csv")
print(f"Suggested category: {suggestion.suggested_category}")
print(f"Confidence: {suggestion.confidence:.2f}")
print(f"Reason: {suggestion.reason}")

# Organize entire project
result = organizer.organize_project(
    project_path="/path/to/project",
    auto_move=True,
    min_confidence=0.7
)
print(f"Processed {result.processed_files}/{result.total_files} files")
print(f"Moved {len(result.moved_files)} files automatically")
```

### 2. Project Template System

Create structured project directories with multiple templates:

```python
from backend.organizer import create_project_structure

# Create research project structure
result = create_project_structure(
    project_path="/path/to/new_project",
    template="research"
)

# Available templates:
# - 'research': Full research workflow (literature, data, code, results, etc.)
# - 'minimal': Simple structure (input, output, workspace)
# - 'data_science': Data science workflow (data, notebooks, models, reports)
# - 'software_dev': Software development (src, tests, docs, examples)
# - 'custom': Use custom_folders parameter

# Custom template example
custom_result = create_project_structure(
    project_path="/path/to/custom_project",
    template="custom",
    custom_folders=["raw_data", "processed_data", "analysis", "outputs"]
)
```

## File Classification Rules

### Supported Categories

1. **documents**: `.pdf`, `.doc`, `.docx`, `.txt`, `.md`, `.rtf`, `.odt`, `.tex`
2. **data**: `.csv`, `.xlsx`, `.json`, `.xml`, `.yaml`, `.parquet`, `.h5`, `.sqlite`
3. **code**: `.py`, `.js`, `.r`, `.ipynb`, `.java`, `.cpp`, `.sql`, etc.
4. **images**: `.png`, `.jpg`, `.gif`, `.svg`, `.bmp`, `.tiff`, etc.
5. **results**: Files with patterns like "result", "output", "figure", "plot"
6. **config**: `.conf`, `.ini`, `.cfg`, `.toml`, `.env`
7. **archives**: `.zip`, `.rar`, `.7z`, `.tar`, `.gz`

### Classification Logic

The system uses multiple criteria for classification:

1. **File Extension** (High confidence: 0.8)
2. **Filename Patterns** (Medium confidence: 0.6)
3. **Directory Context** (Medium confidence: 0.5)
4. **Content Keywords** (Lower confidence: 0.4)
5. **Research-Specific Patterns** (Variable confidence)

### Research-Specific Patterns

- **Experiment files**: `exp1.csv`, `experiment_2024.py`, `run_001.txt`
- **Version files**: `v1.2.pdf`, `version_3.docx`, `draft_v2.md`
- **Date patterns**: `2024-01-15_analysis.ipynb`, `20240115_results.png`
- **Analysis files**: `correlation_analysis.py`, `regression_model.r`

## Integration with Database

### Tracking Organization Actions

```python
from backend.organizer import track_organization_action

# Track when files are organized
track_organization_action(
    project_id="project-uuid",
    action_type="move",
    file_path="/new/path/to/file.csv",
    details={
        'original_path': "/old/path/file.csv",
        'category': "data",
        'confidence': 0.85,
        'auto_moved': True
    }
)
```

### Getting Organization Statistics

```python
from backend.organizer import get_project_organization_stats

stats = get_project_organization_stats("project-uuid")
print(f"Organization: {stats['organized_percentage']}%")
print(f"Files with content: {stats['content_percentage']}%")
print(f"File types: {stats['file_types']}")
```

## Integration with File Watcher

The organizer integrates seamlessly with the file watcher system:

```python
from backend.organizer import integrate_with_file_watcher, FileOrganizer

# In file watcher event handler
organizer = FileOrganizer()

suggestion = integrate_with_file_watcher(
    organizer=organizer,
    project_id="project-uuid",
    file_path="/path/to/new/file.csv",
    auto_organize=True  # Automatically move if confidence > 0.7
)

if suggestion and suggestion.action == OrganizationAction.MOVE:
    print(f"File automatically moved to: {suggestion.suggested_path}")
```

## Project Analysis

Analyze existing project organization:

```python
from backend.organizer import analyze_project_organization

analysis = analyze_project_organization("/path/to/project")
print(f"Organization status: {analysis['organization_percentage']}%")
print(f"Recommendations: {analysis['recommendations']}")
print(f"File types found: {analysis['file_types']}")
print(f"Large files (>100MB): {len(analysis['large_files'])}")
print(f"Potential duplicates: {len(analysis['duplicate_candidates'])}")
```

## Custom Rules

Add project-specific organization rules:

```python
from backend.organizer import OrganizationRule

# Create custom rule
custom_rule = OrganizationRule(
    category="protocols",
    extensions={'.protocol', '.method'},
    name_patterns={'protocol', 'method', 'procedure'},
    content_keywords={'step', 'procedure', 'protocol'},
    priority=12  # Higher than default rules
)

# Add to organizer
organizer.add_custom_rule(custom_rule)
```

## Error Handling

The system includes comprehensive error handling:

```python
try:
    result = organizer.organize_project("/path/to/project")
    
    # Check for errors
    if result.errors:
        print(f"Errors encountered: {len(result.errors)}")
        for error in result.errors:
            print(f"  - {error}")
    
    # Check skipped files
    if result.skipped_files:
        print(f"Skipped files: {len(result.skipped_files)}")
    
except Exception as e:
    print(f"Organization failed: {e}")
```

## Performance Considerations

- **File Size Limits**: Files > 10MB are skipped for content analysis
- **Ignore Patterns**: System files, temp files, and build artifacts are ignored
- **Batch Processing**: Multiple files processed efficiently
- **Database Integration**: Actions tracked without blocking operations

## API Integration Examples

For use in FastAPI endpoints:

```python
from fastapi import HTTPException
from backend.organizer import FileOrganizer, create_project_structure

@app.post("/projects/{project_id}/organize")
async def organize_project_endpoint(project_id: str, auto_move: bool = False):
    try:
        # Get project from database
        with db_session() as session:
            project = get_project_by_id(session, project_id)
            if not project:
                raise HTTPException(status_code=404, detail="Project not found")
        
        # Organize project
        organizer = FileOrganizer()
        result = organizer.organize_project(
            project_path=project.path,
            auto_move=auto_move,
            min_confidence=0.3
        )
        
        return {
            "total_files": result.total_files,
            "processed_files": result.processed_files,
            "suggestions": len(result.suggestions),
            "moved_files": len(result.moved_files),
            "execution_time": result.execution_time
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/projects/create-structure")
async def create_project_structure_endpoint(
    name: str, 
    path: str, 
    template: str = "research"
):
    try:
        result = create_project_structure(path, template)
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## Testing and Validation

The system includes comprehensive testing capabilities:

```python
# Test with sample files
organizer = FileOrganizer()

# Test suggestions for different file types
test_files = [
    "experiment_data.csv",
    "analysis_script.py", 
    "results_figure.png",
    "research_paper.pdf",
    "config.yaml"
]

for filename in test_files:
    suggestion = organizer.suggest_organization(f"./test/{filename}")
    print(f"{filename} → {suggestion.suggested_category} "
          f"(confidence: {suggestion.confidence:.2f})")
```

## Best Practices

1. **Use descriptive filenames** with dates and version numbers
2. **Set appropriate confidence thresholds** for auto-organization
3. **Review suggestions** before bulk operations
4. **Track organization actions** for audit trails
5. **Customize rules** for domain-specific file types
6. **Monitor performance** for large projects
7. **Handle errors gracefully** in production environments

This integration guide provides a complete overview of the intelligent file organization system and its capabilities within the Research File Manager MVP.