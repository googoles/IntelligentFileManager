"""
Intelligent File Organization and Classification System for Research File Manager MVP.

This module provides comprehensive file organization capabilities with rule-based
classification, project template creation, and intelligent pattern matching for
research workflows.
"""

import os
import shutil
import re
import json
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Union, Any
import logging
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

from database import (
    db_session, get_project_by_id, get_files_by_project, 
    create_file, get_file_by_id, File, Project
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OrganizationAction(Enum):
    """Types of organization actions that can be performed."""
    MOVE = "move"
    COPY = "copy"
    SUGGEST = "suggest"
    SKIP = "skip"


@dataclass
class OrganizationRule:
    """Represents a file organization rule."""
    category: str
    extensions: Set[str] = field(default_factory=set)
    name_patterns: Set[str] = field(default_factory=set)
    path_patterns: Set[str] = field(default_factory=set)
    content_keywords: Set[str] = field(default_factory=set)
    priority: int = 0  # Higher priority rules are applied first


@dataclass
class OrganizationSuggestion:
    """Represents a file organization suggestion."""
    file_path: str
    current_location: str
    suggested_category: str
    suggested_path: str
    confidence: float
    reason: str
    action: OrganizationAction = OrganizationAction.SUGGEST


@dataclass
class OrganizationResult:
    """Results from an organization operation."""
    total_files: int
    processed_files: int
    suggestions: List[OrganizationSuggestion]
    moved_files: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    skipped_files: List[str] = field(default_factory=list)
    execution_time: float = 0.0


class FileOrganizer:
    """
    Intelligent file organization system with rule-based classification.
    
    Supports multiple organization strategies:
    - Extension-based classification
    - Name pattern matching
    - Content keyword detection
    - Research-specific naming conventions
    - Date-based fallback organization
    """
    
    def __init__(self):
        """Initialize the file organizer with default rules."""
        self.rules: List[OrganizationRule] = []
        self._init_default_rules()
        
        # Track already organized files to prevent re-organization
        self.organized_paths: Set[str] = set()
        
        # Research-specific patterns
        self.research_patterns = self._init_research_patterns()
        
        logger.info("FileOrganizer initialized with comprehensive rule-based classification")
    
    def _init_default_rules(self) -> None:
        """Initialize default organization rules based on MVP requirements."""
        # Documents rule
        self.rules.append(OrganizationRule(
            category="documents",
            extensions={'.pdf', '.doc', '.docx', '.txt', '.md', '.rtf', '.odt', '.tex'},
            name_patterns={'readme', 'license', 'changelog', 'manual', 'guide'},
            content_keywords={'abstract', 'introduction', 'conclusion', 'references'},
            priority=10
        ))
        
        # Data rule
        self.rules.append(OrganizationRule(
            category="data",
            extensions={'.csv', '.xlsx', '.xls', '.json', '.xml', '.yaml', '.yml', 
                       '.parquet', '.h5', '.hdf5', '.sqlite', '.db'},
            name_patterns={'dataset', 'data', 'raw', 'processed', 'clean'},
            path_patterns={'data', 'datasets', 'raw', 'processed'},
            priority=10
        ))
        
        # Code rule
        self.rules.append(OrganizationRule(
            category="code",
            extensions={'.py', '.js', '.r', '.ipynb', '.java', '.cpp', '.c', '.h', 
                       '.cs', '.php', '.go', '.rs', '.swift', '.kt', '.scala', '.sql'},
            name_patterns={'script', 'notebook', 'analysis', 'model', 'pipeline'},
            path_patterns={'src', 'scripts', 'notebooks', 'code'},
            content_keywords={'import', 'function', 'class', 'def', 'var', 'let'},
            priority=10
        ))
        
        # Images rule
        self.rules.append(OrganizationRule(
            category="images",
            extensions={'.png', '.jpg', '.jpeg', '.gif', '.svg', '.bmp', '.tiff', 
                       '.webp', '.ico', '.eps', '.ps'},
            name_patterns={'figure', 'plot', 'chart', 'graph', 'image', 'photo'},
            path_patterns={'figures', 'images', 'plots', 'graphics'},
            priority=10
        ))
        
        # Results rule with high priority for research files
        self.rules.append(OrganizationRule(
            category="results",
            extensions={'.png', '.jpg', '.pdf', '.svg', '.html', '.txt', '.csv'},
            name_patterns={'result', 'output', 'figure', 'plot', 'chart', 'graph', 
                          'analysis', 'report', 'summary', 'conclusion'},
            path_patterns={'results', 'output', 'figures', 'plots', 'reports'},
            content_keywords={'results', 'findings', 'conclusion', 'summary'},
            priority=15  # Higher priority than general categories
        ))
        
        # Configuration files
        self.rules.append(OrganizationRule(
            category="config",
            extensions={'.conf', '.config', '.ini', '.cfg', '.toml', '.env'},
            name_patterns={'config', 'settings', 'configuration'},
            priority=5
        ))
        
        # Archive files
        self.rules.append(OrganizationRule(
            category="archives",
            extensions={'.zip', '.rar', '.7z', '.tar', '.gz', '.bz2', '.xz'},
            priority=5
        ))
    
    def _init_research_patterns(self) -> Dict[str, List[str]]:
        """Initialize research-specific filename patterns."""
        return {
            'experiment': [
                r'exp\d+', r'experiment_\d+', r'run_\d+', r'trial_\d+',
                r'test_\d+', r'batch_\d+'
            ],
            'version': [
                r'v\d+(\.\d+)*', r'version_\d+', r'_v\d+', r'rev\d+'
            ],
            'date': [
                r'\d{4}-\d{2}-\d{2}', r'\d{8}', r'\d{2}_\d{2}_\d{4}',
                r'\d{4}\d{2}\d{2}', r'\d{2}-\d{2}-\d{4}'
            ],
            'analysis': [
                r'analysis', r'analyze', r'stats', r'statistics',
                r'correlation', r'regression', r'model'
            ],
            'draft': [
                r'draft', r'rough', r'temp', r'tmp', r'preliminary',
                r'working', r'wip'
            ]
        }
    
    def add_custom_rule(self, rule: OrganizationRule) -> None:
        """
        Add a custom organization rule.
        
        Args:
            rule: Organization rule to add
        """
        self.rules.append(rule)
        # Sort rules by priority (highest first)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
        logger.info(f"Added custom rule for category: {rule.category}")
    
    def suggest_organization(self, file_path: str) -> OrganizationSuggestion:
        """
        Suggest organization for a single file.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            Organization suggestion with category and confidence
        """
        try:
            file_path = os.path.abspath(file_path)
            
            if not os.path.exists(file_path):
                return OrganizationSuggestion(
                    file_path=file_path,
                    current_location=os.path.dirname(file_path),
                    suggested_category="error",
                    suggested_path="",
                    confidence=0.0,
                    reason="File does not exist",
                    action=OrganizationAction.SKIP
                )
            
            # Check if file is already in an organized location
            if self._is_already_organized(file_path):
                return OrganizationSuggestion(
                    file_path=file_path,
                    current_location=os.path.dirname(file_path),
                    suggested_category="already_organized",
                    suggested_path=file_path,
                    confidence=1.0,
                    reason="File is already in an organized folder",
                    action=OrganizationAction.SKIP
                )
            
            # Analyze file for best category
            file_name = os.path.basename(file_path).lower()
            file_ext = os.path.splitext(file_path)[1].lower()
            parent_dir = os.path.basename(os.path.dirname(file_path)).lower()
            
            # Try to read content for text files (limited)
            content = self._extract_limited_content(file_path)
            
            # Find best matching rule
            best_category = None
            best_confidence = 0.0
            best_reason = ""
            
            for rule in self.rules:
                confidence, reason = self._calculate_rule_confidence(
                    rule, file_name, file_ext, parent_dir, content
                )
                
                if confidence > best_confidence:
                    best_category = rule.category
                    best_confidence = confidence
                    best_reason = reason
            
            # Apply research-specific pattern matching
            research_category, research_confidence, research_reason = self._apply_research_patterns(
                file_name, file_path
            )
            
            if research_confidence > best_confidence:
                best_category = research_category
                best_confidence = research_confidence
                best_reason = research_reason
            
            # Fallback to date-based organization if confidence is low
            if best_confidence < 0.3:
                best_category = self._generate_date_fallback_category(file_path)
                best_confidence = 0.2
                best_reason = "Date-based fallback organization"
            
            return OrganizationSuggestion(
                file_path=file_path,
                current_location=os.path.dirname(file_path),
                suggested_category=best_category or "unsorted",
                suggested_path="",  # Will be set by caller
                confidence=best_confidence,
                reason=best_reason,
                action=OrganizationAction.SUGGEST
            )
            
        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {e}")
            return OrganizationSuggestion(
                file_path=file_path,
                current_location=os.path.dirname(file_path),
                suggested_category="error",
                suggested_path="",
                confidence=0.0,
                reason=f"Error during analysis: {str(e)}",
                action=OrganizationAction.SKIP
            )
    
    def organize_project(self, 
                        project_path: str, 
                        auto_move: bool = False,
                        dry_run: bool = False,
                        min_confidence: float = 0.3) -> OrganizationResult:
        """
        Organize all files in a project directory.
        
        Args:
            project_path: Root path of the project
            auto_move: Whether to automatically move files
            dry_run: If True, only generate suggestions without moving
            min_confidence: Minimum confidence threshold for auto-move
            
        Returns:
            Organization result with statistics and suggestions
        """
        start_time = datetime.now()
        
        try:
            project_path = os.path.abspath(project_path)
            
            if not os.path.exists(project_path):
                raise ValueError(f"Project path does not exist: {project_path}")
            
            result = OrganizationResult(
                total_files=0,
                processed_files=0,
                suggestions=[]
            )
            
            # Collect all files to process
            all_files = []
            for root, dirs, files in os.walk(project_path):
                # Skip already organized directories
                dirs[:] = [d for d in dirs if not self._is_organized_directory(os.path.join(root, d))]
                
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    if not self._should_ignore_file(file_path):
                        all_files.append(file_path)
            
            result.total_files = len(all_files)
            logger.info(f"Found {result.total_files} files to analyze in {project_path}")
            
            # Process each file
            category_counts = defaultdict(int)
            
            for file_path in all_files:
                try:
                    suggestion = self.suggest_organization(file_path)
                    
                    if suggestion.action == OrganizationAction.SKIP:
                        result.skipped_files.append(file_path)
                        continue
                    
                    # Set suggested path
                    suggestion.suggested_path = os.path.join(
                        project_path, 
                        suggestion.suggested_category,
                        os.path.basename(file_path)
                    )
                    
                    result.suggestions.append(suggestion)
                    category_counts[suggestion.suggested_category] += 1
                    
                    # Auto-move if requested and confidence is high enough
                    if (auto_move and 
                        not dry_run and 
                        suggestion.confidence >= min_confidence and
                        suggestion.suggested_category not in ["error", "already_organized"]):
                        
                        if self._execute_move(suggestion):
                            result.moved_files.append(file_path)
                            suggestion.action = OrganizationAction.MOVE
                    
                    result.processed_files += 1
                    
                except Exception as e:
                    error_msg = f"Error processing {file_path}: {str(e)}"
                    result.errors.append(error_msg)
                    logger.error(error_msg)
            
            result.execution_time = (datetime.now() - start_time).total_seconds()
            
            # Log summary
            logger.info(f"Organization complete: {result.processed_files}/{result.total_files} files processed")
            logger.info(f"Categories: {dict(category_counts)}")
            if auto_move and not dry_run:
                logger.info(f"Moved {len(result.moved_files)} files automatically")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during project organization: {e}")
            result = OrganizationResult(0, 0, [])
            result.errors.append(str(e))
            result.execution_time = (datetime.now() - start_time).total_seconds()
            return result
    
    def _calculate_rule_confidence(self, 
                                  rule: OrganizationRule,
                                  file_name: str,
                                  file_ext: str,
                                  parent_dir: str,
                                  content: str) -> Tuple[float, str]:
        """Calculate confidence score for a rule match."""
        confidence = 0.0
        reasons = []
        
        # Extension matching (high confidence)
        if file_ext in rule.extensions:
            confidence += 0.8
            reasons.append(f"Extension '{file_ext}' matches category")
        
        # Name pattern matching (medium confidence)
        for pattern in rule.name_patterns:
            if pattern in file_name:
                confidence += 0.6
                reasons.append(f"Name contains '{pattern}'")
                break
        
        # Path pattern matching (medium confidence)
        for pattern in rule.path_patterns:
            if pattern in parent_dir:
                confidence += 0.5
                reasons.append(f"In directory containing '{pattern}'")
                break
        
        # Content keyword matching (lower confidence, but valuable)
        if content and rule.content_keywords:
            matched_keywords = []
            content_lower = content.lower()
            for keyword in rule.content_keywords:
                if keyword in content_lower:
                    matched_keywords.append(keyword)
            
            if matched_keywords:
                confidence += 0.4 * min(len(matched_keywords) / len(rule.content_keywords), 1.0)
                reasons.append(f"Content contains keywords: {', '.join(matched_keywords[:3])}")
        
        # Cap confidence at 1.0
        confidence = min(confidence, 1.0)
        
        return confidence, "; ".join(reasons)
    
    def _apply_research_patterns(self, file_name: str, file_path: str) -> Tuple[str, float, str]:
        """Apply research-specific pattern matching."""
        confidence = 0.0
        category = ""
        reason = ""
        
        # Check for research-specific patterns
        for pattern_type, patterns in self.research_patterns.items():
            for pattern in patterns:
                if re.search(pattern, file_name, re.IGNORECASE):
                    if pattern_type == 'experiment':
                        category = "results"
                        confidence = 0.7
                        reason = f"Experiment file pattern: {pattern}"
                    elif pattern_type == 'analysis':
                        category = "code"
                        confidence = 0.6
                        reason = f"Analysis file pattern: {pattern}"
                    elif pattern_type == 'draft':
                        category = "drafts"
                        confidence = 0.5
                        reason = f"Draft file pattern: {pattern}"
                    break
            
            if confidence > 0:
                break
        
        return category, confidence, reason
    
    def _extract_limited_content(self, file_path: str, max_bytes: int = 1024) -> str:
        """Extract limited content from text files for analysis."""
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            text_extensions = {'.txt', '.md', '.py', '.js', '.r', '.json', '.csv', '.xml'}
            
            if file_ext not in text_extensions:
                return ""
            
            # Check file size
            if os.path.getsize(file_path) > 10 * 1024 * 1024:  # 10MB limit
                return ""
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(max_bytes)
                return content.lower()
        
        except Exception:
            return ""
    
    def _is_already_organized(self, file_path: str) -> bool:
        """Check if file is already in an organized location."""
        path_parts = Path(file_path).parts
        
        # Check if any part of the path is a known category
        known_categories = {rule.category for rule in self.rules}
        known_categories.update(['literature', 'drafts', 'notes', 'raw', 'processed'])
        
        for part in path_parts:
            if part.lower() in known_categories:
                return True
        
        return False
    
    def _is_organized_directory(self, dir_path: str) -> bool:
        """Check if directory is already organized (should be skipped)."""
        dir_name = os.path.basename(dir_path).lower()
        
        # Known organized directory names
        organized_dirs = {
            'documents', 'data', 'code', 'images', 'results', 'archives', 'config',
            'literature', 'drafts', 'notes', 'raw', 'processed', 'figures', 'tables',
            'input', 'output', 'workspace'
        }
        
        return dir_name in organized_dirs
    
    def _should_ignore_file(self, file_path: str) -> bool:
        """Check if file should be ignored during organization."""
        file_name = os.path.basename(file_path)
        
        # System and temporary files
        ignore_patterns = [
            '.DS_Store', 'Thumbs.db', 'desktop.ini',
            '.tmp', '.temp', '~$', '.swp', '.swo',
            '.gitkeep', '.gitignore', '.git',
            '__pycache__', '.pytest_cache',
            'README.md'  # Keep README in root
        ]
        
        for pattern in ignore_patterns:
            if pattern in file_name or file_name.startswith(pattern):
                return True
        
        # Very small files (likely empty)
        try:
            if os.path.getsize(file_path) == 0:
                return True
        except OSError:
            return True
        
        return False
    
    def _generate_date_fallback_category(self, file_path: str) -> str:
        """Generate date-based fallback category."""
        try:
            # Try to get file modification time
            mtime = os.path.getmtime(file_path)
            file_date = datetime.fromtimestamp(mtime)
            return f"unsorted_{file_date.strftime('%Y%m')}"
        except Exception:
            # Fall back to current date
            return f"unsorted_{datetime.now().strftime('%Y%m')}"
    
    def _execute_move(self, suggestion: OrganizationSuggestion) -> bool:
        """
        Execute file move operation.
        
        Args:
            suggestion: Organization suggestion to execute
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create destination directory
            dest_dir = os.path.dirname(suggestion.suggested_path)
            os.makedirs(dest_dir, exist_ok=True)
            
            # Handle name conflicts
            final_path = suggestion.suggested_path
            counter = 1
            while os.path.exists(final_path):
                name, ext = os.path.splitext(suggestion.suggested_path)
                final_path = f"{name}_{counter}{ext}"
                counter += 1
            
            # Move the file
            shutil.move(suggestion.file_path, final_path)
            self.organized_paths.add(final_path)
            
            logger.info(f"📁 Moved: {suggestion.file_path} → {final_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to move {suggestion.file_path}: {e}")
            return False


def create_project_structure(project_path: str, 
                           template: str = 'research',
                           custom_folders: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Create structured project folders with README generation.
    
    Args:
        project_path: Root path for the project
        template: Template type ('research', 'minimal', or 'custom')
        custom_folders: Custom folder structure (used with 'custom' template)
        
    Returns:
        Dictionary with creation results and folder information
    """
    try:
        project_path = os.path.abspath(project_path)
        os.makedirs(project_path, exist_ok=True)
        
        # Define templates
        templates = {
            'research': [
                'literature',
                'data/raw',
                'data/processed', 
                'code',
                'results/figures',
                'results/tables',
                'drafts',
                'notes'
            ],
            'minimal': [
                'input',
                'output',
                'workspace'
            ],
            'data_science': [
                'data/raw',
                'data/processed',
                'data/external',
                'notebooks',
                'src',
                'models',
                'reports/figures',
                'reports/tables',
                'config'
            ],
            'software_dev': [
                'src',
                'tests',
                'docs',
                'examples',
                'data',
                'config',
                'scripts'
            ]
        }
        
        # Select folder structure
        if template == 'custom' and custom_folders:
            folders = custom_folders
        elif template in templates:
            folders = templates[template]
        else:
            logger.warning(f"Unknown template '{template}', using 'minimal'")
            folders = templates['minimal']
        
        # Create folders
        created_folders = []
        for folder in folders:
            folder_path = os.path.join(project_path, folder)
            os.makedirs(folder_path, exist_ok=True)
            created_folders.append(folder_path)
            logger.info(f"📁 Created: {folder_path}")
        
        # Generate README.md with project structure description
        readme_content = generate_project_readme(
            project_name=os.path.basename(project_path),
            template=template,
            folders=folders,
            created_date=datetime.now()
        )
        
        readme_path = os.path.join(project_path, 'README.md')
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        logger.info(f"📄 Generated README.md at {readme_path}")
        
        # Create .gitkeep files in empty directories
        for folder_path in created_folders:
            gitkeep_path = os.path.join(folder_path, '.gitkeep')
            if not os.path.exists(gitkeep_path):
                with open(gitkeep_path, 'w') as f:
                    f.write("# This file keeps the directory in git\n")
        
        result = {
            'project_path': project_path,
            'template': template,
            'folders_created': len(created_folders),
            'folder_list': folders,
            'readme_generated': True,
            'created_at': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Project structure created: {len(created_folders)} folders")
        return result
        
    except Exception as e:
        logger.error(f"Error creating project structure: {e}")
        return {
            'project_path': project_path,
            'template': template,
            'folders_created': 0,
            'error': str(e),
            'created_at': datetime.now().isoformat()
        }


def generate_project_readme(project_name: str, 
                          template: str,
                          folders: List[str],
                          created_date: datetime) -> str:
    """
    Generate README.md content for a project.
    
    Args:
        project_name: Name of the project
        template: Template used
        folders: List of folders created
        created_date: When the project was created
        
    Returns:
        README.md content as string
    """
    # Folder descriptions based on common research patterns
    folder_descriptions = {
        'literature': 'Research papers, articles, and reference materials',
        'data': 'Data storage directory',
        'data/raw': 'Original, immutable data dump',
        'data/processed': 'Cleaned and processed datasets ready for analysis',
        'data/external': 'External datasets and third-party data',
        'code': 'Source code and scripts',
        'src': 'Source code and main application files',
        'notebooks': 'Jupyter notebooks for exploration and analysis',
        'results': 'Analysis results and outputs',
        'results/figures': 'Generated plots, charts, and visualizations',
        'results/tables': 'Summary tables and statistical outputs',
        'drafts': 'Working drafts and preliminary documents',
        'notes': 'Research notes, ideas, and documentation',
        'input': 'Input files and data',
        'output': 'Generated outputs and results',
        'workspace': 'Working directory for temporary files',
        'models': 'Trained models and model artifacts',
        'reports': 'Final reports and presentations',
        'config': 'Configuration files and settings',
        'tests': 'Unit tests and test data',
        'docs': 'Documentation and guides',
        'examples': 'Example code and usage demonstrations',
        'scripts': 'Utility scripts and automation'
    }
    
    readme_content = f"""# {project_name}

**Created:** {created_date.strftime('%Y-%m-%d %H:%M')}  
**Template:** {template}  
**Auto-generated by:** Research File Manager

## Project Structure

This project follows the `{template}` template for organized research workflows.

"""
    
    # Add folder structure with descriptions
    for folder in sorted(folders):
        description = folder_descriptions.get(folder, 'Project directory')
        if '/' in folder:
            # Indent subdirectories
            indent = "  " * (folder.count('/'))
            folder_name = folder.split('/')[-1]
            readme_content += f"{indent}- **{folder_name}/** - {description}\n"
        else:
            readme_content += f"- **{folder}/** - {description}\n"
    
    # Add usage guidelines
    readme_content += f"""

## Usage Guidelines

### File Organization
- Files are automatically organized based on type and content
- Use descriptive filenames with dates when applicable
- Avoid spaces in filenames; use underscores or hyphens instead

### Recommended Naming Conventions
- **Data files:** `YYYY-MM-DD_dataset_description.ext`
- **Analysis scripts:** `YYYY-MM-DD_analysis_description.py`  
- **Results:** `YYYY-MM-DD_result_description.ext`
- **Drafts:** `YYYY-MM-DD_draft_title.md`

### Getting Started
1. Place raw data in the appropriate data directory
2. Create analysis scripts in the code/notebooks directory
3. Save results and outputs to the results directory
4. Use the drafts folder for work-in-progress documents

### File Monitoring
This project is monitored by the Research File Manager. New files will be:
- Automatically indexed for semantic search
- Suggested for organization based on content and type
- Tracked for version control and collaboration

---
*This README was auto-generated. You can safely edit it to add project-specific information.*
"""

    return readme_content


def analyze_project_organization(project_path: str) -> Dict[str, Any]:
    """
    Analyze current organization status of a project.
    
    Args:
        project_path: Path to the project directory
        
    Returns:
        Analysis results with organization statistics
    """
    try:
        project_path = os.path.abspath(project_path)
        
        if not os.path.exists(project_path):
            raise ValueError(f"Project path does not exist: {project_path}")
        
        analysis = {
            'project_path': project_path,
            'total_files': 0,
            'organized_files': 0,
            'unorganized_files': 0,
            'empty_directories': [],
            'file_types': defaultdict(int),
            'organization_categories': defaultdict(int),
            'large_files': [],  # Files > 100MB
            'duplicate_candidates': [],  # Files with same name
            'recommendations': []
        }
        
        organizer = FileOrganizer()
        file_names = defaultdict(list)
        
        # Walk through all files
        for root, dirs, files in os.walk(project_path):
            # Check for empty directories
            if not files and not dirs:
                analysis['empty_directories'].append(root)
            
            for file_name in files:
                file_path = os.path.join(root, file_name)
                
                # Skip system files
                if organizer._should_ignore_file(file_path):
                    continue
                
                analysis['total_files'] += 1
                
                # File extension analysis
                ext = os.path.splitext(file_name)[1].lower()
                analysis['file_types'][ext or 'no_extension'] += 1
                
                # Check file size
                try:
                    size = os.path.getsize(file_path)
                    if size > 100 * 1024 * 1024:  # 100MB
                        analysis['large_files'].append({
                            'path': file_path,
                            'size_mb': round(size / (1024 * 1024), 2)
                        })
                except OSError:
                    continue
                
                # Track potential duplicates
                file_names[file_name].append(file_path)
                
                # Check if file is organized
                if organizer._is_already_organized(file_path):
                    analysis['organized_files'] += 1
                    # Determine category
                    path_parts = Path(file_path).parts
                    for part in path_parts:
                        if part.lower() in {'documents', 'data', 'code', 'images', 'results', 
                                          'literature', 'drafts', 'notes'}:
                            analysis['organization_categories'][part.lower()] += 1
                            break
                else:
                    analysis['unorganized_files'] += 1
        
        # Find duplicate candidates
        for name, paths in file_names.items():
            if len(paths) > 1:
                analysis['duplicate_candidates'].append({
                    'filename': name,
                    'locations': paths
                })
        
        # Generate recommendations
        if analysis['unorganized_files'] > 0:
            analysis['recommendations'].append(
                f"Consider organizing {analysis['unorganized_files']} unorganized files"
            )
        
        if len(analysis['empty_directories']) > 0:
            analysis['recommendations'].append(
                f"Remove {len(analysis['empty_directories'])} empty directories"
            )
        
        if len(analysis['duplicate_candidates']) > 0:
            analysis['recommendations'].append(
                f"Review {len(analysis['duplicate_candidates'])} potential duplicate files"
            )
        
        if len(analysis['large_files']) > 0:
            analysis['recommendations'].append(
                f"Consider archiving {len(analysis['large_files'])} large files"
            )
        
        # Calculate organization percentage
        if analysis['total_files'] > 0:
            org_percentage = (analysis['organized_files'] / analysis['total_files']) * 100
            analysis['organization_percentage'] = round(org_percentage, 1)
        else:
            analysis['organization_percentage'] = 0.0
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing project organization: {e}")
        return {
            'project_path': project_path,
            'error': str(e),
            'total_files': 0
        }


# Integration with database for tracking organization actions
def track_organization_action(project_id: str, action_type: str, 
                            file_path: str, details: Dict[str, Any]) -> None:
    """
    Track organization actions in the database.
    
    Args:
        project_id: Project ID
        action_type: Type of action (move, copy, suggest)
        file_path: File path affected
        details: Additional details about the action
    """
    try:
        with db_session() as session:
            # This could be extended to create an OrganizationLog table
            # For now, we can add metadata to the file record
            files = session.query(File).filter(
                File.project_id == project_id,
                File.path == file_path
            ).all()
            
            for file_record in files:
                if not file_record.metadata:
                    file_record.metadata = {}
                
                if 'organization_history' not in file_record.metadata:
                    file_record.metadata['organization_history'] = []
                
                file_record.metadata['organization_history'].append({
                    'action': action_type,
                    'timestamp': datetime.now().isoformat(),
                    'details': details
                })
                
                # Update the metadata JSON field
                session.add(file_record)
            
            logger.debug(f"Tracked organization action: {action_type} for {file_path}")
        
    except Exception as e:
        logger.error(f"Error tracking organization action: {e}")


def get_project_organization_stats(project_id: str) -> Dict[str, Any]:
    """
    Get organization statistics for a project from the database.
    
    Args:
        project_id: Project ID
        
    Returns:
        Dictionary with organization statistics
    """
    try:
        with db_session() as session:
            project = get_project_by_id(session, project_id)
            if not project:
                return {'error': 'Project not found'}
            
            files = get_files_by_project(session, project_id)
            
            stats = {
                'project_id': project_id,
                'project_name': project.name,
                'project_path': project.path,
                'total_files': len(files),
                'file_types': defaultdict(int),
                'organized_files': 0,
                'has_content': 0,
                'has_embeddings': 0,
                'organization_actions': 0
            }
            
            for file_record in files:
                # Count file types
                stats['file_types'][file_record.type] += 1
                
                # Check if file has content
                if file_record.content:
                    stats['has_content'] += 1
                
                # Check if file has embeddings
                if file_record.embedding:
                    stats['has_embeddings'] += 1
                
                # Check organization history
                if (file_record.metadata and 
                    'organization_history' in file_record.metadata):
                    stats['organization_actions'] += len(
                        file_record.metadata['organization_history']
                    )
                
                # Check if file appears to be organized (in subfolder)
                path_parts = Path(file_record.path).parts
                if len(path_parts) > 2:  # More than just project/file
                    stats['organized_files'] += 1
            
            # Convert defaultdict to regular dict
            stats['file_types'] = dict(stats['file_types'])
            
            # Calculate percentages
            if stats['total_files'] > 0:
                stats['organized_percentage'] = round(
                    (stats['organized_files'] / stats['total_files']) * 100, 1
                )
                stats['content_percentage'] = round(
                    (stats['has_content'] / stats['total_files']) * 100, 1
                )
                stats['embedding_percentage'] = round(
                    (stats['has_embeddings'] / stats['total_files']) * 100, 1
                )
            else:
                stats['organized_percentage'] = 0.0
                stats['content_percentage'] = 0.0
                stats['embedding_percentage'] = 0.0
            
            return stats
            
    except Exception as e:
        logger.error(f"Error getting organization stats for project {project_id}: {e}")
        return {'error': str(e)}


def integrate_with_file_watcher(organizer: FileOrganizer, 
                              project_id: str, 
                              file_path: str,
                              auto_organize: bool = False) -> Optional[OrganizationSuggestion]:
    """
    Integration point for file watcher to get organization suggestions.
    
    Args:
        organizer: FileOrganizer instance
        project_id: Project ID
        file_path: Path to the new/modified file
        auto_organize: Whether to automatically move the file
        
    Returns:
        Organization suggestion or None if error
    """
    try:
        suggestion = organizer.suggest_organization(file_path)
        
        # Track the suggestion in database
        track_organization_action(
            project_id=project_id,
            action_type="suggest",
            file_path=file_path,
            details={
                'suggested_category': suggestion.suggested_category,
                'confidence': suggestion.confidence,
                'reason': suggestion.reason
            }
        )
        
        # Auto-organize if requested and confidence is high
        if (auto_organize and 
            suggestion.confidence >= 0.7 and
            suggestion.action != OrganizationAction.SKIP):
            
            # Set the suggested path within the project
            with db_session() as session:
                project = get_project_by_id(session, project_id)
                if project:
                    suggestion.suggested_path = os.path.join(
                        project.path,
                        suggestion.suggested_category,
                        os.path.basename(file_path)
                    )
                    
                    # Execute the move
                    if organizer._execute_move(suggestion):
                        suggestion.action = OrganizationAction.MOVE
                        
                        # Track the move action
                        track_organization_action(
                            project_id=project_id,
                            action_type="move",
                            file_path=suggestion.suggested_path,  # New path
                            details={
                                'original_path': file_path,
                                'category': suggestion.suggested_category,
                                'auto_moved': True
                            }
                        )
        
        return suggestion
        
    except Exception as e:
        logger.error(f"Error in file watcher integration: {e}")
        return None


if __name__ == "__main__":
    # Example usage and testing
    print("🔬 Testing File Organizer...")
    
    # Create a test organizer
    organizer = FileOrganizer()
    
    # Test project structure creation
    test_project_path = "./test_project_structure"
    result = create_project_structure(test_project_path, "research")
    print(f"Created project structure: {result}")
    
    # Test file organization suggestions
    if os.path.exists("./test_data"):
        suggestions = []
        for root, dirs, files in os.walk("./test_data"):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                suggestion = organizer.suggest_organization(file_path)
                suggestions.append(suggestion)
                print(f"📄 {file_name} → {suggestion.suggested_category} "
                      f"(confidence: {suggestion.confidence:.2f})")
    
    # Test project analysis
    if os.path.exists("./test_project_structure"):
        analysis = analyze_project_organization("./test_project_structure")
        print(f"\nProject Analysis: {analysis['organization_percentage']}% organized")
        print(f"Recommendations: {analysis['recommendations']}")
    
    print("✅ File Organizer testing complete!")