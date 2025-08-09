"""
Test script for the semantic search functionality.

This script tests the SemanticSearch class and its integration with the database models.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.database import init_database, create_project, create_file, db_session
from backend.search import SemanticSearch, initialize_semantic_search, index_project_files, search_files
from backend.config import TestingConfig


def test_semantic_search():
    """Test the semantic search functionality."""
    print("🔍 Testing Semantic Search Functionality")
    print("=" * 50)
    
    # Create temporary directory for test data
    test_dir = tempfile.mkdtemp(prefix="semantic_search_test_")
    print(f"Test directory: {test_dir}")
    
    try:
        # Override config for testing
        original_config = os.environ.get('ENVIRONMENT')
        os.environ['ENVIRONMENT'] = 'testing'
        
        # Set custom paths for testing
        chroma_path = os.path.join(test_dir, 'chroma_test')
        
        # Initialize database with test configuration
        print("\n1. Initializing database...")
        db_manager = init_database('sqlite:///:memory:')
        
        # Initialize semantic search
        print("2. Initializing semantic search...")
        search_engine = SemanticSearch(
            model_name='all-MiniLM-L6-v2',
            chroma_path=chroma_path,
            collection_name='test_embeddings'
        )
        
        # Test data
        test_project_data = {
            'name': 'AI Research Project',
            'path': os.path.join(test_dir, 'ai_project')
        }
        
        test_files_data = [
            {
                'name': 'introduction_to_ml.md',
                'content': '''# Introduction to Machine Learning

Machine learning is a powerful subset of artificial intelligence (AI) that enables 
computers to learn and make decisions from data without being explicitly programmed 
for every scenario. It involves the development of algorithms and statistical models 
that can identify patterns, make predictions, and improve their performance over time.

## Key Concepts
- Supervised learning uses labeled data to train models
- Unsupervised learning finds patterns in unlabeled data  
- Reinforcement learning learns through interaction with an environment

Machine learning has applications in computer vision, natural language processing, 
robotics, and many other domains.'''
            },
            {
                'name': 'deep_learning_overview.txt',
                'content': '''Deep Learning: A Comprehensive Overview

Deep learning is a specialized branch of machine learning that uses artificial neural 
networks with multiple layers (hence "deep") to model and understand complex patterns 
in data. These neural networks are inspired by the structure and function of the human brain.

Key characteristics of deep learning:
1. Uses multiple layers of artificial neurons
2. Can automatically learn hierarchical feature representations  
3. Particularly effective for image recognition, speech processing, and natural language tasks
4. Requires large amounts of training data and computational resources

Popular deep learning architectures include:
- Convolutional Neural Networks (CNNs) for image analysis
- Recurrent Neural Networks (RNNs) for sequence data
- Transformers for natural language processing

Deep learning has revolutionized fields like computer vision, speech recognition, 
and language translation, achieving human-level or superhuman performance in many tasks.'''
            },
            {
                'name': 'nlp_fundamentals.py',
                'content': '''"""
Natural Language Processing Fundamentals

This module covers the basic concepts and techniques used in NLP.
"""

import nltk
import spacy
from transformers import pipeline

def tokenization_example():
    """Demonstrate text tokenization techniques."""
    text = "Natural language processing enables computers to understand human language."
    
    # Word tokenization
    tokens = text.split()
    print("Tokens:", tokens)
    
    # Sentence tokenization  
    sentences = text.split('.')
    print("Sentences:", sentences)

def named_entity_recognition():
    """Example of named entity recognition using spaCy."""
    nlp = spacy.load("en_core_web_sm")
    text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
    
    doc = nlp(text)
    for ent in doc.ents:
        print(f"Entity: {ent.text}, Label: {ent.label_}")

def sentiment_analysis():
    """Perform sentiment analysis using transformers."""
    classifier = pipeline("sentiment-analysis")
    
    texts = [
        "I love machine learning!",
        "This algorithm is terrible.",
        "The results are quite interesting."
    ]
    
    for text in texts:
        result = classifier(text)
        print(f"Text: {text}")
        print(f"Sentiment: {result[0]['label']}, Score: {result[0]['score']:.3f}")

# Natural language processing combines computational linguistics with machine learning
# to enable computers to process, analyze, and understand human language in a meaningful way.
'''
            },
            {
                'name': 'computer_vision_basics.md',
                'content': '''# Computer Vision Fundamentals

Computer vision is a field of artificial intelligence that trains computers to interpret 
and understand visual information from the world. It seeks to automate tasks that the 
human visual system can do, such as recognizing objects, faces, or activities in images and videos.

## Core Concepts

### Image Processing
- Image filtering and enhancement
- Edge detection and feature extraction
- Color space transformations
- Noise reduction techniques

### Object Detection and Recognition
- Template matching methods
- Feature-based approaches (SIFT, SURF, ORB)
- Deep learning-based detection (YOLO, R-CNN)
- Facial recognition systems

### Image Segmentation  
- Thresholding techniques
- Region growing algorithms
- Clustering-based segmentation
- Semantic and instance segmentation with neural networks

## Applications
- Autonomous vehicles and navigation systems
- Medical image analysis and diagnosis
- Quality control in manufacturing
- Augmented reality and virtual reality
- Security and surveillance systems

Computer vision leverages machine learning algorithms, particularly deep learning 
with convolutional neural networks, to achieve high accuracy in visual recognition tasks.'''
            },
            {
                'name': 'research_notes.txt',
                'content': '''Research Notes - AI/ML Literature Review

Date: 2024-01-15

## Paper: "Attention Is All You Need" (Vaswani et al., 2017)
- Introduces the Transformer architecture
- Revolutionary approach to sequence-to-sequence tasks
- Uses self-attention mechanism instead of recurrence
- Foundation for modern language models like BERT, GPT

## Paper: "ImageNet Classification with Deep CNNs" (Krizhevsky et al., 2012)  
- AlexNet - breakthrough in computer vision
- Demonstrated effectiveness of deep convolutional networks
- Won ImageNet competition with significant margin
- Sparked the deep learning revolution

## Current Research Trends
1. Foundation models and large language models
2. Multimodal AI (text + vision + audio)
3. Few-shot and zero-shot learning
4. Explainable AI and model interpretability
5. Efficient model architectures (MobileNets, EfficientNets)

## Research Questions
- How can we make AI models more efficient and sustainable?
- What are the limits of current transformer architectures?
- How can we improve AI safety and alignment?

Next steps: Investigate recent work on parameter-efficient fine-tuning methods.'''
            }
        ]
        
        # Create test project and files
        print("\n3. Creating test project and files...")
        with db_session() as session:
            project = create_project(
                session,
                name=test_project_data['name'],
                path=test_project_data['path']
            )
            print(f"   Created project: {project.name}")
            
            file_records = []
            for file_data in test_files_data:
                file_record = create_file(
                    session,
                    project_id=project.id,
                    path=os.path.join(test_project_data['path'], file_data['name']),
                    name=file_data['name'],
                    file_type=Path(file_data['name']).suffix or '.txt',
                    content=file_data['content']
                )
                file_records.append(file_record)
                print(f"   Created file: {file_record.name}")
        
        # Test text chunking
        print("\n4. Testing text chunking...")
        sample_content = test_files_data[0]['content']
        chunks = search_engine._split_text(sample_content)
        print(f"   Split {len(sample_content)} characters into {len(chunks)} chunks")
        if chunks:
            print(f"   First chunk: {chunks[0].content[:100]}...")
            print(f"   Chunk size: {len(chunks[0].content)} characters")
        
        # Test embedding generation  
        print("\n5. Testing embedding generation...")
        sample_texts = [chunk.content for chunk in chunks[:3]]
        embeddings = search_engine._generate_embeddings(sample_texts)
        print(f"   Generated {len(embeddings)} embeddings")
        print(f"   Embedding dimension: {embeddings[0].shape if len(embeddings) > 0 else 'N/A'}")
        
        # Test file indexing
        print("\n6. Testing file indexing...")
        indexing_stats = search_engine.batch_index_files(file_records)
        print(f"   Indexing statistics:")
        for key, value in indexing_stats.items():
            print(f"     {key}: {value}")
        
        # Test semantic search
        print("\n7. Testing semantic search...")
        test_queries = [
            "machine learning algorithms",
            "neural networks and deep learning", 
            "natural language processing",
            "computer vision and image recognition",
            "transformer architecture attention",
            "research papers and literature"
        ]
        
        for query in test_queries:
            results = search_engine.search(query, top_k=3)
            print(f"\n   Query: '{query}'")
            
            if results:
                for i, result in enumerate(results, 1):
                    print(f"     {i}. {result.file_name} (similarity: {result.similarity_score:.3f})")
                    print(f"        Snippet: {result.content_snippet[:150]}...")
            else:
                print("     No results found")
        
        # Test project-scoped search
        print("\n8. Testing project-scoped search...")
        project_results = search_engine.search(
            "artificial intelligence", 
            project_id=project.id, 
            top_k=5
        )
        print(f"   Found {len(project_results)} results in project scope")
        
        # Test file type filtering
        print("\n9. Testing file type filtering...")
        code_results = search_engine.search(
            "natural language processing",
            file_types=['.py'],
            top_k=3
        )
        print(f"   Found {len(code_results)} Python files matching query")
        
        # Test similar file finding
        print("\n10. Testing similar file discovery...")
        if file_records:
            similar_files = search_engine.find_similar_files(
                file_records[0].id,
                top_k=3,
                similarity_threshold=0.1
            )
            print(f"    Files similar to '{file_records[0].name}':")
            for result in similar_files:
                print(f"      {result.file_name} (similarity: {result.similarity_score:.3f})")
        
        # Test collection statistics
        print("\n11. Collection statistics...")
        stats = search_engine.get_collection_stats()
        print(f"    Total embeddings: {stats.get('total_embeddings', 'N/A')}")
        print(f"    Model: {stats.get('model_name', 'N/A')}")
        print(f"    Chunk size: {stats.get('chunk_size', 'N/A')}")
        
        # Test utility functions
        print("\n12. Testing utility functions...")
        utility_results = search_files("machine learning", project_id=project.id)
        print(f"    Utility function found {len(utility_results)} results")
        
        print("\n✅ All semantic search tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up
        if 'original_config' in locals() and original_config:
            os.environ['ENVIRONMENT'] = original_config
        elif 'original_config' in locals():
            os.environ.pop('ENVIRONMENT', None)
            
        # Remove test directory
        try:
            shutil.rmtree(test_dir)
            print(f"\nCleaned up test directory: {test_dir}")
        except:
            print(f"\nWarning: Could not clean up test directory: {test_dir}")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n🧪 Testing Edge Cases")
    print("=" * 30)
    
    try:
        # Test empty content
        search_engine = SemanticSearch(
            chroma_path=tempfile.mkdtemp(prefix="edge_test_"),
            collection_name='edge_test'
        )
        
        empty_chunks = search_engine._split_text("")
        print(f"Empty text produces {len(empty_chunks)} chunks ✓")
        
        none_chunks = search_engine._split_text(None)
        print(f"None text produces {len(none_chunks)} chunks ✓") 
        
        # Test empty search
        empty_results = search_engine.search("")
        print(f"Empty query returns {len(empty_results)} results ✓")
        
        # Test very short text
        short_chunks = search_engine._split_text("Hi")
        print(f"Short text produces {len(short_chunks)} chunks ✓")
        
        # Test very long text
        long_text = "This is a test sentence. " * 1000
        long_chunks = search_engine._split_text(long_text)
        print(f"Long text ({len(long_text)} chars) produces {len(long_chunks)} chunks ✓")
        
        print("\n✅ Edge case tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Edge case test failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Starting Semantic Search Test Suite")
    print("=" * 60)
    
    # Test main functionality
    main_success = test_semantic_search()
    
    # Test edge cases  
    edge_success = test_edge_cases()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    print(f"Main functionality tests: {'✅ PASSED' if main_success else '❌ FAILED'}")
    print(f"Edge case tests: {'✅ PASSED' if edge_success else '❌ FAILED'}")
    
    if main_success and edge_success:
        print("\n🎉 All tests passed! Semantic search is ready for integration.")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Please check the implementation.")
        sys.exit(1)