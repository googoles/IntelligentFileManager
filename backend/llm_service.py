#!/usr/bin/env python3
"""
Local LLM Service with Ollama Integration

This module provides AI-powered features using locally running LLMs via Ollama.
Features include document summarization, context-aware querying, and intelligent
file organization suggestions.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, AsyncGenerator
from pathlib import Path
import time
from datetime import datetime

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    ollama = None

logger = logging.getLogger(__name__)


class LLMConfig:
    """Configuration for LLM service"""
    
    def __init__(self):
        self.model_name = "llama3.2:3b"  # Lightweight default model
        self.timeout = 60  # seconds
        self.max_context_length = 4000  # characters
        self.temperature = 0.3  # Lower for more focused responses
        self.enabled = True
        self.fallback_enabled = True
        self.max_retries = 3
        self.chunk_size = 2000  # For processing large documents


class LocalLLMService:
    """Local LLM service using Ollama for AI-powered file management features"""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._available = False
        self._client = None
        self._model_loaded = False
        
        if not OLLAMA_AVAILABLE:
            logger.warning("⚠️  Ollama Python client not available. LLM features disabled.")
            self.config.enabled = False
            return
        
        if self.config.enabled:
            asyncio.create_task(self._initialize())
    
    async def _initialize(self):
        """Initialize the Ollama client and check model availability"""
        try:
            self._client = ollama.AsyncClient()
            
            # Check if Ollama service is running
            await self._check_ollama_service()
            
            # Ensure the model is available
            await self._ensure_model_available()
            
            self._available = True
            logger.info(f"✅ LLM service initialized with model: {self.config.model_name}")
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to initialize LLM service: {e}")
            if self.config.fallback_enabled:
                logger.info("LLM service will operate in fallback mode")
            else:
                self.config.enabled = False
    
    async def _check_ollama_service(self):
        """Check if Ollama service is running"""
        try:
            models = await self._client.list()
            logger.debug(f"Ollama service responding, {len(models.get('models', []))} models available")
        except Exception as e:
            raise Exception(f"Ollama service not available: {e}")
    
    async def _ensure_model_available(self):
        """Ensure the configured model is available, pull if necessary"""
        try:
            models = await self._client.list()
            model_names = [model['name'] for model in models.get('models', [])]
            
            if self.config.model_name not in model_names:
                logger.info(f"Pulling model: {self.config.model_name}")
                await self._client.pull(self.config.model_name)
                logger.info(f"✅ Model {self.config.model_name} pulled successfully")
            
            self._model_loaded = True
            
        except Exception as e:
            raise Exception(f"Failed to ensure model availability: {e}")
    
    @property
    def is_available(self) -> bool:
        """Check if LLM service is available and ready"""
        return self._available and self.config.enabled and self._model_loaded
    
    async def _generate_response(self, prompt: str, **kwargs) -> Optional[str]:
        """Generate a response using the local LLM"""
        if not self.is_available:
            return None
        
        try:
            response = await self._client.generate(
                model=self.config.model_name,
                prompt=prompt,
                options={
                    'temperature': kwargs.get('temperature', self.config.temperature),
                    'num_predict': kwargs.get('max_tokens', 500),
                    'stop': kwargs.get('stop_sequences', []),
                }
            )
            return response['response'].strip()
            
        except Exception as e:
            logger.error(f"Failed to generate LLM response: {e}")
            return None
    
    async def _generate_streaming_response(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate a streaming response using the local LLM"""
        if not self.is_available:
            return
        
        try:
            stream = await self._client.generate(
                model=self.config.model_name,
                prompt=prompt,
                stream=True,
                options={
                    'temperature': kwargs.get('temperature', self.config.temperature),
                    'num_predict': kwargs.get('max_tokens', 500),
                }
            )
            
            async for chunk in stream:
                if 'response' in chunk:
                    yield chunk['response']
                    
        except Exception as e:
            logger.error(f"Failed to generate streaming LLM response: {e}")
            return
    
    async def summarize_content(self, content: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Summarize file content with optional context
        
        Args:
            content: The text content to summarize
            context: Additional context (file name, type, project info)
            
        Returns:
            Dict with summary, key_points, and metadata
        """
        if not self.is_available:
            return {
                "error": "LLM service not available",
                "fallback_summary": self._create_fallback_summary(content),
                "service_available": False
            }
        
        # Truncate content if too long
        if len(content) > self.config.max_context_length:
            content = content[:self.config.max_context_length] + "..."
        
        # Build context-aware prompt
        context_info = ""
        if context:
            file_name = context.get('file_name', 'Unknown file')
            file_type = context.get('file_type', 'Unknown type')
            project_name = context.get('project_name', 'Unknown project')
            context_info = f"File: {file_name} (Type: {file_type}, Project: {project_name})\n\n"
        
        prompt = f"""You are an AI assistant helping researchers organize their files. Please provide a concise summary of the following content.

{context_info}Content to summarize:
{content}

Please provide:
1. A brief summary (2-3 sentences)
2. Key points or findings (bullet points)
3. The main topic or category

Format your response as JSON with keys: summary, key_points (array), main_topic, confidence_score (0-1).
"""
        
        try:
            response = await self._generate_response(prompt, max_tokens=300)
            
            if response:
                # Try to parse JSON response
                try:
                    result = json.loads(response)
                    result['service_available'] = True
                    result['generated_at'] = datetime.now().isoformat()
                    return result
                except json.JSONDecodeError:
                    # If JSON parsing fails, create structured response
                    return {
                        "summary": response[:200] + "..." if len(response) > 200 else response,
                        "key_points": [],
                        "main_topic": "Unknown",
                        "confidence_score": 0.5,
                        "service_available": True,
                        "generated_at": datetime.now().isoformat(),
                        "note": "Response was not in expected JSON format"
                    }
            else:
                return {
                    "error": "Failed to generate summary",
                    "fallback_summary": self._create_fallback_summary(content),
                    "service_available": True
                }
                
        except Exception as e:
            logger.error(f"Error in summarize_content: {e}")
            return {
                "error": str(e),
                "fallback_summary": self._create_fallback_summary(content),
                "service_available": True
            }
    
    async def query_content(self, query: str, content: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Answer questions about file content
        
        Args:
            query: The user's question
            content: The file content to query against
            context: Additional context about the file
            
        Returns:
            Dict with answer, confidence, and metadata
        """
        if not self.is_available:
            return {
                "error": "LLM service not available",
                "fallback_answer": f"Unable to process query '{query}' - LLM service unavailable",
                "service_available": False
            }
        
        # Truncate content if too long
        if len(content) > self.config.max_context_length:
            content = content[:self.config.max_context_length] + "..."
        
        # Build context-aware prompt
        context_info = ""
        if context:
            file_name = context.get('file_name', 'Unknown file')
            file_type = context.get('file_type', 'Unknown type')
            context_info = f"File: {file_name} (Type: {file_type})\n\n"
        
        prompt = f"""You are an AI assistant helping researchers analyze their files. Based on the content below, please answer the user's question accurately and concisely.

{context_info}Content:
{content}

User Question: {query}

Please provide a clear, accurate answer based on the content. If the answer cannot be found in the content, say so clearly. Format your response as JSON with keys: answer, confidence_score (0-1), source_found (boolean), additional_notes.
"""
        
        try:
            response = await self._generate_response(prompt, max_tokens=400)
            
            if response:
                # Try to parse JSON response
                try:
                    result = json.loads(response)
                    result['service_available'] = True
                    result['query'] = query
                    result['generated_at'] = datetime.now().isoformat()
                    return result
                except json.JSONDecodeError:
                    # If JSON parsing fails, create structured response
                    return {
                        "answer": response,
                        "confidence_score": 0.5,
                        "source_found": True,
                        "query": query,
                        "service_available": True,
                        "generated_at": datetime.now().isoformat(),
                        "note": "Response was not in expected JSON format"
                    }
            else:
                return {
                    "error": "Failed to generate answer",
                    "fallback_answer": f"Unable to process query: '{query}'",
                    "service_available": True
                }
                
        except Exception as e:
            logger.error(f"Error in query_content: {e}")
            return {
                "error": str(e),
                "fallback_answer": f"Error processing query: '{query}'",
                "service_available": True
            }
    
    async def suggest_organization(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Suggest file organization based on content and metadata
        
        Args:
            file_info: Dict with file_name, content, type, size, etc.
            
        Returns:
            Dict with suggested_folder, reasoning, and confidence
        """
        if not self.is_available:
            return {
                "error": "LLM service not available",
                "fallback_folder": self._create_fallback_organization(file_info),
                "service_available": False
            }
        
        file_name = file_info.get('file_name', 'unknown')
        content = file_info.get('content', '')
        file_type = file_info.get('file_type', 'unknown')
        
        # Truncate content for analysis
        if len(content) > 1000:
            content = content[:1000] + "..."
        
        prompt = f"""You are an AI assistant helping researchers organize their files. Based on the file information below, suggest the best folder/category for organization.

File Name: {file_name}
File Type: {file_type}
Content Preview: {content}

Common research folders include:
- literature (papers, articles, references)
- data (datasets, raw data, processed data)
- code (scripts, notebooks, programs)
- results (outputs, figures, analyses)
- drafts (work in progress, manuscripts)
- notes (meeting notes, ideas, observations)
- presentations (slides, talks)
- documentation (manuals, guides)

Please suggest the most appropriate folder and provide reasoning. Format as JSON with keys: suggested_folder, reasoning, confidence_score (0-1), alternative_folders (array).
"""
        
        try:
            response = await self._generate_response(prompt, max_tokens=200)
            
            if response:
                try:
                    result = json.loads(response)
                    result['service_available'] = True
                    result['file_name'] = file_name
                    result['generated_at'] = datetime.now().isoformat()
                    return result
                except json.JSONDecodeError:
                    # Extract folder suggestion from text response
                    return {
                        "suggested_folder": self._extract_folder_from_text(response),
                        "reasoning": response,
                        "confidence_score": 0.5,
                        "alternative_folders": [],
                        "service_available": True,
                        "file_name": file_name,
                        "generated_at": datetime.now().isoformat()
                    }
            else:
                return {
                    "error": "Failed to generate organization suggestion",
                    "fallback_folder": self._create_fallback_organization(file_info),
                    "service_available": True
                }
                
        except Exception as e:
            logger.error(f"Error in suggest_organization: {e}")
            return {
                "error": str(e),
                "fallback_folder": self._create_fallback_organization(file_info),
                "service_available": True
            }
    
    def _create_fallback_summary(self, content: str) -> str:
        """Create a simple fallback summary when LLM is unavailable"""
        if not content:
            return "No content available to summarize"
        
        # Simple extractive summary - first sentence and length info
        sentences = content.replace('\n', ' ').split('. ')
        first_sentence = sentences[0] if sentences else ""
        
        word_count = len(content.split())
        char_count = len(content)
        
        return f"Content preview: {first_sentence[:100]}{'...' if len(first_sentence) > 100 else ''}. Document contains {word_count} words ({char_count} characters)."
    
    def _create_fallback_organization(self, file_info: Dict[str, Any]) -> str:
        """Create fallback organization suggestion based on file extension"""
        file_name = file_info.get('file_name', '').lower()
        file_type = file_info.get('file_type', '').lower()
        
        # Simple rule-based classification
        if file_type in ['.pdf', '.doc', '.docx', '.txt', '.md']:
            if any(keyword in file_name for keyword in ['paper', 'article', 'journal', 'literature']):
                return 'literature'
            elif any(keyword in file_name for keyword in ['note', 'meeting', 'idea']):
                return 'notes'
            else:
                return 'documents'
        elif file_type in ['.csv', '.xlsx', '.json', '.xml']:
            return 'data'
        elif file_type in ['.py', '.js', '.r', '.ipynb', '.java']:
            return 'code'
        elif file_type in ['.png', '.jpg', '.jpeg', '.gif', '.svg']:
            return 'results'
        else:
            return 'misc'
    
    def _extract_folder_from_text(self, text: str) -> str:
        """Extract folder suggestion from text response"""
        common_folders = ['literature', 'data', 'code', 'results', 'drafts', 'notes', 'presentations', 'documentation']
        
        text_lower = text.lower()
        for folder in common_folders:
            if folder in text_lower:
                return folder
        
        return 'misc'
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get detailed service status information"""
        status = {
            "enabled": self.config.enabled,
            "available": self.is_available,
            "model_name": self.config.model_name,
            "model_loaded": self._model_loaded,
            "ollama_client_available": OLLAMA_AVAILABLE,
            "config": {
                "timeout": self.config.timeout,
                "max_context_length": self.config.max_context_length,
                "temperature": self.config.temperature,
                "fallback_enabled": self.config.fallback_enabled
            }
        }
        
        if self.is_available:
            try:
                # Test basic functionality
                test_response = await self._generate_response("Say 'OK' if you can respond.", max_tokens=10)
                status["test_response"] = test_response
                status["last_tested"] = datetime.now().isoformat()
            except Exception as e:
                status["test_error"] = str(e)
        
        return status


# Global instance
_llm_service_instance = None


def get_llm_service() -> LocalLLMService:
    """Get the global LLM service instance"""
    global _llm_service_instance
    if _llm_service_instance is None:
        _llm_service_instance = LocalLLMService()
    return _llm_service_instance


async def initialize_llm_service(config: Optional[LLMConfig] = None) -> LocalLLMService:
    """Initialize the global LLM service instance"""
    global _llm_service_instance
    _llm_service_instance = LocalLLMService(config)
    return _llm_service_instance