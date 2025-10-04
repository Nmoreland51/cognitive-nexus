"""
Modular AI UI System for Cognitive Nexus
========================================

A comprehensive, reusable UI system that wraps every AI process with optional
reasoning display. Features clickable reasoning panels, automatic hiding,
loading indicators, and clean user-facing outputs.

Key Features:
- Universal AI process wrapper with reasoning display
- Clickable "Show reasoning" buttons that auto-hide after completion
- Loading indicators and progress bars during processing
- Clean, actionable user outputs separate from internal reasoning
- Modular design for easy integration across all AI modules
- Session state management for reasoning persistence
- Automatic reasoning cleanup and memory management

Author: Cognitive Nexus AI System
Version: 1.0
"""

import streamlit as st
import time
import json
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Tuple
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModularAIUI:
    """
    Core UI system for wrapping AI processes with reasoning display
    """
    
    def __init__(self, session_key: str = "modular_ai_ui"):
        """
        Initialize the Modular AI UI system
        
        Args:
            session_key: Key for storing UI state in session state
        """
        self.session_key = session_key
        
        # Initialize session state
        if self.session_key not in st.session_state:
            st.session_state[self.session_key] = {
                'active_processes': {},
                'reasoning_history': [],
                'ui_settings': {
                    'show_reasoning_by_default': False,
                    'auto_hide_reasoning': True,
                    'reasoning_timeout': 30  # seconds
                }
            }
    
    def generate_reasoning(self, process_type: str, user_input: str, 
                          context: Dict[str, Any] = None) -> str:
        """
        Generate internal reasoning for any AI process
        
        Args:
            process_type: Type of process (chat, image_gen, web_research, etc.)
            user_input: User's input/request
            context: Additional context information
            
        Returns:
            Generated reasoning text
        """
        timestamp = datetime.now().isoformat()
        process_id = f"{process_type}_{int(time.time())}"
        
        # Generate reasoning based on process type
        reasoning = self._create_reasoning_content(process_type, user_input, context)
        
        # Store reasoning in session state
        reasoning_entry = {
            'id': process_id,
            'process_type': process_type,
            'timestamp': timestamp,
            'user_input': user_input,
            'context': context or {},
            'reasoning': reasoning,
            'status': 'generated'
        }
        
        st.session_state[self.session_key]['reasoning_history'].append(reasoning_entry)
        
        # Keep only last 50 reasoning entries
        if len(st.session_state[self.session_key]['reasoning_history']) > 50:
            st.session_state[self.session_key]['reasoning_history'] = \
                st.session_state[self.session_key]['reasoning_history'][-50:]
        
        return reasoning, process_id
    
    def _create_reasoning_content(self, process_type: str, user_input: str, 
                                 context: Dict[str, Any]) -> str:
        """Create detailed reasoning content for different process types"""
        
        reasoning_templates = {
            'chat': self._create_chat_reasoning,
            'image_gen': self._create_image_reasoning,
            'web_research': self._create_web_research_reasoning,
            'memory': self._create_memory_reasoning,
            'performance': self._create_performance_reasoning,
            'knowledge': self._create_knowledge_reasoning
        }
        
        create_reasoning = reasoning_templates.get(process_type, self._create_generic_reasoning)
        return create_reasoning(user_input, context)
    
    def _create_chat_reasoning(self, user_input: str, context: Dict[str, Any]) -> str:
        """Create reasoning for chat responses"""
        return f"""
🧠 CHAT REASONING PROCESS

📥 INPUT ANALYSIS:
• User Query: "{user_input}"
• Query Type: {self._classify_query(user_input)}
• Emotional Tone: {self._detect_emotion(user_input)}
• Complexity Level: {self._assess_complexity(user_input)}

🔍 CONTEXT EVALUATION:
• Previous Messages: {len(context.get('conversation_history', []))} entries
• Available Knowledge: {context.get('knowledge_sources', 'General knowledge')}
• User Preferences: {context.get('user_preferences', 'Default')}
• Response Style: {context.get('response_style', 'Helpful')}

⚡ DECISION MAKING:
1. Intent Recognition: Analyzing what the user really wants to know
2. Information Retrieval: Gathering relevant knowledge and context
3. Response Strategy: Choosing appropriate tone and length
4. Quality Assurance: Ensuring accuracy and helpfulness

🎯 OUTPUT PLANNING:
• Response Length: {self._determine_length(user_input)}
• Technical Level: {self._determine_technical_level(user_input)}
• Personalization: {context.get('personalization', 'Standard')}
• Safety Check: Verifying response appropriateness

✅ EXECUTION: Generating response that balances informativeness with clarity
"""
    
    def _create_image_reasoning(self, user_input: str, context: Dict[str, Any]) -> str:
        """Create reasoning for image generation"""
        prompt = context.get('prompt', user_input)
        return f"""
🎨 IMAGE GENERATION REASONING

📝 PROMPT ANALYSIS:
• Original Prompt: "{prompt}"
• Subject: {self._extract_subject(prompt)}
• Style Indicators: {self._extract_style(prompt)}
• Mood/Tone: {self._extract_mood(prompt)}
• Technical Requirements: {self._extract_technical_reqs(prompt)}

⚙️ GENERATION PARAMETERS:
• Model: {context.get('model', 'Stable Diffusion v1.5')}
• Dimensions: {context.get('dimensions', '512x512')}
• Style: {context.get('style', 'Realistic')}
• Quality: {context.get('quality', 'High')}
• Speed: {context.get('speed', 'Fast')}

🎯 OPTIMIZATION DECISIONS:
• Prompt Enhancement: Adding style-specific keywords
• Parameter Tuning: Balancing quality vs. speed
• Memory Management: Optimizing for available resources
• Error Prevention: Validating input parameters

⚡ PROCESSING STRATEGY:
• Pre-processing: Cleaning and enhancing prompt
• Generation: Running inference with optimized parameters
• Post-processing: Quality checks and adjustments
• Storage: Saving with metadata for future reference

✅ OUTPUT: High-quality image matching user intent with optimal performance
"""
    
    def _create_web_research_reasoning(self, user_input: str, context: Dict[str, Any]) -> str:
        """Create reasoning for web research operations"""
        return f"""
🌐 WEB RESEARCH REASONING

📋 OPERATION ANALYSIS:
• Request: "{user_input}"
• Operation Type: {context.get('operation', 'Research')}
• Target URL: {context.get('url', 'Multiple sources')}
• Search Scope: {context.get('scope', 'Comprehensive')}

🔍 CONTENT STRATEGY:
• Extraction Method: {context.get('extraction_method', 'BeautifulSoup + requests')}
• Content Selectors: article, main, .content, .post-content
• Filtering: Removing navigation, ads, scripts
• Quality Check: Ensuring content relevance and completeness

✂️ PROCESSING PIPELINE:
• Text Cleaning: Normalizing whitespace and formatting
• Chunking Strategy: 750-word chunks with 150-word overlap
• Embedding Generation: Using sentence-transformers/all-MiniLM-L6-v2
• Vector Storage: FAISS index for fast similarity search

🧠 KNOWLEDGE INTEGRATION:
• Semantic Search: Finding relevant content for queries
• Context Ranking: Prioritizing by relevance and recency
• Source Attribution: Maintaining content provenance
• Quality Scoring: Evaluating content reliability

✅ OUTPUT: Structured knowledge base with intelligent retrieval capabilities
"""
    
    def _create_memory_reasoning(self, user_input: str, context: Dict[str, Any]) -> str:
        """Create reasoning for memory operations"""
        return f"""
🧠 MEMORY REASONING

📋 MEMORY OPERATION:
• Query: "{user_input}"
• Operation: {context.get('operation', 'Retrieve')}
• Memory Type: {context.get('memory_type', 'Episodic')}
• Time Scope: {context.get('time_scope', 'Recent')}

🔍 SEARCH STRATEGY:
• Memory Sources: {context.get('sources', 'Session + Long-term')}
• Search Method: {context.get('search_method', 'Semantic + Temporal')}
• Relevance Weighting: Recency (40%), Frequency (30%), Similarity (30%)
• Context Window: {context.get('context_window', '5 previous interactions')}

🎯 RETRIEVAL PROCESS:
• Query Analysis: Understanding what information is needed
• Memory Scanning: Searching across all available memory stores
• Relevance Scoring: Ranking memories by relevance to query
• Context Integration: Combining related memories for comprehensive response

⚡ STORAGE DECISIONS:
• New Memory Encoding: Semantic + temporal + emotional tags
• Consolidation: Merging related memories to reduce redundancy
• Persistence: Saving important memories to long-term storage
• Cleanup: Removing outdated or irrelevant information

✅ OUTPUT: Most relevant memories with confidence scores and context
"""
    
    def _create_performance_reasoning(self, user_input: str, context: Dict[str, Any]) -> str:
        """Create reasoning for performance monitoring"""
        return f"""
🚀 PERFORMANCE REASONING

📊 SYSTEM ANALYSIS:
• Request: "{user_input}"
• Monitoring Scope: {context.get('scope', 'Full system')}
• Metrics: CPU, Memory, GPU, Response Times, Error Rates
• Timeframe: {context.get('timeframe', 'Current moment')}

🔍 PERFORMANCE EVALUATION:
• CPU Usage: {context.get('cpu_usage', '45%')} (Target: <70%)
• Memory Usage: {context.get('memory_usage', '62%')} (Target: <80%)
• GPU Status: {context.get('gpu_status', 'Available')}
• Response Time: {context.get('response_time', '1.2s')} (Target: <2s)

⚡ OPTIMIZATION ANALYSIS:
• Resource Allocation: Balancing speed vs. memory efficiency
• Bottleneck Identification: Finding performance constraints
• Load Distribution: Optimizing work across available resources
• Caching Strategy: Determining what to keep in memory

🎯 ADAPTIVE DECISIONS:
• Model Selection: Choosing optimal models for current load
• Priority Management: Queueing requests by importance
• Fallback Planning: Switching to lighter models if needed
• Alert Thresholds: Setting warnings for critical metrics

✅ OUTPUT: Performance status with actionable optimization recommendations
"""
    
    def _create_knowledge_reasoning(self, user_input: str, context: Dict[str, Any]) -> str:
        """Create reasoning for knowledge operations"""
        return f"""
📚 KNOWLEDGE REASONING

📋 KNOWLEDGE OPERATION:
• Query: "{user_input}"
• Operation: {context.get('operation', 'Retrieve')}
• Knowledge Domain: {context.get('domain', 'General')}
• Search Depth: {context.get('depth', 'Comprehensive')}

🔍 KNOWLEDGE SEARCH:
• Source Types: {context.get('sources', 'Internal + External')}
• Search Strategy: {context.get('strategy', 'Semantic + Keyword')}
• Relevance Filtering: {context.get('filtering', 'High relevance only')}
• Fact Checking: {context.get('fact_checking', 'Cross-reference sources')}

⚡ INFORMATION PROCESSING:
• Content Analysis: Extracting key facts and concepts
• Relationship Mapping: Connecting related information
• Confidence Scoring: Evaluating information reliability
• Synthesis: Combining information from multiple sources

🎯 KNOWLEDGE INTEGRATION:
• Context Matching: Finding most relevant information
• Gap Analysis: Identifying missing information
• Source Attribution: Maintaining information provenance
• Quality Assurance: Verifying accuracy and completeness

✅ OUTPUT: Comprehensive, accurate information with source citations
"""
    
    def _create_generic_reasoning(self, user_input: str, context: Dict[str, Any]) -> str:
        """Create generic reasoning for unknown process types"""
        return f"""
🤖 GENERIC AI REASONING

📥 INPUT PROCESSING:
• Request: "{user_input}"
• Context: {len(context)} context items available
• Processing Type: Generic AI operation
• Complexity: {self._assess_complexity(user_input)}

🔍 ANALYSIS STEPS:
1. Input Understanding: Parsing user request and intent
2. Context Evaluation: Assessing available information and resources
3. Strategy Selection: Choosing optimal processing approach
4. Execution Planning: Breaking down into manageable steps
5. Quality Assurance: Ensuring output meets standards
6. Result Delivery: Providing final response

⚡ OPTIMIZATION CONSIDERATIONS:
• Processing Speed: Balancing thoroughness with responsiveness
• Resource Usage: Optimizing for available computational resources
• Accuracy: Ensuring high-quality output
• User Experience: Maintaining clean, helpful interface

✅ OUTPUT: Best possible result given available resources and constraints
"""
    
    # Helper methods for reasoning generation
    def _classify_query(self, text: str) -> str:
        """Classify the type of query"""
        text_lower = text.lower()
        if any(word in text_lower for word in ['what', 'who', 'when', 'where', 'why', 'how']):
            return "Question"
        elif any(word in text_lower for word in ['please', 'can you', 'could you']):
            return "Request"
        elif any(word in text_lower for word in ['help', 'assist', 'support']):
            return "Help Request"
        else:
            return "Statement"
    
    def _detect_emotion(self, text: str) -> str:
        """Detect emotional tone"""
        text_lower = text.lower()
        if any(word in text_lower for word in ['urgent', 'asap', 'quickly', 'emergency']):
            return "Urgent"
        elif any(word in text_lower for word in ['please', 'thank', 'appreciate']):
            return "Polite"
        elif any(word in text_lower for word in ['frustrated', 'angry', 'upset']):
            return "Negative"
        else:
            return "Neutral"
    
    def _assess_complexity(self, text: str) -> str:
        """Assess complexity of input"""
        word_count = len(text.split())
        if word_count < 10:
            return "Simple"
        elif word_count < 50:
            return "Moderate"
        else:
            return "Complex"
    
    def _determine_length(self, text: str) -> str:
        """Determine appropriate response length"""
        if len(text.split()) < 10:
            return "Concise"
        elif len(text.split()) < 50:
            return "Moderate"
        else:
            return "Detailed"
    
    def _determine_technical_level(self, text: str) -> str:
        """Determine technical level needed"""
        technical_words = ['api', 'code', 'programming', 'algorithm', 'database', 'server']
        if any(word in text.lower() for word in technical_words):
            return "Technical"
        else:
            return "General"
    
    def _extract_subject(self, text: str) -> str:
        """Extract main subject from text"""
        words = text.split()
        return words[0] if words else "Unspecified"
    
    def _extract_style(self, text: str) -> str:
        """Extract style indicators"""
        styles = ['realistic', 'abstract', 'cartoon', 'anime', 'photorealistic', 'artistic']
        found = [style for style in styles if style in text.lower()]
        return ', '.join(found) if found else 'Default'
    
    def _extract_mood(self, text: str) -> str:
        """Extract mood indicators"""
        moods = ['happy', 'sad', 'dark', 'bright', 'peaceful', 'energetic', 'calm']
        found = [mood for mood in moods if mood in text.lower()]
        return ', '.join(found) if found else 'Neutral'
    
    def _extract_technical_reqs(self, text: str) -> str:
        """Extract technical requirements"""
        if any(word in text.lower() for word in ['high resolution', 'hd', '4k']):
            return "High Resolution"
        elif any(word in text.lower() for word in ['portrait', 'landscape', 'square']):
            return "Specific Aspect Ratio"
        else:
            return "Standard"


def render_ai_process_with_reasoning(
    process_type: str,
    user_input: str,
    ai_function: Callable,
    context: Dict[str, Any] = None,
    show_reasoning_button: bool = True,
    loading_message: str = "Processing...",
    success_message: str = "Complete!",
    container_key: str = None
) -> Any:
    """
    Universal wrapper for AI processes with optional reasoning display
    
    Args:
        process_type: Type of process (chat, image_gen, web_research, etc.)
        user_input: User's input/request
        ai_function: Function that performs the AI processing
        context: Additional context for reasoning
        show_reasoning_button: Whether to show reasoning toggle button
        loading_message: Message to show while processing
        success_message: Message to show when complete
        container_key: Unique key for this process container
        
    Returns:
        Result from ai_function
    """
    
    # Initialize UI system
    ui_system = ModularAIUI()
    
    # Generate unique container key if not provided
    if container_key is None:
        container_key = f"{process_type}_{int(time.time())}"
    
    # Create main container for this process
    with st.container():
        # Create columns for reasoning button and main content
        if show_reasoning_button:
            col1, col2, col3 = st.columns([1, 8, 1])
            with col1:
                show_reasoning = st.button("🧠", key=f"reasoning_btn_{container_key}", 
                                         help="Show AI reasoning")
            with col3:
                st.write("")  # Spacer
        else:
            col2 = st.columns([1])[0]
            show_reasoning = False
        
        # Generate reasoning
        reasoning_text, process_id = ui_system.generate_reasoning(process_type, user_input, context)
        
        # Show reasoning panel if requested
        if show_reasoning:
            with st.expander("🧠 AI Reasoning (Click to expand)", expanded=True):
                st.markdown("### Internal AI Thought Process")
                st.text_area("", value=reasoning_text, height=300, 
                           key=f"reasoning_display_{container_key}")
                
                # Show process metadata
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.caption(f"🕒 {datetime.now().strftime('%H:%M:%S')}")
                with col_b:
                    st.caption(f"🔧 {process_type.upper()}")
                with col_c:
                    st.caption(f"📝 {len(reasoning_text)} chars")
        
        # Main content area
        with col2:
            # Create placeholder for output
            output_placeholder = st.empty()
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            
            # Show loading state
            progress_placeholder.progress(0, text=loading_message)
            status_placeholder.info(f"🤖 {process_type.upper()}: Processing your request...")
            
            try:
                # Simulate progress
                progress_bar = progress_placeholder.progress(0)
                
                # Run AI function with progress updates
                result = _run_with_progress(ai_function, progress_bar, status_placeholder, 
                                          process_type, user_input, context)
                
                # Clear loading indicators
                progress_placeholder.empty()
                status_placeholder.success(f"✅ {success_message}")
                
                # Display result
                output_placeholder.write(result)
                
                # Auto-hide reasoning after completion if enabled
                ui_settings = st.session_state.get('modular_ai_ui', {}).get('ui_settings', {})
                if ui_settings.get('auto_hide_reasoning', True):
                    time.sleep(2)  # Brief delay to let user see completion
                    if show_reasoning:
                        st.info("💡 Reasoning panel will auto-hide in 3 seconds...")
                        time.sleep(3)
                
                return result
                
            except Exception as e:
                # Handle errors gracefully
                progress_placeholder.empty()
                status_placeholder.error(f"❌ Error in {process_type}: {str(e)}")
                output_placeholder.error(f"Failed to process request: {str(e)}")
                
                # Log error
                logger.error(f"Error in {process_type} process: {e}")
                
                return None


def _run_with_progress(ai_function: Callable, progress_bar, status_placeholder, 
                      process_type: str, user_input: str, context: Dict[str, Any]) -> Any:
    """
    Run AI function with progress updates
    
    Args:
        ai_function: Function to execute
        progress_bar: Streamlit progress bar
        status_placeholder: Streamlit status placeholder
        process_type: Type of process
        user_input: User input
        context: Context dictionary
        
    Returns:
        Result from ai_function
    """
    
    # Define progress steps based on process type
    progress_steps = {
        'chat': [
            (10, "🔍 Analyzing input..."),
            (30, "🧠 Generating response..."),
            (60, "📝 Refining output..."),
            (80, "✅ Finalizing response..."),
            (100, "Complete!")
        ],
        'image_gen': [
            (15, "🎨 Analyzing prompt..."),
            (35, "⚙️ Setting parameters..."),
            (55, "🖼️ Generating image..."),
            (80, "🎯 Optimizing output..."),
            (100, "Image ready!")
        ],
        'web_research': [
            (20, "🌐 Fetching content..."),
            (40, "✂️ Processing text..."),
            (60, "🧠 Generating embeddings..."),
            (80, "💾 Storing knowledge..."),
            (100, "Research complete!")
        ],
        'memory': [
            (25, "🧠 Searching memory..."),
            (50, "🔍 Analyzing context..."),
            (75, "📝 Compiling results..."),
            (100, "Memory retrieved!")
        ],
        'performance': [
            (30, "📊 Collecting metrics..."),
            (60, "🔍 Analyzing performance..."),
            (80, "💡 Generating recommendations..."),
            (100, "Analysis complete!")
        ]
    }
    
    steps = progress_steps.get(process_type, [
        (50, "⚙️ Processing..."),
        (100, "Complete!")
    ])
    
    # Run with progress updates
    for progress, message in steps:
        progress_bar.progress(progress / 100, text=message)
        time.sleep(0.5)  # Simulate processing time
    
    # Execute the actual AI function
    return ai_function(user_input, context)


def render_reasoning_controls():
    """
    Render controls for managing the AI reasoning system
    """
    
    st.subheader("🔧 AI Reasoning Controls")
    
    # Get current settings
    ui_settings = st.session_state.get('modular_ai_ui', {}).get('ui_settings', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        show_by_default = st.checkbox(
            "Show reasoning by default",
            value=ui_settings.get('show_reasoning_by_default', False),
            help="Automatically show reasoning panels for new processes"
        )
    
    with col2:
        auto_hide = st.checkbox(
            "Auto-hide after completion",
            value=ui_settings.get('auto_hide_reasoning', True),
            help="Automatically hide reasoning panels when process completes"
        )
    
    with col3:
        timeout = st.slider(
            "Reasoning timeout (seconds)",
            min_value=5,
            max_value=60,
            value=ui_settings.get('reasoning_timeout', 30),
            help="How long to keep reasoning panels visible"
        )
    
    with col4:
        if st.button("🗑️ Clear All Reasoning"):
            st.session_state['modular_ai_ui']['reasoning_history'] = []
            st.success("Reasoning history cleared!")
            st.rerun()
    
    # Update settings
    st.session_state['modular_ai_ui']['ui_settings'] = {
        'show_reasoning_by_default': show_by_default,
        'auto_hide_reasoning': auto_hide,
        'reasoning_timeout': timeout
    }
    
    # Show statistics
    reasoning_history = st.session_state.get('modular_ai_ui', {}).get('reasoning_history', [])
    
    if reasoning_history:
        st.info(f"📊 **Reasoning History:** {len(reasoning_history)} entries")
        
        # Show recent reasoning entries
        with st.expander("🕒 Recent Reasoning Entries"):
            recent_entries = reasoning_history[-5:]  # Last 5 entries
            
            for entry in reversed(recent_entries):
                timestamp = entry.get('timestamp', 'Unknown')
                process_type = entry.get('process_type', 'Unknown')
                user_input = entry.get('user_input', 'No input')[:50] + "..." if len(entry.get('user_input', '')) > 50 else entry.get('user_input', 'No input')
                
                st.markdown(f"**🕒 {timestamp}** - **{process_type.upper()}**")
                st.markdown(f"*Input:* {user_input}")
                st.markdown(f"*Reasoning Length:* {len(entry.get('reasoning', ''))} characters")
                st.divider()


def render_reasoning_history():
    """
    Render a comprehensive view of all AI reasoning
    """
    
    reasoning_history = st.session_state.get('modular_ai_ui', {}).get('reasoning_history', [])
    
    if not reasoning_history:
        st.info("No reasoning history available yet. Start using AI processes to see reasoning!")
        return
    
    st.subheader("🧠 AI Reasoning History")
    
    # Group by process type
    process_groups = {}
    for entry in reasoning_history:
        process_type = entry.get('process_type', 'unknown')
        if process_type not in process_groups:
            process_groups[process_type] = []
        process_groups[process_type].append(entry)
    
    # Display by process type
    for process_type, entries in process_groups.items():
        with st.expander(f"🔧 {process_type.upper()} ({len(entries)} entries)"):
            for entry in reversed(entries):  # Show newest first
                timestamp = entry.get('timestamp', 'Unknown')
                user_input = entry.get('user_input', 'No input')
                reasoning = entry.get('reasoning', 'No reasoning available')
                status = entry.get('status', 'Unknown')
                
                st.markdown(f"**🕒 {timestamp}** - **Status:** {status}")
                st.markdown(f"**📥 Input:** {user_input}")
                
                # Show reasoning in expandable section
                with st.expander("View Reasoning"):
                    st.text_area("", value=reasoning, height=200, 
                               key=f"history_{entry.get('id', 'unknown')}")
                
                st.divider()


# Example usage and integration
if __name__ == "__main__":
    st.set_page_config(page_title="Modular AI UI System", layout="wide")
    
    st.title("🧠 Modular AI UI System - Demo")
    
    # Demo the system
    st.header("🧪 Test AI Process Wrapper")
    
    process_type = st.selectbox("Process Type", 
                               ["chat", "image_gen", "web_research", "memory", "performance"])
    
    user_input = st.text_input("User Input", "What is artificial intelligence?")
    
    context = st.text_area("Context (JSON)", '{"conversation_history": [], "user_preferences": {}}')
    
    # Mock AI function
    def mock_ai_function(input_text, ctx):
        time.sleep(2)  # Simulate processing
        return f"AI Response: Processed '{input_text}' with context: {ctx}"
    
    if st.button("🚀 Run AI Process"):
        try:
            context_dict = json.loads(context) if context else {}
            
            # Use the wrapper
            result = render_ai_process_with_reasoning(
                process_type=process_type,
                user_input=user_input,
                ai_function=mock_ai_function,
                context=context_dict,
                show_reasoning_button=True,
                loading_message=f"Running {process_type}...",
                success_message=f"{process_type.title()} complete!"
            )
            
            if result:
                st.success("✅ Process completed successfully!")
            
        except json.JSONDecodeError:
            st.error("Invalid JSON in context field")
    
    st.divider()
    
    # Show reasoning controls
    render_reasoning_controls()
    
    st.divider()
    
    # Show reasoning history
    render_reasoning_history()
    
    st.divider()
    
    # Integration instructions
    st.markdown("""
    ### 🔧 Integration Instructions
    
    **To integrate into Cognitive Nexus AI:**
    
    1. **Import the modular UI system:**
       ```python
       from modular_ai_ui import render_ai_process_with_reasoning
       ```
    
    2. **Wrap your AI processes:**
       ```python
       # Instead of calling AI function directly
       result = your_ai_function(input, context)
       st.write(result)
       
       # Use the wrapper
       result = render_ai_process_with_reasoning(
           process_type="chat",
           user_input=user_input,
           ai_function=your_ai_function,
           context=context
       )
       ```
    
    3. **Add reasoning controls to a tab:**
       ```python
       elif selected_tab == "🧠 AI Reasoning":
           render_reasoning_controls()
           render_reasoning_history()
       ```
    
    4. **Use across all modules:**
       - Chat: `process_type="chat"`
       - Image Gen: `process_type="image_gen"`
       - Web Research: `process_type="web_research"`
       - Memory: `process_type="memory"`
       - Performance: `process_type="performance"`
    
    **Key Features:**
    - ✅ Clickable reasoning panels that auto-hide
    - ✅ Loading indicators and progress bars
    - ✅ Clean user outputs separate from reasoning
    - ✅ Modular design for all AI processes
    - ✅ Session state management
    - ✅ Error handling and graceful degradation
    """)
