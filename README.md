# 🧠 Cognitive Nexus AI

A comprehensive self-hosted, privacy-focused AI assistant that combines real-time web search capabilities with local language model support. Complete single-file version for easy deployment and distribution.

## ✨ Features

- **🔒 Privacy-First Design**: All processing happens locally when possible
- **🌐 Real-Time Web Search**: Multi-source search with intelligent content extraction
- **🤖 Local LLM Support**: Integration with Ollama and Hugging Face Transformers
- **🧠 Learning & Memory**: Remembers conversations and user preferences
- **⚡ Intelligent Fallbacks**: Graceful degradation when services are unavailable
- **🎨 Enhanced UI**: Dark/light theme support with responsive design
- **📊 Comprehensive Logging**: Detailed operation tracking and debugging
- **💾 Conversation Persistence**: SQLite database for conversation history

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download the single file:**
   ```bash
   # Download the cognitive_nexus_ai.py file
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Optional: Install Ollama for local LLM support:**
   ```bash
   # Follow instructions at https://ollama.ai/
   # Then pull a model: ollama pull llama2
   ```

4. **Run the application:**
   ```bash
   streamlit run cognitive_nexus_ai.py
   ```

5. **Open your browser to:** `http://localhost:8501`

## 🔧 Configuration

### Environment Variables (Optional)

```bash
# Ollama server URL (default: http://localhost:11434)
export OLLAMA_URL=http://localhost:11434

# Anthropic API key (for cloud LLM support)
export ANTHROPIC_API_KEY=your_api_key_here
```

### Settings Panel

The application includes a comprehensive settings panel where you can:

- **🤖 AI Provider**: Switch between local and cloud AI providers
- **📚 Show Sources**: Toggle source citations in responses
- **🧠 Learning Mode**: Enable/disable conversation memory
- **🌐 Web Search**: Control web search functionality
- **🌡️ Temperature**: Adjust response creativity (for local models)

## 🏗️ Architecture

### Core Components

1. **OllamaManager**: Handles local LLM integration and model detection
2. **ChainedWebSearchSystem**: 8-step modular search framework with semantic intent detection
3. **LearningSystem**: Advanced conversation learning and memory management
4. **FallbackResponseSystem**: Intelligent fallback responses with pattern matching
5. **CognitiveNexusCore**: Main integration system with provider management
6. **ConversationDB**: SQLite-based conversation persistence

### Search Chain Process

The web search system uses an 8-step chain:

1. **Semantic Intent Detection**: Classifies query intent and determines if search is needed
2. **Enhanced Query Execution**: Parallel search across multiple providers
3. **Advanced Content Extraction**: Semantic analysis and content extraction
4. **Intelligent Filtering**: Quality assessment and spam detection
5. **Contextual Synthesis**: Information synthesis from multiple sources
6. **Citation Grounding**: Proper source attribution and provenance tracking
7. **Adaptive Response Generation**: Intent-aware response formatting
8. **Quality Evaluation**: Confidence scoring and iteration decisions

## 🛠️ Dependencies

### Required
- `streamlit`: Web interface framework
- `requests`: HTTP client for web search
- `beautifulsoup4`: HTML parsing and content extraction
- `psutil`: System monitoring and resource management

### Optional (for enhanced functionality)
- `trafilatura`: Advanced content extraction
- `transformers`: Hugging Face model support
- `torch`: PyTorch for local models
- `anthropic`: Cloud AI provider integration

## 📁 File Structure

```
cognitive_nexus_ai.py          # Main application (single file)
requirements.txt               # Python dependencies
README.md                      # This file
data/                          # Auto-created data directory
├── cognitive_nexus_knowledge.json  # Learned facts and preferences
├── conversations.db               # SQLite conversation history
└── cognitive_nexus.log            # Application logs
```

## 🔍 Usage Examples

### Basic Queries
- "What is artificial intelligence?"
- "Explain quantum computing"
- "How does machine learning work?"

### Current Information
- "Latest news about AI developments"
- "Current weather in Tokyo"
- "Recent updates on Python 3.12"

### Comparisons
- "Compare Python vs JavaScript"
- "What's the difference between React and Vue?"
- "Better: Docker or Kubernetes?"

### Research
- "Find information about renewable energy"
- "Research on climate change impacts"
- "Statistics on global internet usage"

## 🔒 Privacy Features

- **Local Processing**: All AI inference happens locally when using Ollama
- **No Data Sharing**: Conversations are stored locally only
- **Graceful Degradation**: Works offline with built-in knowledge base
- **User Control**: Full control over data retention and learning

## 🚨 Troubleshooting

### Common Issues

1. **Ollama not detected:**
   - Ensure Ollama is installed and running
   - Check if models are pulled: `ollama list`
   - Verify the service is accessible at `http://localhost:11434`

2. **Web search not working:**
   - Check internet connection
   - Verify firewall settings
   - Try disabling and re-enabling web search in settings

3. **Memory issues:**
   - Clear browser cache
   - Restart the Streamlit application
   - Check available system memory

### Logs and Debugging

- Application logs are saved to `cognitive_nexus.log`
- Use the browser's developer console for UI debugging
- Enable debug mode in the settings panel

## 🤝 Contributing

This is a single-file implementation designed for easy deployment and modification. To contribute:

1. Fork or download the file
2. Make your modifications
3. Test thoroughly
4. Share your improvements

## 📄 License

This project is provided as-is for educational and personal use. Please respect the terms of service of any external APIs or services used.

## 🙏 Acknowledgments

- **Streamlit** for the excellent web framework
- **Ollama** for local LLM capabilities
- **Hugging Face** for transformer models
- **DuckDuckGo** for privacy-focused search API
- **Wikipedia** for authoritative content

---

**Cognitive Nexus AI** - Your privacy-focused, intelligent assistant 🧠✨
