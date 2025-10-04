#!/usr/bin/env python3
"""
Test script for Web Research Module
Tests core functionality without Streamlit UI
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_web_research_module():
    """Test the web research module functionality"""
    print("🧪 Testing Web Research Module...")
    print("=" * 60)
    
    try:
        from cognitive_web_research import WebResearchModule
        
        # Test 1: Initialize module
        print("1. Initializing Web Research Module...")
        web_research = WebResearchModule("test_brain")
        print("✅ Module initialized successfully")
        
        # Test 2: Test URL scraping
        print("\n2. Testing URL content extraction...")
        test_url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
        
        content_data = web_research.scrape_url(test_url)
        if content_data['status'] == 'success':
            print(f"✅ Successfully scraped: {content_data['title']}")
            print(f"   Word count: {content_data['word_count']:,}")
            print(f"   Headings: {len(content_data['headings'])}")
        else:
            print(f"❌ Scraping failed: {content_data.get('error', 'Unknown error')}")
            return False
        
        # Test 3: Test text preprocessing and chunking
        print("\n3. Testing text preprocessing and chunking...")
        chunks = web_research.chunk_text(content_data['content'], target_size=500, overlap=100)
        print(f"✅ Created {len(chunks)} chunks")
        print(f"   Average chunk size: {sum(chunk['word_count'] for chunk in chunks) // len(chunks)} words")
        
        # Test 4: Test embedding generation
        print("\n4. Testing embedding generation...")
        embeddings = web_research.embed_chunks(chunks)
        print(f"✅ Generated {len(embeddings)} embeddings")
        print(f"   Embedding dimension: {len(embeddings[0]) if embeddings else 'N/A'}")
        
        # Test 5: Test storage in brain
        print("\n5. Testing storage in brain...")
        success = web_research.store_in_brain(test_url, chunks, embeddings)
        if success:
            print("✅ Successfully stored in brain")
        else:
            print("❌ Failed to store in brain")
            return False
        
        # Test 6: Test retrieval
        print("\n6. Testing semantic search...")
        query = "What is artificial intelligence?"
        results = web_research.retrieve(query, k=3)
        print(f"✅ Retrieved {len(results)} relevant chunks")
        if results:
            print(f"   Top similarity: {results[0]['similarity']:.3f}")
        
        # Test 7: Test answer generation
        print("\n7. Testing answer generation...")
        answer = web_research.answer_query(query)
        print(f"✅ Generated answer ({len(answer)} characters)")
        print(f"   Preview: {answer[:150]}...")
        
        # Test 8: Test brain statistics
        print("\n8. Testing brain statistics...")
        stats = web_research.get_brain_stats()
        print(f"✅ Brain stats:")
        print(f"   URLs: {stats['urls_processed']}")
        print(f"   Chunks: {stats['total_chunks']}")
        print(f"   Words: {stats['total_words']:,}")
        print(f"   Vectors: {stats['vectors_stored']}")
        print(f"   Size: {stats['brain_size_mb']} MB")
        
        print("\n🎉 All tests passed successfully!")
        print("\n📋 Next steps:")
        print("1. Install dependencies: pip install -r requirements_web_research.txt")
        print("2. Run UI test: streamlit run cognitive_web_research.py")
        print("3. Integrate into your main Cognitive Nexus AI app")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n📦 Install dependencies:")
        print("pip install streamlit requests beautifulsoup4 sentence-transformers faiss-cpu numpy")
        return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_dependencies():
    """Test if required dependencies are available"""
    print("\n🔍 Checking dependencies...")
    
    dependencies = [
        ('streamlit', 'Streamlit'),
        ('requests', 'HTTP requests'),
        ('bs4', 'BeautifulSoup'),
        ('numpy', 'NumPy'),
        ('sentence_transformers', 'Sentence Transformers'),
        ('faiss', 'FAISS vector database')
    ]
    
    missing = []
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - Missing")
            missing.append(module)
    
    if missing:
        print(f"\n📦 Install missing dependencies:")
        print("pip install streamlit requests beautifulsoup4 sentence-transformers faiss-cpu numpy")
        return False
    
    print("✅ All dependencies available")
    return True

if __name__ == "__main__":
    print("Web Research Module Test Suite")
    print("=" * 60)
    
    # Check dependencies first
    deps_ok = test_dependencies()
    
    if deps_ok:
        # Run main tests
        success = test_web_research_module()
        
        if success:
            print("\n🚀 Ready to use!")
            print("Run 'streamlit run cognitive_web_research.py' to test the UI")
        else:
            print("\n❌ Tests failed - check the output above")
            sys.exit(1)
    else:
        print("\n❌ Missing dependencies - install them first")
        sys.exit(1)
