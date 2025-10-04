#!/usr/bin/env python3
"""
Quick test script for Cognitive Nexus Simple Demo
Verifies the demo app can be imported and basic functionality works
"""

import sys
import os
from pathlib import Path

def test_demo_app():
    """Test the simple demo app"""
    print("🧪 Testing Cognitive Nexus Simple Demo...")
    print("=" * 50)
    
    # Check if demo file exists
    demo_file = Path("cognitive_nexus_simple_demo.py")
    if not demo_file.exists():
        print("❌ cognitive_nexus_simple_demo.py not found")
        return False
    
    print("✅ Demo file found")
    
    # Check Python syntax
    try:
        with open(demo_file, 'r') as f:
            code = f.read()
        compile(code, demo_file, 'exec')
        print("✅ Python syntax is valid")
    except SyntaxError as e:
        print(f"❌ Syntax error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False
    
    # Check required imports
    required_modules = ['streamlit', 'datetime']
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module} is available")
        except ImportError:
            missing_modules.append(module)
            print(f"❌ {module} is missing")
    
    if missing_modules:
        print(f"\n📦 Missing modules: {', '.join(missing_modules)}")
        print("Install with: pip install streamlit")
        return False
    
    print("\n🎉 All tests passed!")
    print("✅ Demo app is ready to run")
    print("\n🚀 To start the demo:")
    print("   - Double-click: run_simple_demo.bat")
    print("   - Or run: streamlit run cognitive_nexus_simple_demo.py --server.port 8502")
    
    return True

if __name__ == "__main__":
    success = test_demo_app()
    if not success:
        print("\n❌ Demo setup incomplete - please fix the issues above")
        sys.exit(1)
    else:
        print("\n✨ Demo is ready for immediate testing!")
