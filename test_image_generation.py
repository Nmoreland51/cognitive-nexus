#!/usr/bin/env python3
"""
Test script for Image Generation functionality
Tests the ImageGenerationService without running the full Streamlit app
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_dependencies():
    """Test if image generation dependencies are available"""
    print("🔍 Testing Image Generation Dependencies...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"❌ PyTorch not available: {e}")
        return False
    
    try:
        import diffusers
        print(f"✅ Diffusers: {diffusers.__version__}")
    except ImportError as e:
        print(f"❌ Diffusers not available: {e}")
        return False
    
    try:
        from PIL import Image
        print(f"✅ Pillow: {Image.__version__}")
    except ImportError as e:
        print(f"❌ Pillow not available: {e}")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Transformers not available: {e}")
        return False
    
    return True

def test_image_service():
    """Test the ImageGenerationService"""
    print("\n🧪 Testing ImageGenerationService...")
    
    try:
        # Import the service (this will test the imports in the main file)
        from cognitive_nexus_advanced import ImageGenerationService
        
        # Create service instance
        service = ImageGenerationService()
        
        print(f"✅ Service created successfully")
        print(f"   Available: {service.available}")
        print(f"   Device: {service.device}")
        print(f"   Images directory: {service.images_dir}")
        
        if service.available:
            print(f"   Available styles: {service.get_available_styles()}")
            
            # Test generation (this will download the model if not present)
            print("\n🎨 Testing image generation...")
            print("   Note: This may take several minutes on first run (model download)")
            
            result = service.generate_image(
                prompt="a simple red circle on white background",
                width=512,
                height=512,
                style="realistic"
            )
            
            if result and result.get("success"):
                print("✅ Image generation successful!")
                print(f"   File: {result['metadata']['filename']}")
                print(f"   Seed: {result['metadata']['seed']}")
            else:
                print(f"❌ Image generation failed: {result.get('error', 'Unknown error')}")
                
        else:
            print("❌ Service not available - dependencies missing")
            
    except Exception as e:
        print(f"❌ Service test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main test function"""
    print("🎨 Cognitive Nexus AI - Image Generation Test")
    print("=" * 50)
    
    # Test dependencies
    if not test_dependencies():
        print("\n❌ Dependencies test failed!")
        print("Please install required packages:")
        print("pip install torch diffusers pillow transformers accelerate safetensors")
        return False
    
    # Test service
    test_image_service()
    
    print("\n✅ Test completed!")
    return True

if __name__ == "__main__":
    main()
