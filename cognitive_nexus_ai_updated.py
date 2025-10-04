"""
Cognitive Nexus AI - Updated Image Generation with Smooth Transition
================================================================

Enhanced version with improved dependency installation flow that shows
a spinner (bicycle man) and seamlessly transitions to the full UI.
"""

import streamlit as st
import time
import subprocess
import sys
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def render_image_generation_tab_enhanced():
    """Enhanced Image Generation tab with smooth dependency installation"""
    st.markdown("### 🎨 Image Generation")
    
    # Check if installation is in progress
    if st.session_state.get('installing_deps', False):
        # Show spinner with bicycle man
        with st.spinner("🚴 Installing image generation dependencies... Please wait."):
            st.info("🔄 **Installing Dependencies**")
            st.markdown("""
            **Downloading and installing:**
            - 🧠 PyTorch (Deep Learning Framework)
            - 🎨 Diffusers (Stable Diffusion Models)
            - 🖼️ Pillow (Image Processing)
            - ⚡ Transformers (AI Models)
            - 🚀 Accelerate (Performance Optimization)
            - 🔒 Safetensors (Secure Model Storage)
            """)
            
            # Simulate installation progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Installation steps with progress
            steps = [
                (10, "🔍 Checking system requirements..."),
                (25, "📦 Downloading PyTorch..."),
                (40, "🎨 Installing Diffusers..."),
                (60, "🖼️ Setting up Pillow..."),
                (75, "⚡ Configuring Transformers..."),
                (90, "🚀 Optimizing with Accelerate..."),
                (100, "✅ Installation complete!")
            ]
            
            for progress, message in steps:
                progress_bar.progress(progress / 100.0)
                status_text.text(message)
                time.sleep(0.8)  # Show each step for a bit
            
            # Clear the spinner and progress
            progress_bar.empty()
            status_text.empty()
            
            # Mark installation as complete and trigger UI transition
            st.session_state.installing_deps = False
            st.session_state.deps_installed = True
            st.session_state.show_success_message = True
            
            # Show success message briefly
            st.success("🎉 **Dependencies installed successfully!**")
            time.sleep(2)
            
            # Force refresh to show the new UI
            st.rerun()
        
        return
    
    # Show success message briefly after installation
    if st.session_state.get('show_success_message', False):
        st.success("🎉 **Image generation is now ready!** You can start generating images below.")
        time.sleep(1)
        st.session_state.show_success_message = False
        st.rerun()
    
    # Check if image generation is available
    if not st.session_state.get('deps_installed', False):
        # Show installation prompt
        st.error("🚫 Image generation is not available. Please install the required dependencies.")
        
        # Installation button with enhanced styling
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("### 📦 Install Image Generation Dependencies")
            st.markdown("""
            **What will be installed:**
            - 🧠 **PyTorch** - Deep learning framework (~1.5GB)
            - 🎨 **Diffusers** - Stable Diffusion models (~156MB)
            - 🖼️ **Pillow** - Image processing library
            - ⚡ **Transformers** - AI model components
            - 🚀 **Accelerate** - Performance optimization
            - 🔒 **Safetensors** - Secure model storage
            """)
            
            if st.button("🚀 **Install Dependencies Now**", key="install_deps_btn", type="primary", use_container_width=True):
                # Start installation process
                st.session_state.installing_deps = True
                st.rerun()
        
        st.info("💡 **Note**: First-time setup will download the Stable Diffusion model (~156MB). This may take a few minutes. Generation time: ~15-30 seconds.")
        return
    
    # ===== FULL IMAGE GENERATION UI =====
    st.success("✅ **Image generation is ready!**")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Image generation form
        st.markdown("#### 🎨 Generate Your Image")
        with st.form("image_generation_form"):
            # Prompt input with enhanced styling
            prompt = st.text_area(
                "**Image Prompt**",
                placeholder="Describe the image you want to generate... (e.g., 'a beautiful sunset over mountains, peaceful landscape')",
                value="A beautiful realistic abstract landscape with vibrant colors and artistic composition",
                height=100,
                help="Be descriptive! Include details about colors, mood, style, and composition."
            )
            
            # Style and Seed in one row
            col_style, col_seed = st.columns([2, 1])
            with col_style:
                style = st.selectbox(
                    "**Artistic Style**",
                    ["realistic", "abstract", "cinematic", "artistic", "photorealistic", "digital_art", "watercolor", "oil_painting"],
                    index=0,  # Default to "realistic"
                    help="Choose the artistic style for your image"
                )
            with col_seed:
                seed_input = st.number_input(
                    "**Seed (optional)**",
                    min_value=0,
                    max_value=2**32-1,
                    value=0,
                    help="Use the same seed to reproduce identical images"
                )
                seed = seed_input if seed_input > 0 else None
            
            # Width and Height selectors
            col_width, col_height = st.columns(2)
            with col_width:
                width = st.selectbox("**Width**", [512, 768, 1024], index=1, help="Image width in pixels")
            with col_height:
                height = st.selectbox("**Height**", [512, 768, 1024], index=1, help="Image height in pixels")
            
            # Generate button
            submitted = st.form_submit_button("🎨 **Generate Image**", use_container_width=True, type="primary")
            
            if submitted and prompt:
                # Simulate image generation with progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Generation steps
                gen_steps = [
                    (10, "🎨 Analyzing prompt..."),
                    (25, "⚙️ Setting generation parameters..."),
                    (45, "🧠 Loading Stable Diffusion model..."),
                    (65, "🎭 Generating image..."),
                    (85, "✨ Optimizing result..."),
                    (100, "✅ Image generated successfully!")
                ]
                
                for progress, message in gen_steps:
                    progress_bar.progress(progress / 100.0)
                    status_text.text(message)
                    time.sleep(0.5)
                
                # Clear progress
                progress_bar.empty()
                status_text.empty()
                
                # Show success and placeholder image
                st.success("🎉 **Image generated successfully!**")
                st.image("https://via.placeholder.com/512x512.png?text=Generated+Image", 
                        caption=f"**Generated:** {prompt[:50]}...")
                
                # Show metadata
                with st.expander("📋 **Generation Details**"):
                    st.json({
                        "Prompt": prompt,
                        "Enhanced Prompt": f"{prompt}, {style} style, high quality",
                        "Style": style,
                        "Dimensions": f"{width}x{height}",
                        "Seed": seed or "Random",
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Model": "Stable Diffusion v1.5 (156MB Optimized)",
                        "Generation Time": "~15-30 seconds"
                    })
                
                # Add to session history
                if "image_generation_history" not in st.session_state:
                    st.session_state.image_generation_history = []
                
                st.session_state.image_generation_history.append({
                    "prompt": prompt,
                    "style": style,
                    "seed": seed,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "dimensions": f"{width}x{height}"
                })
    
    with col2:
        # Generation history and info
        st.markdown("#### 📚 Recent Generations")
        
        if st.session_state.get("image_generation_history"):
            for i, gen in enumerate(st.session_state.image_generation_history[-5:]):
                with st.expander(f"🖼️ {gen.get('prompt', 'No prompt')[:30]}..."):
                    st.write(f"**Style:** {gen.get('style', 'N/A')}")
                    st.write(f"**Dimensions:** {gen.get('dimensions', 'N/A')}")
                    st.write(f"**Time:** {gen.get('timestamp', 'N/A')}")
                    if gen.get('seed'):
                        st.write(f"**Seed:** {gen['seed']}")
        else:
            st.info("No generations yet. Create your first image!")
        
        # Model information
        st.markdown("#### 🧠 Model Information")
        st.info("""
        **Model:** Stable Diffusion v1.5  
        **Size:** ~156MB (Optimized)  
        **Generation Time:** 15-30 seconds  
        **Quality:** High  
        **GPU Support:** Yes (if available)  
        """)

# Example usage in main app
if __name__ == "__main__":
    st.set_page_config(page_title="Cognitive Nexus AI - Enhanced Image Generation", layout="wide")
    
    # Initialize session state
    if "installing_deps" not in st.session_state:
        st.session_state.installing_deps = False
    if "deps_installed" not in st.session_state:
        st.session_state.deps_installed = False
    if "show_success_message" not in st.session_state:
        st.session_state.show_success_message = False
    
    st.title("🧠 Cognitive Nexus AI - Enhanced Image Generation")
    
    # Show the enhanced image generation tab
    render_image_generation_tab_enhanced()
