# VoicePersona Dataset

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-blue)](https://huggingface.co/datasets)
[![License: CC0](https://img.shields.io/badge/License-CC0%201.0-lightgrey.svg)](http://creativecommons.org/publicdomain/cc0/1.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A comprehensive voice persona dataset for character consistency in voice synthesis, generated using advanced audio-language models.

## 📋 Overview

VoicePersona Dataset serves as the **training foundation** for [**VoiceForge**](https://github.com/PranavMishra17/VoiceForge--Forge-Character-Voices-from-Pure-Text) - an AI architecture that generates character voices from pure text descriptions.

**The Connection:**
- **VoicePersona** provides detailed voice characteristics and personality profiles
- **VoiceForge** uses this data to learn text→voice mapping for character consistency
- Together, they enable voice synthesis from natural language descriptions alone

**VoiceForge Applications:**
- 🎮 Game developers creating unique NPCs
- 📚 Interactive storytelling applications  
- 🎬 Content creators needing character voices
- 🔬 Researchers in voice synthesis

This dataset bridges the gap between voice analysis and synthesis, providing the structured training data needed for consistent character voice generation without audio samples or voice actors.

## 🎯 What We Do

This pipeline processes audio from multiple voice datasets and generates detailed character profiles using [Qwen2-Audio-7B-Instruct](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct). The system:

1. **Extracts Voice Characteristics**: Analyzes pitch, tone, timbre, resonance, and speaking patterns
2. **Identifies Demographics**: Estimates gender, age range, and accent
3. **Profiles Personality**: Determines character traits and suitable roles
4. **Maintains Consistency**: Focuses on "how" speakers talk rather than "what" they say

## 📊 Dataset Structure

```
voicepersona_dataset/
├── globe_v2/
│   ├── audio/                    # Original audio files (.wav)
│   ├── globe_v2_descriptions.json
│   └── globe_v2_hf_dataset/      # HuggingFace format
├── laions/
│   ├── audio/
│   ├── laions_descriptions.json
│   └── laions_hf_dataset/
├── animevox/
│   ├── audio/
│   ├── animevox_descriptions.json
│   └── animevox_hf_dataset/
└── anispeech/
    ├── audio/
    ├── anispeech_descriptions.json
    └── anispeech_hf_dataset/
```

### Sample Output Format
```json
{
  "index": 0,
  "dataset": "globe_v2",
  "speaker_id": "S_000658",
  "transcript": "each member has one share and one vote.",
  "audio_path": "/path/to/audio.wav",
  "duration": 2.9,
  "gender": "female",
  "age": "thirties",
  "accent": "New Zealand English",
  "voice_description": "Detailed voice profile including vocal qualities, speaking style, emotional undertones, character impression, and distinctive features...",
  "processing_timestamp": "2025-07-17T01:57:41.590598"
}
```

## 🗃️ Source Datasets

| Dataset | Description | Samples | Link |
|---------|-------------|---------|------|
| **GLOBE_V2** | Global accents, 52 accents × 3 genders | ~50K | [MushanW/GLOBE_V2](https://huggingface.co/datasets/MushanW/GLOBE_V2) |
| **Laions Got Talent** | Emotional speech synthesis | ~15K | [laion/laions_got_talent](https://huggingface.co/datasets/laion/laions_got_talent) |
| **AnimeVox** | Anime character voices | ~10K | [taresh18/AnimeVox](https://huggingface.co/datasets/taresh18/AnimeVox) |
| **AniSpeech** | Anime speech synthesis | ~8K | [ShoukanLabs/AniSpeech](https://huggingface.co/datasets/ShoukanLabs/AniSpeech) |

## 🤖 Model Used

**Qwen2-Audio-7B-Instruct**: [Alibaba's multimodal audio-language model](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct)
- 7B parameters optimized for audio understanding
- Supports voice chat and audio analysis
- Multilingual capabilities (8+ languages)

## 🚀 Usage

### Installation
```bash
git clone <repository>
cd voicepersona-dataset
pip install -r requirements.txt
```

### Quick Start
```bash
# List available datasets
python main.py list

# Test processing
python main.py test globe_v2 --samples 5

# Process full dataset
python main.py process laions --max 1000

# Analyze results
python main.py analyze animevox
```

### Configuration
Key settings in `src/config.py`:
- `USE_QUANTIZATION`: Enable 4-bit quantization for 6GB VRAM
- `USE_STREAMING`: Stream datasets without full download
- `CHECKPOINT_INTERVAL`: Auto-save frequency

## 📈 Dataset Statistics

- **Total Samples**: 80K+ voice samples across 4 datasets
- **Languages**: 8+ languages and 52+ accent variations
- **Demographics**: Balanced gender and age distributions
- **Domains**: Conversational, emotional, anime, and synthetic speech

## 🔧 System Requirements

**Minimum:**
- GPU: 6GB VRAM (RTX 3060+)
- RAM: 16GB
- Storage: 50GB free space
- CUDA 11.8+

**Recommended:**
- GPU: 12GB+ VRAM
- RAM: 32GB
- Storage: 100GB+ SSD

## 👤 About

This project was developed by me to address the need for consistent voice characterization in AI voice synthesis. By providing detailed voice personas, it enables better character consistency across different speaking contexts and applications.

[![Portfolio](https://img.shields.io/badge/-Portfolio-000?style=for-the-badge&logo=vercel&logoColor=white)](https://portfolio-pranav-mishra-paranoid.vercel.app)
[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/pranavgamedev/)
[![Resume](https://img.shields.io/badge/-Resume-4B0082?style=for-the-badge&logo=read-the-docs&logoColor=white)](https://portfolio-pranav-mishra-paranoid.vercel.app/resume)
[![YouTube](https://img.shields.io/badge/-YouTube-8B0000?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@parano1dgames/featured)

**Research Interests:**
- Voice synthesis and character consistency
- Multimodal AI applications
- Audio-language model development

## 🤝 Contributing

Contributions welcome! Areas for improvement:

**Datasets:**
- Additional voice datasets integration
- Multilingual voice collections
- Emotional speech datasets

**Technical:**
- Model optimization for lower VRAM
- Faster processing pipelines
- Better voice characteristic extraction

**Analysis:**
- Voice similarity metrics
- Character consistency evaluation
- Demographic bias analysis

### How to Contribute
1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push branch (`git push origin feature/improvement`)
5. Open Pull Request

## 📄 License

This project is licensed under the CC0 1.0 Universal License - see the [LICENSE](LICENSE) file for details.

**CC0 1.0 Universal Summary:**
- ✅ Commercial use
- ✅ Modification  
- ✅ Distribution
- ✅ Private use
- ❌ No warranties or liability

## 🙏 Acknowledgments

- **Qwen Team** for the Qwen2-Audio model
- **Dataset Contributors**: GLOBE_V2, Laions, AnimeVox, AniSpeech teams
- **HuggingFace** for dataset hosting and tools
- **Open Source Community** for supporting libraries

## 📞 Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{voicepersona2025,
  title={VoicePersona Dataset: Comprehensive Voice Character Profiles for Synthesis Consistency},
  author={Pranav Mishra},
  year={2025},
  url={https://github.com/PranavMishra17/VoicePersona-Dataset}
}
```