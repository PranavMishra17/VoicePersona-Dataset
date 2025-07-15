"""
Prompt templates for Qwen2-Audio voice analysis
"""

class VoiceDescriptionPrompts:
    """Collection of prompts for extracting voice characteristics"""
    
    @staticmethod
    def get_analysis_prompt() -> str:
        """Main voice analysis prompt"""
        return """<|audio_bos|><|AUDIO|><|audio_eos|>Please provide a detailed analysis of this voice recording. Focus on:

1. **Voice Characteristics**: Describe the tone, pitch, timbre, and overall quality of the voice. Is it deep, high-pitched, raspy, smooth, breathy, nasal, clear, or muffled?

2. **Speaking Style**: How does the person speak? Is it fast or slow? Are they articulate or mumbling? Do they have a formal or casual speaking style? Any notable pauses or hesitations?

3. **Emotional Tone**: What emotions or mood can you detect in the voice? Does the speaker sound confident, nervous, happy, sad, excited, calm, tired, or energetic?

4. **Distinctive Features**: Are there any unique characteristics that make this voice memorable? Any vocal quirks, speech patterns, or notable pronunciations?

5. **Voice Age and Maturity**: Based on vocal characteristics alone, describe the perceived age or maturity level of the voice.

6. **Overall Impression**: How would you help someone imagine this voice if they wanted to recreate or find a similar voice? What kind of person or character would have this voice?

Please provide a comprehensive description that would help someone vividly imagine this voice."""

    @staticmethod
    def get_character_prompt() -> str:
        """Character/persona description prompt"""
        return """<|audio_bos|><|AUDIO|><|audio_eos|>Based on this voice recording, describe what kind of character or persona this voice would suit. Consider:

- What type of character in a story, game, or animation would have this voice?
- What profession or role would match this voice?
- What personality traits does this voice suggest?
- In what context would this voice be most fitting?

Provide a creative but grounded character description based solely on the vocal qualities."""

    @staticmethod
    def get_technical_prompt() -> str:
        """Technical voice analysis prompt"""
        return """<|audio_bos|><|AUDIO|><|audio_eos|>Provide a technical analysis of this voice recording:

1. **Frequency Range**: Estimate the fundamental frequency range and dominant frequencies
2. **Voice Quality**: Identify technical qualities (breathiness, roughness, strain, resonance)
3. **Articulation**: Assess clarity of consonants, vowel quality, and overall intelligibility
4. **Prosody**: Analyze rhythm, intonation patterns, stress patterns, and pacing
5. **Acoustic Features**: Note any specific acoustic characteristics that stand out

Focus on objective, measurable aspects of the voice."""

    @staticmethod
    def get_combined_prompt() -> str:
        """Combined comprehensive analysis"""
        return """<|audio_bos|><|AUDIO|><|audio_eos|>Analyze this voice recording comprehensively:

**Part 1 - Voice Profile:**
- Physical characteristics (pitch, tone, timbre, quality)
- Speaking patterns and style
- Emotional undertones and energy level

**Part 2 - Character Match:**
- What character or persona fits this voice?
- Suitable contexts and applications

**Part 3 - Unique Identifiers:**
- Distinctive features that make this voice memorable
- How to recreate or find similar voices

Provide a detailed description that captures the essence of this voice for creative applications."""

# Prompt selection based on use case
PROMPT_TYPES = {
    "analysis": VoiceDescriptionPrompts.get_analysis_prompt,
    "character": VoiceDescriptionPrompts.get_character_prompt,
    "technical": VoiceDescriptionPrompts.get_technical_prompt,
    "combined": VoiceDescriptionPrompts.get_combined_prompt
}

def get_prompt(prompt_type: str = "analysis") -> str:
    """Get prompt by type"""
    if prompt_type not in PROMPT_TYPES:
        raise ValueError(f"Invalid prompt type. Choose from: {list(PROMPT_TYPES.keys())}")
    return PROMPT_TYPES[prompt_type]()