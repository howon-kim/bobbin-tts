# Voice: https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md

from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
import torch
import os
pipeline = KPipeline(lang_code='a')
text = '''
Welcome AGV. Thank you for delivering the new bobbin. Its time to tape it to new bobbin!
'''

# Create voice directory if it doesn't exist
os.makedirs('./voice', exist_ok=True)

generator = pipeline(text, voice='af_heart', speed = 1.2)
for i, (gs, ps, audio) in enumerate(generator):
    print(i, gs, ps)
    display(Audio(data=audio, rate=24000, autoplay=i==0))
    sf.write('./voice/inspection.wav', audio, 24000)