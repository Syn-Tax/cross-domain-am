import transformers
import torch 
import torchaudio

model = transformers.Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B", trust_remote_code=True)
processor = transformers.AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B", trust_remote_code=True)

prompt = """
<|audio_bos|><|AUDIO|><|audio_eos|>
"""
audio, _ = torchaudio.load("data/Moral Maze/Banking/audio/94998.wav")

inputs = processor(text=prompt, audios=audio, return_tensors="pt")

generated_ids = model.generate(**inputs, max_length=256)
generated_ids = generated_ids[:, inputs.input_ids.size(1):]
response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
