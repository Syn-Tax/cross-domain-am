import transformers
import librosa
import torchaudio

model = transformers.Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B", trust_remote_code=True)
processor = transformers.AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B", trust_remote_code=True)

prompt = """
You are a 3 class classifier, predicting the relationship between pairs of sentences in a debate. You will classify them as one of support, attack or unrelated using only those words. You will 

A support relationship holds when one argument improves the strength of another.
An attack relationship holds when one argument undermines or reduces the strength of another.

<|audio_bos|><|AUDIO|><|audio_eos|>
"""
audio, _ = librosa.load("data/Moral Maze/Banking/audio/94998.wav", sr=processor.feature_extractor.sample_rate)

inputs = processor(text=prompt, audios=audio, return_tensors="pt")

generated_ids = model.generate(**inputs, max_length=256)
generated_ids = generated_ids[:, inputs.input_ids.size(1):]
response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
