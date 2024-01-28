from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Query
from PIL import Image
import requests
import base64
from config import url, pil_to_base64, llama_host, get_llama_request_body
from rq import Queue
from redis import Redis
from jobs import txt2img, upscale,img2img
from rq.registry import FinishedJobRegistry
import io
import torchaudio
from audiocraft.data.audio import audio_write
from audiocraft.models.musicgen import MusicGen
from audiocraft.models import AudioGen
from pydantic import BaseModel
from fastapi.responses import FileResponse
import time
from fastapi.middleware.cors import CORSMiddleware
import torch
import gc
from transformers import AutoProcessor, BarkModel
import soundfile as sf
from typing import Optional
from pathlib import Path
from pydub import AudioSegment
import numpy as np
from clone_vc import load_codec_model, generate_text_semantic
from hubert_manager import HuBERTManager
from pre_kmeans_hubert import CustomHubert
from customtokenizer import CustomTokenizer
from encodec.utils import convert_audio


app = FastAPI()
redis_conn = Redis()
ai_queue = Queue('ai_queue', connection=redis_conn)
registry = FinishedJobRegistry(queue=ai_queue)



app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
   return {"message": "Hello World"}

@app.get("/txt2img")
async def txt2img_endpoint(request_body: dict):
   prompt = request_body.get("prompt")
   if not prompt:
      return {"error": "Missing mandatory 'prompt' field in the request."}

   query = {
      "prompt": request_body.get("prompt"),
      "negative_prompt": request_body.get("negative_prompt", ""),
      "seed": request_body.get("seed", -1),
      "cfg_scale": request_body.get("cfg_scale", 7),
      "sampler_index": request_body.get("sampler_index", "DPM++ 2M Karras"),
      "width": request_body.get("width", 512),
      "height": request_body.get("height", 512),
      "steps": request_body.get("steps", 25),
      "n_iter": request_body.get("n_iter", 1),
      "Asymmetric_tiling":{"args":[True,True,False,0,-1]},
      "tiling":request_body.get("tiling", False),
       "hr_sampler_name": request_body.get("hr_sampler_name", "Euler"),
       "sampler_name": request_body.get("sampler_name", "Euler")
   }

   job = ai_queue.enqueue(txt2img, query)
   return {
      'task_id': job.id,
   }

@app.get("/img2img")
async def img2img_endpoint(request_body: dict):
   prompt = request_body.get("prompt")
   init_images = request_body.get("init_images")
   if not prompt:
      return {"error": "Missing mandatory 'prompt' field in the request."}
   if not init_images:
      return {"error": "Missing mandatory 'init_images' field in the request."}

   query = {
      "prompt": request_body.get("prompt"),
      "negative_prompt": request_body.get("negative_prompt", ""),
      "seed": request_body.get("seed", -1),
      "cfg_scale": request_body.get("cfg_scale", 7),
      "sampler_index": request_body.get("sampler_index", "DPM++ 2M Karras"),
      "width": request_body.get("width", 512),
      "height": request_body.get("height", 512),
      "steps": request_body.get("steps", 25),
      "n_iter": request_body.get("n_iter", 1),
      "Asymmetric_tiling":{"args":[True,True,False,0,-1]},
      "tiling":request_body.get("tiling", False),
      "hr_sampler_name": request_body.get("hr_sampler_name", "Euler"),
      "sampler_name": request_body.get("sampler_name", "Euler"),
      "init_images":request_body.get("init_images")
   }

   job = ai_queue.enqueue(img2img, query)
   return {
      'task_id': job.id,
   }


@app.get("/txt2img/{task_id}")
async def txt2img_endpoint(task_id: str):
   job = ai_queue.fetch_job(task_id)
   if not job:
      raise HTTPException(status_code=404, detail="request not found")

   if job.get_status() == 'failed':
      raise HTTPException(status_code=500, detail="Job failed")

   if job.get_status() == 'started':
      response = requests.get(url=f'{url}/sdapi/v1/progress?skip_current_image=true')
      return {
         'status': job.get_status(),
         'progress': response.json()['progress'],
         'eta_relative': response.json()['eta_relative']
      }

   if job.get_status() != 'finished':
      return {
         'status': job.get_status()
      }

   return {
      'status': job.get_status(),
      'result': job.result
   }

import base64

@app.get("/upscale")
async def upscale_endpoint(request_body: dict):
   img_base64 = request_body.get("img_base64")
   if not img_base64:
      return {"error": "Missing mandatory 'img_base64' field in the request."}

   img_data = base64.b64decode(img_base64)
   pil_image = Image.open(io.BytesIO(img_data))

   request_body = {
      "denoising_strength": 0.1,
      "init_images": [pil_to_base64(pil_image)],
      "script_args": ["", 64, "ESRGAN_4x", 2],
      "script_name": "SD upscale"
   }

   job = ai_queue.enqueue(upscale, request_body)
   return {
      'task_id': job.id,
   }

@app.get("/models")
async def current_model():
   response = requests.get(f'{url}/sdapi/v1/sd-models')
   titles = []

   for item in response.json():
      titles.append(item['model_name'])
   return {"models": titles}

@app.get("/current-model")
async def current_model():
   opt = requests.get(url=f'{url}/sdapi/v1/options')
   current_model = opt.json()["sd_model_checkpoint"]
   return {"message": f"current loaded model is {current_model}"}

@app.get("/set-model")
async def switch_model(model_name: str):
   opt = requests.get(url=f'{url}/sdapi/v1/options')
   opt_json = opt.json()
   opt_json['sd_model_checkpoint'] = model_name
   response = requests.post(url=f'{url}/sdapi/v1/options', json=opt_json)
   if response.status_code == 200:
      return {"message": f"Successfully switched to {model_name} model"}
   else:
      print(response.json())
      return {"message": "Error occurred while switching the model"}

@app.get("/progress")
async def get_progress():
   response = requests.get(url=f'{url}/sdapi/v1/progress?skip_current_image=true')
   return response.json()

class MusicRequest(BaseModel):
    prompt: str
    duration: float = 10

@app.post("/generatemusic")
async def generate_music(request: Request, music_request: MusicRequest):
    model = MusicGen.get_pretrained('facebook/musicgen-medium')
    prompt = music_request.prompt
    duration = music_request.duration

    if not prompt:
        return {"error": "Missing mandatory 'prompt' field in the request."}

    if duration > 60:
        duration = 60
    if not duration:
        duration = 10

    model.set_generation_params(duration=duration)
    descriptions = [prompt]
    cond_wav = model.generate(descriptions)

    mp3_files = []

    for idx, wav_file in enumerate(cond_wav):
        mp3_path = f'output/{idx}'
        audio_write(mp3_path, wav_file.cpu(), sample_rate=model.sample_rate,format='mp3',mp3_rate=320)
        mp3_files.append(mp3_path + '.mp3')

    del model
    del cond_wav
    torch.cuda.empty_cache()
    gc.collect()

    return FileResponse(mp3_files[0], media_type="audio/mp3")


class AudioRequest(BaseModel):
    prompt: str
    duration: float = 10

@app.post("/generateaudio")
async def generate_audio(request: Request, audio_request: AudioRequest):
    model = AudioGen.get_pretrained('facebook/audiogen-medium')
    prompt = audio_request.prompt
    duration = audio_request.duration

    if not prompt:
        return {"error": "Missing mandatory 'prompt' field in the request."}

    if duration > 60:
        duration = 60
    if not duration:
        duration = 10

    model.set_generation_params(duration=duration)
    descriptions = [prompt]
    cond_wav = model.generate(descriptions)

    mp3_files = []

    for idx, wav_file in enumerate(cond_wav):
        mp3_path = f'output/{idx}'
        audio_write(mp3_path, wav_file.cpu(), sample_rate=model.sample_rate,strategy="loudness", loudness_compressor=True,format='mp3',mp3_rate=320)
        mp3_files.append(mp3_path + '.mp3')

    del model
    del cond_wav
    torch.cuda.empty_cache()
    gc.collect()

    return FileResponse(mp3_files[0], media_type="audio/mp3")







@app.get('/png-info')
async def png_info(payload: dict):
   response = requests.post(url=f'{url}/sdapi/v1/png-info', json=payload)
   return response

@app.get('/llama')
async def llama(prompt: str):
   payload = get_llama_request_body(prompt)
   response = requests.post(url=f'{llama_host}/api/v1/generate', json=payload)
   return response

class AudioRequest(BaseModel):
      prompt: str
      voice_preset: Optional[str] = None

@app.post("/generate_bark")
def generate_audio_endpoint(request: AudioRequest):

      # Bark AI config
      device = "cuda" if torch.cuda.is_available() else "cpu"
      processor = AutoProcessor.from_pretrained("suno/bark")
      model = BarkModel.from_pretrained("suno/bark").to(device)

      voice_preset = f'prompts/{request.voice_preset}' if request.voice_preset else "v2/en_speaker_6"
      inputs = processor(request.prompt, voice_preset=voice_preset, return_tensors="pt")

      inputs = {k: v.to(device) for k, v in inputs.items()}

      with torch.no_grad():
         audio_array = model.generate(**inputs)

      audio_array = audio_array.cpu().numpy().squeeze()

      buffer = io.BytesIO()
      sf.write(buffer, audio_array, 24000, format='WAV')
      buffer.seek(0)

      audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')

      del processor
      del model
      torch.cuda.empty_cache()
      gc.collect()

      return {"audio_base64": audio_base64}

@app.post("/generate_npz")
async def generate_npz(file: UploadFile = File(...), voice_name: str = Query(...)):
      # Save the uploaded file
      temp_dir = Path('temp_files')
      temp_dir.mkdir(exist_ok=True)
      file_path = temp_dir / file.filename
      with open(file_path, 'wb') as buffer:
         buffer.write(file.file.read())

      # Convert MP3 to WAV if necessary
      if file_path.suffix.lower() == '.mp3':
         audio = AudioSegment.from_mp3(file_path)
         wav_path = file_path.with_suffix('.wav')
         audio.export(wav_path, format='wav')
         file_path.unlink()
         file_path = wav_path

      device = 'cuda' # or 'cpu'
      model = load_codec_model(use_gpu=True if device == 'cuda' else False)

      hubert_manager = HuBERTManager()
      hubert_manager.make_sure_hubert_installed()
      hubert_manager.make_sure_tokenizer_installed()

      # Load the HuBERT model
      hubert_model = CustomHubert(checkpoint_path='data/models/hubert/hubert.pt').to(device)

      # Load the CustomTokenizer model
      tokenizer = CustomTokenizer.load_from_checkpoint('data/models/hubert/tokenizer.pth').to(device)

      # Use the uploaded file
      wav, sr = torchaudio.load(file_path)
      wav = convert_audio(wav, sr, model.sample_rate, model.channels)
      wav = wav.to(device)

      semantic_vectors = hubert_model.forward(wav, input_sample_hz=model.sample_rate)
      semantic_tokens = tokenizer.get_token(semantic_vectors)

      with torch.no_grad():
         encoded_frames = model.encode(wav.unsqueeze(0))
      codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()  # [n_q, T]

      # move codes to cpu
      codes = codes.cpu().numpy()
      # move semantic tokens to cpu
      semantic_tokens = semantic_tokens.cpu().numpy()

      output_path = 'prompts/' + voice_name + '.npz'
      np.savez(output_path, fine_prompt=codes, coarse_prompt=codes[:2, :], semantic_prompt=semantic_tokens)

      return output_path
