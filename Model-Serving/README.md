### vLLM Backend

'''
pip install torch torchvision torchaudio
pip install "transformers==4.40.2"
pip install "xinference[llamacpp]"
xinference --help
'''

### Start Xinference Server

'''
xinference-local -H 0.0.0.0 --port 9997
'''

### Launch a model (vLLM)

'''
xinference launch \
  --model-name qwen2:1.5b \
  --model-format gguf

curl http://127.0.0.1:9997/v1/models
'''


### Docker

'''
docker pull xprobe/xinference

docker save xprobe/xinference > xinfer.tar 

docker run -e XINFERENCE_MODEL_SRC=local -p 9002:9997 --gpus all -e API_HOST=0.0.0.0 -v /data/aihub/:/models xprobe/xinference:latest xinference-local -H 0.0.0.0 --log-level debug --port 9997

docker logs -f xinfer
'''

