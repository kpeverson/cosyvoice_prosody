# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import time
from typing import Generator
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml
from modelscope import snapshot_download
import torch
import torchaudio
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.cli.model import CosyVoiceModel, CosyVoice2Model, CosyVoice3Model
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.class_utils import get_model_type


class CosyVoice:

    def __init__(self, model_dir, load_jit=False, load_trt=False, fp16=False, trt_concurrent=1):
        self.model_dir = model_dir
        self.fp16 = fp16
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        hyper_yaml_path = '{}/cosyvoice.yaml'.format(model_dir)
        if not os.path.exists(hyper_yaml_path):
            raise ValueError('{} not found!'.format(hyper_yaml_path))
        with open(hyper_yaml_path, 'r') as f:
            configs = load_hyperpyyaml(f)
        assert get_model_type(configs) == CosyVoiceModel, 'do not use {} for CosyVoice initialization!'.format(model_dir)
        self.frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          '{}/campplus.onnx'.format(model_dir),
                                          '{}/speech_tokenizer_v1.onnx'.format(model_dir),
                                          '{}/spk2info.pt'.format(model_dir),
                                          configs['allowed_special'])
        self.sample_rate = configs['sample_rate']
        if torch.cuda.is_available() is False and (load_jit is True or load_trt is True or fp16 is True):
            load_jit, load_trt, fp16 = False, False, False
            logging.warning('no cuda device, set load_jit/load_trt/fp16 to False')
        self.model = CosyVoiceModel(configs['llm'], configs['flow'], configs['hift'], fp16)
        self.model.load('{}/llm.pt'.format(model_dir),
                        '{}/flow.pt'.format(model_dir),
                        '{}/hift.pt'.format(model_dir))
        if load_jit:
            self.model.load_jit('{}/llm.text_encoder.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
                                '{}/llm.llm.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
                                '{}/flow.encoder.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'))
        if load_trt:
            self.model.load_trt('{}/flow.decoder.estimator.{}.mygpu.plan'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
                                '{}/flow.decoder.estimator.fp32.onnx'.format(model_dir),
                                trt_concurrent,
                                self.fp16)
        del configs

    def list_available_spks(self):
        spks = list(self.frontend.spk2info.keys())
        return spks

    def add_zero_shot_spk(self, prompt_text, prompt_wav, zero_shot_spk_id):
        assert zero_shot_spk_id != '', 'do not use empty zero_shot_spk_id'
        model_input = self.frontend.frontend_zero_shot('', prompt_text, prompt_wav, self.sample_rate, '')
        del model_input['text']
        del model_input['text_len']
        self.frontend.spk2info[zero_shot_spk_id] = model_input
        return True

    def save_spkinfo(self):
        torch.save(self.frontend.spk2info, '{}/spk2info.pt'.format(self.model_dir))

    def inference_sft(self, tts_text, spk_id, stream=False, speed=1.0, text_frontend=True):
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_sft(i, spk_id)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_zero_shot(self, tts_text, prompt_text, prompt_wav, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True):
        prompt_text = self.frontend.text_normalize(prompt_text, split=False, text_frontend=text_frontend)
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            if (not isinstance(i, Generator)) and len(i) < 0.5 * len(prompt_text):
                logging.warning('synthesis text {} too short than prompt text {}, this may lead to bad performance'.format(i, prompt_text))
            model_input = self.frontend.frontend_zero_shot(i, prompt_text, prompt_wav, self.sample_rate, zero_shot_spk_id)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_cross_lingual(self, tts_text, prompt_wav, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True):
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_cross_lingual(i, prompt_wav, self.sample_rate, zero_shot_spk_id)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_instruct(self, tts_text, spk_id, instruct_text, stream=False, speed=1.0, text_frontend=True):
        assert self.__class__.__name__ == 'CosyVoice', 'inference_instruct is only implemented for CosyVoice!'
        instruct_text = self.frontend.text_normalize(instruct_text, split=False, text_frontend=text_frontend)
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_instruct(i, spk_id, instruct_text)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_vc(self, source_wav, prompt_wav, stream=False, speed=1.0):
        model_input = self.frontend.frontend_vc(source_wav, prompt_wav, self.sample_rate)
        start_time = time.time()
        for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
            speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
            logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
            yield model_output
            start_time = time.time()


class CosyVoice2(CosyVoice):

    def __init__(self, model_dir, load_jit=False, load_trt=False, load_vllm=False, fp16=False, trt_concurrent=1, prosody_encoder_path=''):
        self.model_dir = model_dir
        self.fp16 = fp16
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        hyper_yaml_path = '{}/cosyvoice2.yaml'.format(model_dir)
        if not os.path.exists(hyper_yaml_path):
            raise ValueError('{} not found!'.format(hyper_yaml_path))
        with open(hyper_yaml_path, 'r') as f:
            configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})
        assert get_model_type(configs) == CosyVoice2Model, 'do not use {} for CosyVoice2 initialization!'.format(model_dir)
        self.frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          '{}/campplus.onnx'.format(model_dir),
                                          '{}/speech_tokenizer_v2.onnx'.format(model_dir),
                                          '{}/spk2info.pt'.format(model_dir),
                                          configs['allowed_special'])
        self.sample_rate = configs['sample_rate']
        if torch.cuda.is_available() is False and (load_jit is True or load_trt is True or load_vllm is True or fp16 is True):
            load_jit, load_trt, load_vllm, fp16 = False, False, False, False
            logging.warning('no cuda device, set load_jit/load_trt/load_vllm/fp16 to False')
        self.model = CosyVoice2Model(configs['llm'], configs['flow'], configs['hift'], fp16)
        if prosody_encoder_path:
            self.model.llm.init_prosody_encoder(prosody_encoder_path)
            self.model.flow.encoder.init_prosody_encoder(prosody_encoder_path)
            logging.info('Initialized prosody encoder from {}'.format(prosody_encoder_path))
        self.model.load('{}/llm.pt'.format(model_dir),
                        '{}/flow.pt'.format(model_dir),
                        '{}/hift.pt'.format(model_dir))
        if load_vllm:
            self.model.load_vllm('{}/vllm'.format(model_dir))
        if load_jit:
            self.model.load_jit('{}/flow.encoder.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'))
        if load_trt:
            self.model.load_trt('{}/flow.decoder.estimator.{}.mygpu.plan'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
                                '{}/flow.decoder.estimator.fp32.onnx'.format(model_dir),
                                trt_concurrent,
                                self.fp16)
        del configs

    def inference_zero_shot_with_prosody_tokens(self, tts_text, prompt_text, prompt_wav, prosody_tokens, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True):
        """Zero-shot TTS conditioned on prosody tokens (list of int, vocabulary matches training km clusters)."""
        prompt_text = self.frontend.text_normalize(prompt_text, split=False, text_frontend=text_frontend)
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_zero_shot_with_prosody(i, prompt_text, prompt_wav, self.sample_rate, prosody_tokens, zero_shot_spk_id)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_zero_shot_with_prosody_tokens_spkemb_only(self, tts_text, prompt_wav_path, prosody_tokens, stream=False, speed=1.0, text_frontend=True):
        """Zero-shot TTS with prosody tokens, using prompt_wav only for speaker embedding.

        No prompt text, no token prefix, no mel condition in the flow model.
        Speaker identity comes from the x-vector; prosody from the provided tokens.
        """
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            tts_text_token, tts_text_token_len = self.frontend._extract_text_token(i)
            embedding = self.frontend._extract_spk_embedding(prompt_wav_path)
            prosody_token = torch.tensor([prosody_tokens], dtype=torch.int32)
            prosody_token_len = torch.tensor([len(prosody_tokens)], dtype=torch.int32)
            model_input = {
                'text': tts_text_token,
                'text_len': tts_text_token_len,
                'llm_embedding': embedding,
                'flow_embedding': embedding,
                'prosody_token': prosody_token,
                'prosody_token_len': prosody_token_len,
            }
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_zero_shot_with_prosody_tokens_no_spkemb(self, tts_text, prompt_text, prompt_wav_path, prosody_tokens, stream=False, speed=1.0, text_frontend=True):
        """Zero-shot TTS with prosody tokens, without a speaker embedding.

        Speaker characteristics must be replicated by the flow model from the prompt
        speech tokens and mel-spectrogram alone. The x-vector is zeroed out.
        """
        prompt_text = self.frontend.text_normalize(prompt_text, split=False, text_frontend=text_frontend)
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_zero_shot_with_prosody(i, prompt_text, prompt_wav_path, self.sample_rate, prosody_tokens)
            model_input['llm_embedding'] = torch.zeros(1, 192)
            model_input['flow_embedding'] = torch.zeros(1, 192)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def _wav_to_prosody_emb(self, wav_path):
        """Extract prosody embedding from a wav file path via glottal source analysis."""
        from cosyvoice.prosody.glottal import extract_glottal_source
        wav, sr = torchaudio.load(wav_path)
        if wav.ndim == 2 and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav_16k = torchaudio.functional.resample(wav, sr, 16000) if sr != 16000 else wav
        glottal = extract_glottal_source(wav_16k.squeeze(0), src_sample_rate=16000).unsqueeze(0)
        return self.model.llm.extract_prosody_emb(glottal.to(self.model.device))  # [1, T', D]

    def inference_zero_shot_with_prosody_encoder(self, tts_text, prompt_text, prompt_wav_path, tts_wav_path, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True):
        """Zero-shot TTS with prosody features extracted on-the-fly from a prosody reference.

        prompt_wav_path provides speaker identity (x-vector) and the acoustic prompt for the flow model.
        tts_wav_path provides the prosody style. Prosody features are encoded separately for each
        audio and concatenated, covering the full [prompt_text + tts_text] span without mixing
        self-attention context across the two audio sources.
        """
        prompt_text = self.frontend.text_normalize(prompt_text, split=False, text_frontend=text_frontend)
        prompt_prosody_emb = self._wav_to_prosody_emb(prompt_wav_path)  # [1, T_prompt', D]
        tts_prosody_emb = self._wav_to_prosody_emb(tts_wav_path)        # [1, T_tts', D]
        prosody_emb = torch.cat([prompt_prosody_emb, tts_prosody_emb], dim=1)

        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_zero_shot(i, prompt_text, prompt_wav_path, self.sample_rate, zero_shot_spk_id)
            model_input['prosody_emb'] = prosody_emb
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_zero_shot_with_prosody_encoder_spkemb_only(self, tts_text, prompt_wav_path, tts_wav_path, stream=False, speed=1.0, text_frontend=True):
        """Zero-shot TTS with prosody encoder, using prompt_wav only for speaker embedding.

        No prompt text, no token prefix, no mel condition in the flow model.
        Speaker identity comes from the x-vector; prosody from the encoder applied to both wavs.
        """
        # prosody_emb = torch.cat([
        #     self._wav_to_prosody_emb(prompt_wav_path),
        #     self._wav_to_prosody_emb(tts_wav_path),
        # ], dim=1)
        prosody_emb = self._wav_to_prosody_emb(tts_wav_path)  # only use tts_wav for prosody, to avoid potential speaker info leakage from prompt_wav

        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            tts_text_token, tts_text_token_len = self.frontend._extract_text_token(i)
            embedding = self.frontend._extract_spk_embedding(prompt_wav_path)
            model_input = {
                'text': tts_text_token,
                'text_len': tts_text_token_len,
                'llm_embedding': embedding,
                'flow_embedding': embedding,
                'prosody_emb': prosody_emb,
            }
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_zero_shot_with_prosody_encoder_no_spkemb(self, tts_text, prompt_text, prompt_wav_path, tts_wav_path, stream=False, speed=1.0, text_frontend=True):
        """Zero-shot TTS with prosody encoder, without a speaker embedding.

        Speaker characteristics must be replicated by the flow model from the prompt
        speech tokens and mel-spectrogram alone. The x-vector is zeroed out.
        """
        prompt_text = self.frontend.text_normalize(prompt_text, split=False, text_frontend=text_frontend)
        prosody_emb = torch.cat([
            self._wav_to_prosody_emb(prompt_wav_path),
            self._wav_to_prosody_emb(tts_wav_path),
        ], dim=1)

        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_zero_shot(i, prompt_text, prompt_wav_path, self.sample_rate, zero_shot_spk_id='')
            model_input['llm_embedding'] = torch.zeros(1, 192)
            model_input['flow_embedding'] = torch.zeros(1, 192)
            model_input['prosody_emb'] = prosody_emb
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_zero_shot_spkemb_only(self, tts_text, prompt_wav_path, stream=False, speed=1.0, text_frontend=True):
        """Zero-shot TTS using prompt_wav only for speaker embedding extraction.

        No prompt text, no token prefix, no mel condition in the flow model.
        Speaker identity comes entirely from the x-vector.
        """
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            tts_text_token, tts_text_token_len = self.frontend._extract_text_token(i)
            embedding = self.frontend._extract_spk_embedding(prompt_wav_path)
            model_input = {
                'text': tts_text_token,
                'text_len': tts_text_token_len,
                'llm_embedding': embedding,
                'flow_embedding': embedding,
            }
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_zero_shot_no_spkemb(self, tts_text, prompt_text, prompt_wav_path, stream=False, speed=1.0, text_frontend=True):
        """Zero-shot TTS without a speaker embedding.

        Speaker characteristics must be replicated by the flow model from the prompt
        speech tokens and mel-spectrogram alone. The x-vector is zeroed out.
        """
        prompt_text = self.frontend.text_normalize(prompt_text, split=False, text_frontend=text_frontend)
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_zero_shot(i, prompt_text, prompt_wav_path, self.sample_rate, zero_shot_spk_id='')
            model_input['llm_embedding'] = torch.zeros(1, 192)
            model_input['flow_embedding'] = torch.zeros(1, 192)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_instruct2(self, tts_text, instruct_text, prompt_wav, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True):
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_instruct2(i, instruct_text, prompt_wav, self.sample_rate, zero_shot_spk_id)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()


class CosyVoice3(CosyVoice2):

    def __init__(self, model_dir, load_trt=False, load_vllm=False, fp16=False, trt_concurrent=1):
        self.model_dir = model_dir
        self.fp16 = fp16
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        hyper_yaml_path = '{}/cosyvoice3.yaml'.format(model_dir)
        if not os.path.exists(hyper_yaml_path):
            raise ValueError('{} not found!'.format(hyper_yaml_path))
        with open(hyper_yaml_path, 'r') as f:
            configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})
        assert get_model_type(configs) == CosyVoice3Model, 'do not use {} for CosyVoice3 initialization!'.format(model_dir)
        self.frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          '{}/campplus.onnx'.format(model_dir),
                                          '{}/speech_tokenizer_v3.onnx'.format(model_dir),
                                          '{}/spk2info.pt'.format(model_dir),
                                          configs['allowed_special'])
        self.sample_rate = configs['sample_rate']
        if torch.cuda.is_available() is False and (load_trt is True or fp16 is True):
            load_trt, fp16 = False, False
            logging.warning('no cuda device, set load_trt/fp16 to False')
        self.model = CosyVoice3Model(configs['llm'], configs['flow'], configs['hift'], fp16)
        self.model.load('{}/llm.pt'.format(model_dir),
                        '{}/flow.pt'.format(model_dir),
                        '{}/hift.pt'.format(model_dir))
        if load_vllm:
            self.model.load_vllm('{}/vllm'.format(model_dir))
        if load_trt:
            if self.fp16 is True:
                logging.warning('DiT tensorRT fp16 engine have some performance issue, use at caution!')
            self.model.load_trt('{}/flow.decoder.estimator.{}.mygpu.plan'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
                                '{}/flow.decoder.estimator.fp32.onnx'.format(model_dir),
                                trt_concurrent,
                                self.fp16)
        del configs


def AutoModel(**kwargs):
    if not os.path.exists(kwargs['model_dir']):
        kwargs['model_dir'] = snapshot_download(kwargs['model_dir'])
    if os.path.exists('{}/cosyvoice.yaml'.format(kwargs['model_dir'])):
        return CosyVoice(**kwargs)
    elif os.path.exists('{}/cosyvoice2.yaml'.format(kwargs['model_dir'])):
        return CosyVoice2(**kwargs)
    elif os.path.exists('{}/cosyvoice3.yaml'.format(kwargs['model_dir'])):
        return CosyVoice3(**kwargs)
    else:
        raise TypeError('No valid model type found!')
