from ..smp import *
import os
import sys
from .base import BaseAPI

APIBASES = {
    'OFFICIAL': "https://api.openai.com/v1/chat/completions",
}


def GPT_context_window(model):
    length_map = {
        'gpt-4-1106-preview': 128000,
        'gpt-4-vision-preview': 128000,
        'gpt-4': 8192,
        'gpt-4-32k': 32768,
        'gpt-4-0613': 8192,
        'gpt-4-32k-0613': 32768,
        'gpt-3.5-turbo-1106': 16385,
        'gpt-3.5-turbo': 4096,
        'gpt-3.5-turbo-16k': 16385,
        'gpt-3.5-turbo-instruct': 4096,
        'gpt-3.5-turbo-0613': 4096,
        'gpt-3.5-turbo-16k-0613': 16385,
        'o3-mini': 4096,
    }
    return length_map.get(model, 4096)


class OpenAIWrapper(BaseAPI):
    is_api: bool = True

    def __init__(self,
                 model: str = 'gpt-3.5-turbo-0613',
                 retry: int = 5,
                 wait: int = 5,
                 key: str = None,
                 verbose: bool = True,
                 system_prompt: str = None,
                 temperature: float = 0,
                 timeout: int = 60,
                 api_base: str = 'OFFICIAL',
                 max_tokens: int = 1024,
                 img_size: int = 512,
                 img_detail: str = 'low',
                 **kwargs):

        self.model = model
        self.cur_idx = 0
        self.fail_msg = 'Failed to obtain answer via API. '
        self.max_tokens = max_tokens
        self.temperature = temperature

        openai_key = os.environ.get('OPENAI_API_KEY', None) if key is None else key
        self.openai_key = openai_key
        assert img_size > 0 or img_size == -1
        self.img_size = img_size
        assert img_detail in ['high', 'low']
        self.img_detail = img_detail

        self.vision = 'vision' in model or 'o3' in model
        self.timeout = timeout

        assert isinstance(openai_key, str) and openai_key.startswith('sk-'), f'Illegal openai_key {openai_key}. Please set the environment variable OPENAI_API_KEY to your openai key. '
        super().__init__(wait=wait, retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

        if api_base in APIBASES:
            self.api_base = APIBASES[api_base]
        elif api_base.startswith('http'):
            self.api_base = api_base
        else:
            self.logger.error("Unknown API Base. ")
            sys.exit(-1)

        if 'OPENAI_API_BASE' in os.environ:
            self.logger.warning("Environment variable OPENAI_API_BASE is set. Will override the api_base arg. ")
            self.api_base = os.environ['OPENAI_API_BASE']

    def prepare_inputs(self, inputs):
        input_msgs = []
        if self.system_prompt is not None:
            input_msgs.append(dict(role='system', content=self.system_prompt))
        if isinstance(inputs, str):
            input_msgs.append(dict(role='user', content=inputs))
            return input_msgs
        assert isinstance(inputs, list)
        dict_flag = [isinstance(x, dict) for x in inputs]
        if np.all(dict_flag):
            input_msgs.extend(inputs)
            return input_msgs
        str_flag = [isinstance(x, str) for x in inputs]
        if np.all(str_flag):
            content_list = []
            has_text = False
            for msg in inputs:
                if msg.startswith('http') or osp.exists(msg):
                    from PIL import Image
                    img = Image.open(msg)
                    b64 = encode_image_to_base64(img, target_size=self.img_size)
                    img_struct = dict(url=f'data:image/jpeg;base64,{b64}', detail=self.img_detail)
                    content_list.append(dict(type='image_url', image_url=img_struct))
                else:
                    content_list.append(dict(type='text', text=msg))
                    has_text = True

            if self.vision and not has_text:
                 content_list.append(dict(type='text', text=' '))
                 
            input_msgs.append(dict(role='user', content=content_list))
            return input_msgs
        raise NotImplementedError("list of list prompt not implemented now. ")

    def generate_inner(self, inputs, **kwargs) -> str:
        input_msgs = self.prepare_inputs(inputs)
        temperature = kwargs.pop('temperature', self.temperature)
        max_tokens = kwargs.pop('max_tokens', self.max_tokens)

        context_window = GPT_context_window(self.model)
        max_tokens = min(max_tokens, context_window - self.get_token_len(inputs))
        if 0 < max_tokens <= 100:
            self.logger.warning('Less than 100 tokens left, may exceed the context window with some additional meta symbols. ')
        if max_tokens <= 0:
            return 0, self.fail_msg + 'Input string longer than context window. ', 'Length Exceeded. '

        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.openai_key}'}
        payload = dict(
            model=self.model,
            messages=input_msgs,
            max_tokens=max_tokens,
            n=1,
            temperature=temperature,
            **kwargs)
        
        response = requests.post(self.api_base, headers=headers, data=json.dumps(payload), timeout=self.timeout * 1.1)
        ret_code = response.status_code
        ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code
        answer = self.fail_msg
        try:
            resp_struct = json.loads(response.text)
            answer = resp_struct['choices'][0]['message']['content'].strip()
        except:
            pass
        return ret_code, answer, response

    def get_token_len(self, inputs) -> int:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(self.model)
        except KeyError:
            self.logger.warning(f"Model {self.model} not found in tiktoken, using cl100k_base for token calculation.")
            enc = tiktoken.get_encoding("cl100k_base")

        if isinstance(inputs, str):
            if inputs.startswith('http') or osp.exists(inputs):
                return 65 if self.img_detail == 'low' else 130
            else:
                return len(enc.encode(inputs))
        elif isinstance(inputs, dict):
            assert 'content' in inputs
            return self.get_token_len(inputs['content'])
        assert isinstance(inputs, list)
        res = 0
        for item in inputs:
            res += self.get_token_len(item)
        return res

class GPT4V(OpenAIWrapper):
    def __init__(self, **kwargs):
        # ** THE FIX IS HERE **
        # We pop 'model' from kwargs. If it exists, its value is discarded.
        # If it doesn't exist, this does nothing and prevents an error.
        kwargs.pop('model', None)
        
        # Now we can safely call the parent constructor with our desired model name.
        # **kwargs will no longer contain a conflicting 'model' key.
        super().__init__(model='o3-mini', **kwargs)

    def generate(self, messages, img_input=None, **kwargs):
        for _ in range(self.retry):
            try:
                import openai

                # === o3-mini or any text-only model fallback ===
                if self.model == 'o3-mini' and img_input is None:
                    response = openai.ChatCompletion.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=1024,
                    )
                    return response["choices"][0]["message"]["content"]

                # === vision model ===
                if img_input is None:
                    raise ValueError(f"{self.model} expects image input but none was provided.")

                # Assume img_input is a base64 string (without data URL prefix)
                image_data_url = f"data:image/jpeg;base64,{img_input}"

                # Format for vision-based models (e.g., gpt-4-vision-preview)
                vision_messages = []
                for i, m in enumerate(messages):
                    if i == 0:
                        vision_messages.append({
                            "role": m["role"],
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image_data_url,
                                        "detail": self.img_detail
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": m["content"]
                                }
                            ]
                        })
                    else:
                        vision_messages.append(m)

                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=vision_messages,
                    temperature=self.temperature,
                    max_tokens=1024,
                )
                return response["choices"][0]["message"]["content"]

            except Exception as e:
                print(f"[GPT4V ERROR] {e}")

        return "Failed to obtain answer via API."

    def interleave_generate(self, ti_list, dataset=None):
        return super(GPT4V, self).generate(ti_list)