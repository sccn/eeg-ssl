
# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import requests
import json
import torch

data = torch.randn(10, 129, 2500)
with open('input.json', 'w') as f:
        json.dump({'data':data.numpy().tolist()}, f)

with open('input.json', 'r') as f:
    input_data = json.load(f)

# url = "http://127.0.0.1:8000/predict"
url = "https://8000-dep-01jm7yhdkfzc6p2fs2n4zj9m9y-d.cloudspaces.litng.ai/predict"
response = requests.post(url, json=input_data)
print(f"Status: {response.status_code}\nResponse:\n {response.text}")
# save response to a json file
with open('output.json', 'w') as f:
    json.dump(response.json(), f)
