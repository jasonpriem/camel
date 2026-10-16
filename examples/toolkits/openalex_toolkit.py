# ========= Copyright 2023-2026 @ CAMEL-AI.org. All Rights Reserved. =========
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
# ========= Copyright 2023-2026 @ CAMEL-AI.org. All Rights Reserved. =========


from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.toolkits import OpenAlexToolkit
from camel.types import ModelPlatformType, ModelType

# No API key is needed. For your own daily budget, set OPENALEX_API_KEY to a
# free key from https://openalex.org/settings/api.
tools = OpenAlexToolkit().get_tools()

model = ModelFactory.create(
    model_platform=ModelPlatformType.DEFAULT,
    model_type=ModelType.DEFAULT,
)

camel_agent = ChatAgent(
    system_message=(
        "You are a research assistant. Find and check scholarly papers "
        "in any field with OpenAlex."
    ),
    model=model,
    tools=tools,
)

# Example 1: Search the literature
usr_msg = (
    "Find three open-access papers on kelp forest decline published since "
    "2020 and summarize what they found."
)
response = camel_agent.step(usr_msg)
print(str(response.info['tool_calls'])[:1000])
print(response.msgs[0].content)

# Example 2: Check a citation
camel_agent.reset()
response = camel_agent.step(
    "Is 10.1038/s41586-021-03819-2 a real paper? Give its title and year."
)
print(str(response.info['tool_calls'])[:1000])
print(response.msgs[0].content)
