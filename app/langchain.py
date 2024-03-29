from langchain_community.llms import Ollama
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser

from pydantic import BaseModel, Field, validator
from typing import List

class CategoryScoresItem(BaseModel):
    category_name: str
    category_score: float

class CategoryScores(BaseModel):
    result: List[CategoryScoresItem]

parser = PydanticOutputParser(pydantic_object=CategoryScores)
format_instructions = parser.get_format_instructions()

system_message = r"""
    You are instructed to calculate the related score of each categories for a given article.
    Score value represents how much the article is related to a specific category.

    Here are the 22 categories.
    [redictions, Business, Finance, Crypto, Entertainment, Sports, Elections, Health & Wellness, Technology, Science, Environment, Legal & Ethics, Politics, Education, Social Issues, Culture & Arts, Personal Development, Travel, Food, Fashion, Hobbies & Crafts, Philosophy & Spirituality]

    The score value must be between 0 and 1.
    Score 0 means, article is never related to that category.
    Score 0.2 means, article is 20% related to that category.
    Score 0.4 means, article is 40% related to that category.
    ...
    Score 0.7 means, article is 70% related to that category.
    Score 1 means, article is 100% to that category.

    Provide your output in json format with the keys of "result", which is an array of items of that 22 categories (Never omit a category. It should be 22 items). Each item object contains "name" and "score" values.
    "name" refers to the category name. "score" refers to the score value of that category name.
    Remember your output must include only json format result, no explaination of response is needed.

    Additional rule:
    When calculating score, try to give higher score to the sub-category first. For example, an article can be related to Crypto and Technology category. Then, you should give higher score to Crypto category than Technology category.
"""
example_blog = r"""In Salem, PM Narendra Modi gets emotional while recalling ‘Auditor’ Ramesh. Who was he? Prime Minister Narendra Modi on Tuesday turned emotional and paused his speech briefly while remembering a late BJP leader who was hacked to death in 2013. Salem: Prime Minister Narendra Modi drinks water amid his address during a public meeting ahead of Lok Sabha elections, in Salem, Tamil Nadu, Tuesday, March 19, 2024.(PTI) Addressing a public rally in Salem, Modi recalled three personalities related to the district, including the late BJP leader KN Lakshmanan. Hindustan Times - your fastest source for breaking news! Read now. However, he turned emotional while talking about 'Auditor' Ramesh. “Today, I remember Auditor Ramesh,” PM Modi said before pausing his speech for over a minute. The crowd fell silent for a few seconds and then raised slogans in Modi's support. Resuming his speech, PM Modi said, “Unfortunately, My Ramesh of Salem is not among us.” “Ramesh worked hard for the party, day and night and he was a good orator. But he was killed,” he added as he paid tribute to the late BJP leader. Who was ‘Auditor’ Ramesh? V Ramesh, an auditor by profession, was a Salem-based state general secretary of the party. The 52-year-old BJP leader was attacked with sharp-edged weapons near his house on July 19, 2013, by unidentified assailants. The BJP leader had gone to his office to discuss party affairs around 9pm and was attacked by four persons while returning to his residence, according to police. At that time, then Gujarat chief minister Narendra Modi had called and enquired about the murder. The incident caused tensions in the region as protesters stoned five government buses and the authorities had to declare a holiday in schools. In October 2013, HT reported that Narendra Modi, who was BJP's prime ministerial face, didn't offer any praise for then Tamil Nadu CM J Jayalalithaa, during his rally in Tiruchi. TN BJP leaders said that Modi was not too happy with the state government for the lack of progress in the probe into the killing Auditor Ramesh. The prime minister also paid rich tributes to the late Lakshmanan, recalling his contributions for the party's growth in Tamil Nadu. Lakshmanan ji role in the anti-emergency movement and participation in socio-cultural activities will always be remembered. His contribution to the expansion of the BJP in the state is unforgettable. He also started many schools in the state, he added."""

llm = Ollama(
    model="llama2",
    system=system_message,
)

prompt = PromptTemplate(
    template="Here's the article content.\n{format_instructions}\n{question}",
    input_variables=["question"],
    partial_variables={"format_instructions": format_instructions},
)

chain = prompt | llm | parser

response = chain.invoke({
    "question": "Future is decentralized",
})

print(response)
