import json
import ollama

from ollama import generate

model = 'does-not-yet-exist'

system_message = r"""
    Act as a data engineer to calculate the related score of 22 categories for a given article.

    You will be provided an article content in text. Your job is to calculate the 22 categories' related score to that article. 
    Here are the 22 categories separated by comma.
    Predictions, Business, Finance, Crypto, Entertainment, Sports, Elections, Health & Wellness, Technology, Science, Environment, Legal & Ethics, Politics, Education, Social Issues, Culture & Arts, Personal Development, Travel, Food, Fashion, Hobbies & Crafts, Philosophy & Spirituality

    The score value must be between 0 and 1. 
    Score 0 means, article is never related to that category.
    Score 0.2 means, article is hardly related to that category.
    Score 0.4 means, article is few related to that category.
    ...
    Score 0.7 means, article is pretty much related to that category.
    Score 1 means, article is totally related to that category.

    Output format:
    Provide your output in json with the keys of "categories", which is an object array of that 22 categories. And that array must contain 22 objects. Each category object contains "name" and "score".
    "name" value is the category name. "score" value is the score value of that category to the article.
    Sort the array from high score to low score.

    Example response format:
    {"categories":[{"name":"Politics","score":0.95},{"name":"Elections","score":0.8},{"name":"Social Issues","score":0.7},{"name":"Legal & Ethics","score":0.6},{"name":"Personal Development","score":0.4},{"name":"Culture & Arts","score":0.3},{"name":"Education","score":0.2},{"name":"Health & Wellness","score":0.1},{"name":"Environment","score":0},{"name":"Technology","score":0},{"name":"Science","score":0},{"name":"Predictions","score":0},{"name":"Business","score":0},{"name":"Finance","score":0},{"name":"Crypto","score":0},{"name":"Entertainment","score":0},{"name":"Sports","score":0},{"name":"Technology","score":0},{"name":"Science","score":0},{"name":"Environment","score":0},{"name":"Legal & Ethics","score":0},{"name":"Education","score":0},{"name":"Travel","score":0},{"name":"Food","score":0},{"name":"Fashion","score":0},{"name":"Hobbies & Crafts","score":0},{"name":"Philosophy & Spirituality","score":0}]}

    Don't change the categories list.
    Focus on calculating the score correctly.
    Focus on the strict output format like the example response format.
"""

def get_categories(prompt):
    try:
        res = generate(
            model="llama2",
            prompt=prompt,
            stream=False,
            format="json",
            system=system_message,
            options={
                "temperature": 0.1,
            }
        )

        json_string = res["response"]
        json_object = json.loads(json_string)

        print(json_object)
        return json_object

    except ollama.ResponseError as e:
        if e.status_code == 404:
            ollama.pull(model)

        return {'Error: ': e.error}
