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

    Additional rule:
    When calculating score, try to give higher score to the sub-category first. For example, an article can be related to Crypto and Technology category. Then, you should give higher score to Crypto category than Technology category.

    Output format:
    Provide your output in json with the keys of "categories", which is an object array of that 22 categories. And that array must contain 22 objects. Each category object contains "name" and "score".
    "name" value is the category name. "score" value is the score value of that category to the article.
    Sort the array from high score to low score.

    Example response format:
    {"categories":[{"name":"category_name1","score":value1},{"name":"category_name2","score":value2},{"name":"category_name3","score":value3}, ...]}

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
