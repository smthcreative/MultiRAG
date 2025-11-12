import plotly.graph_objects as go
from datasets import Dataset


from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    MultiModalFaithfulness,
    MultiModalRelevance,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
    ContextEntityRecall,
    ResponseRelevancy,
    Faithfulness
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

from MultRAG import MultiModalRAG
from config import config

evaluator_llm = LangchainLLMWrapper(
    ChatGoogleGenerativeAI(model="gemini-2.0-flash")
)
evaluator_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
)


rag_system = MultiModalRAG()

questions = [
    "Which AI tools were used by participants at the Buildathon?",
    "What are reasoning models?",
    "How does differential privacy work in language models?",
    "Performance of top LLMs in US vs China"
]

# data_samples = {
#     "question": [],
#     "reference": ['At the Buildathon, I saw many teams execute quickly using a wide range of tools including Claude Code, GPT-5, Replit, Cursor, Windsurf, Trae, and many others.',
#                   'Reasoning models typically learn to undertake a separate process of “thinking” through their output of before they produce final response. Ant Group built a top non-reasoning model that can take similar steps as part of its immediate response.',
#                   'Differential privacy basics: An algorithm (such as training a neural network) is differentially private if it’s impossible to tell the difference between its product (the learned weights) and its product given that same dataset minus any given example.',
#                   '[]'],
#     "answer": [],
#     "contexts": [],
#     "image": []
# }
data_samples = {'question':
                    ['Which AI tools were used by participants at the Buildathon?',
                     'What are reasoning models?',
                     'How does differential privacy work in language models?',
                     'Performance of top LLMs in US vs China'],
                'reference': ['At the Buildathon, I saw many teams execute quickly using a wide range of tools including Claude Code, GPT-5, Replit, Cursor, Windsurf, Trae, and many others.',
                              'Reasoning models typically learn to undertake a separate process of “thinking” through their output of before they produce final response. Ant Group built a top non-reasoning model that can take similar steps as part of its immediate response.',
                              'Differential privacy basics: An algorithm (such as training a neural network) is differentially private if it’s impossible to tell the difference between its product (the learned weights) and its product given that same dataset minus any given example.',
                              '[]'],
                'answer': ['Participants at the Buildathon used a wide range of tools including Claude Code, GPT-5, Replit, Cursor, Windsurf, and Trae.',
                            'Reasoning models typically learn to undertake a separate process of “thinking” through their output before they produce a final response. The models make reasoning tokens visible within limits. For especially lengthy chains of thought, an unspecified smaller model summarizes reasoning tokens. During fine-tuning, models can be trained to support different reasoning levels.',
                           "Differential privacy in the context of training a neural network means that the algorithm is differentially private if it’s impossible to tell the difference between the learned weights and the learned weights derived from the same dataset minus any given example. Because the presence or absence of a single example can’t significantly change the model's weights, personal information can’t leak from the model’s weights or the model’s outputs.",
                           'I cannot answer this question. The provided context does not contain information about the performance of top LLMs in the US versus China.'],
                'contexts': [['Dear friends,\nWe’re organizing a new event called Buildathon: The Rapid Engineering Competition, to be held in the San Francisco Bay Area on Saturday, August 16, 2025! You can learn more and apply to participate here .\nAI-assisted coding is speeding up software engineering more than most people appreciate. \xa0We’re inviting the best builders from Silicon Valley and around the world to compete in person on rapidly engineering software....', 'At AI Fund and DeepLearning.AI, we pride ourselves on building and iterating quickly. At the Buildathon, I saw many teams execute quickly using a wide range of tools including Claude Code, GPT-5, Replit, Cursor, Windsurf, Trae, and many others.\nI offer my hearty congratulations to all the winners!\n1st Place : Milind Pathak, Mukul Pathak, and\xa0 Sapna Sangmitra (Team Vibe-as-a-Service), a team of three family members. They also received an award for Best Design....', 'Buildathon:\nThe Rapid Engineering Competition\nAugust 16, 2025 | buildathon.ai'],
                             ['Reasoning models typically learn to undertake a separate process of “thinking” through their output of before they produce final response. Ant Group built a top non-reasoning model that can take similar steps as part of its immediate response.\nWhat’s new: Ant Group, an affiliate of Alibaba and owner of the online payments provider Alipay, released Ling-1T, a huge, open, non-reasoning model that outperforms both open and closed counterparts....', 'We’re thinking: This lightning fast progress in weather modeling should precipitate better forecasts.\nDoes a reasoning model’s chain of thought explain how it arrived at its output? Researchers found that often it doesn’t.\nWhat’s new: When prompting large language models with multiple-choice questions, Yanda Chen and colleagues at Anthropic provided hints that pointed to the wrong answers . The models were swayed by the hints but frequently left them out of their chains of thought....', 'The models make reasoning tokens visible within limits. For especially lengthy chains of thought, an unspecified smaller model summarizes reasoning tokens.\nGiven local file access, Claude Opus 4 can create and manipulate files to store information. For instance, prompted to maintain a knowledge base while playing a Pokémon video game, the model produced a guide to the game that offered advice such as, “If stuck, try OPPOSITE approach” and “Change Y-coordinate when horizontal movement fails.”...', 'During fine-tuning, they trained the models to support three reasoning levels by inserting into prompts phrases like “Reasoning:low”.\nSimilarly, they fine-tuned them to search the web, execute Python code, and use arbitrary tools....', 'GPT-4o ($109, blue bar), and Anthropic Claude 3.5 Sonnet ($81, orange bar). "Non-reasoning models" is written above the GPT-4o'],
                             ['Differential privacy basics: An algorithm (such as training a neural network) is differentially private if it’s impossible to tell the difference between its product (the learned weights) and its product given that same dataset minus any given example. Since the presence or absence of a single example can’t significantly change the product, personal information can’t leak from the product (the model’s weights) or the consequences of the product (the model’s outputs). In training a neural network...'],
                             []],
                'image': [['https://charonhub.deeplearning.ai/content/images/2025/07/Buildathon-event-details-2.jpg'],
                          ['https://charonhub.deeplearning.ai/content/images/2025/06/unnamed--70-.jpg'],
                          [],
                          []]}

# for q in questions:
#     response = rag_system.query(q)
#
#     # Collect article snippets
#     article_contexts = [a["snippet"] for a in response.articles]
#
#     # Collect image descriptions
#     image_contexts = [img["description"] for img in response.images]
#
#     # Combine all context for RAGAS evaluation
#     combined_contexts = article_contexts + image_contexts
#
#     # combined_contexts = '\n'.join(combined_contexts)
#
#     # Append to dataset
#     data_samples["question"].append(q)
#     data_samples["answer"].append(response.answer)
#     data_samples["contexts"].append(combined_contexts)
#     data_samples["image"].append([img["image_url"] for img in response.images])
#     print(data_samples)



dataset = Dataset.from_dict(data_samples)


metrics_list = [
    MultiModalFaithfulness(),
    MultiModalRelevance(),
    LLMContextPrecisionWithoutReference(llm=evaluator_llm),
    LLMContextRecall(llm=evaluator_llm),
    ContextEntityRecall(llm=evaluator_llm),
    ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
    Faithfulness(llm=evaluator_llm)
]

score = evaluate(
    dataset,
    metrics=metrics_list,
    llm=evaluator_llm,
    embeddings=evaluator_embeddings
)


df = score.to_pandas()

fig = go.Figure(data=[go.Table(
    header=dict(values=list(df.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[df[col] for col in df.columns],
               fill_color='lavender',
               align='left'))
])

fig.update_layout(title_text="RAGAS Evaluation")
fig.write_html("ragas_evaluation_table.html")

