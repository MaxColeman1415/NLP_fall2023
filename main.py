# Documentation at https://github.com/thunlp/OpenPrompt

# Install the following required packages as needed
# import openprompt
import torch
# import numpy
# import scipy
# import scikit-learn
# import spacy

from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, ManualVerbalizer
from openprompt import PromptForClassification, PromptDataLoader

from datasets import load_dataset

# Step 1: Define a task
# we want to find hateful speech
classes = ["hateful", "not hateful"]

tempData = [
    InputExample(guid=0,
                 text_a="I love you",
                 ),
    InputExample(guid=1,
                 text_a="I hate you",
                 )
]


# Step 2: Define a Pre-trained Language Model (PLM)
# Can pick any model on huggingface
# Go to https://huggingface.co/docs/transformers/index

plm, tokenizer, model_config, WrapperClass = load_plm("gpt", "openai-gpt")

# Step 3: Define a template
promptTemplate = ManualTemplate(
    text='{"placeholder": "text_a"}.  It is {"mask"}',
    tokenizer=tokenizer,
)

# Step 4: Define a Verbalizer
promptVerbalizer = ManualVerbalizer(
    classes=classes,
    label_words={
        "hateful": ["bad", "hate"],
        "not hateful": ["like","good", "wonderful", "great"],
    },
    tokenizer=tokenizer,
)

# Step 5: Combine 2-4 into a PromptModel
promptModel = PromptForClassification(
    template=promptTemplate,
    plm=plm,
    verbalizer=promptVerbalizer,
)

# Step 6: Define a Dataloader
data_loader = PromptDataLoader(
    dataset=tempData,
    tokenizer=tokenizer,
    template=promptTemplate,
    tokenizer_wrapper_class=WrapperClass,
)

# Step 7: Training and Inference
promptModel.eval()
with torch.no_grad():
    for batch in data_loader:
        logits = promptModel(batch)
        preds = torch.argmax(logits, dim=-1)
        print(classes[preds])

# Citations
# @article{ding2021openprompt,
#  title={OpenPrompt: An Open-source Framework for Prompt-learning},
#  author={Ding, Ning and Hu, Shengding and Zhao, Weilin and Chen,
#  Yulin and Liu, Zhiyuan and Zheng, Hai-Tao and Sun, Maosong},
#  journal={arXiv preprint arXiv:2111.01998},
#  year={2021}
# }
