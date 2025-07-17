from .sequence_classification import handle as sequence_classification
from .token_classification import handle as token_classification
from .question_answering import handle as question_answering
from .regression import handle as regression


def get_task_handler(task_type: str):
    if task_type == "sequence-classification":
        return sequence_classification
    elif task_type == "token-classification":
        return token_classification
    elif task_type == "question-answering":
        return question_answering
    elif task_type == "regression":
        return regression
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")
