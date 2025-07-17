import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from typing import Union, List, Callable

from .tasks import get_task_handler  # Task-specific handler

class DropWise:
    def __init__(self, model: torch.nn.Module, tokenizer: AutoTokenizer,
                 task_type: str = "sequence-classification", num_passes: int = 20, use_cuda: bool = True):
        self.model = model
        self.tokenizer = tokenizer
        self.task_type = task_type
        self.num_passes = num_passes
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.custom_metrics = {}
        self._enable_mc_dropout()
        self.task_handler = get_task_handler(task_type)

    def _enable_mc_dropout(self):
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()

    @staticmethod
    def inject_dropout(model: torch.nn.Module, p: float = 0.1):
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and 'classifier' in name:
                setattr(module, "dropout", torch.nn.Dropout(p))

    def add_metric(self, name: str, func: Callable):
        self.custom_metrics[name] = func

    def predict(self, text: Union[str, List[str]], return_all_logits: bool = False, verbose: bool = False):
        if isinstance(text, str):
            text = [text]

        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        all_logits = []

        with torch.no_grad():
            for _ in range(self.num_passes):
                outputs = self.model(**inputs)

                if self.task_type == "question-answering":
                    # For QA, keep both start and end logits
                    combined = torch.stack([outputs.start_logits, outputs.end_logits], dim=-1)
                    all_logits.append(combined)
                else:
                    all_logits.append(outputs.logits)

        stacked_logits = torch.stack(all_logits)  # [num_passes, batch, ...]
        mean_logits = stacked_logits.mean(dim=0)
        std_logits = stacked_logits.std(dim=0)

        # Handle classification tasks with MC dropout softmax aggregation
        if "classification" in self.task_type:
            all_probs = torch.stack([F.softmax(logits, dim=-1) for logits in all_logits])  # [num_passes, batch, num_classes]
            probs = all_probs.mean(dim=0)
        else:
            all_probs = None  # not used
            probs = mean_logits

        # Dispatch to task-specific output formatter
        results = self.task_handler(
            inputs=inputs,
            probs=probs,
            mean_logits=mean_logits,
            std_logits=std_logits,
            text=text,
            tokenizer=self.tokenizer,
            custom_metrics=self.custom_metrics,
            verbose=verbose
        )

        if return_all_logits:
            return {
                "results": results,
                "all_logits": stacked_logits.cpu(),
                "all_probs": all_probs.cpu() if all_probs is not None else None
            }

        return results

    def __call__(self, text: Union[str, List[str]], **kwargs):
        return self.predict(text, **kwargs)
