from torch.utils.data import Dataset
import torch


class AcountingDataset(Dataset):
    def __init__(self, dataframe, valid_indices, tokenizer, valid_categories, max_length=100):
        self.dataframe = dataframe
        self.valid_indices = valid_indices
        self.tokenizer = tokenizer
        self.valid_categories = valid_categories
        self.max_length = max_length

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index):
        valid_index = self.valid_indices[index]
        description = self.dataframe.at[valid_index, 'Memo/Description'].strip()
        inputs = self.tokenizer.encode_plus(
            description,
            None,
            pad_to_max_length=True,
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=self.max_length,
        )

        ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]

        category = self.dataframe.at[valid_index, 'Category'].strip()
        target = self.valid_categories.index(category)
        target = torch.nn.functional.one_hot(torch.tensor(target), num_classes=len(self.valid_categories))

        return {
            'valid_index': torch.tensor(valid_index, dtype=torch.long),
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'target': torch.tensor(target, dtype=torch.long),
        }
