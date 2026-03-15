import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset

class EmotionDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'text': self.data.iloc[idx]['text'],
            'label': self.data.iloc[idx]['label']
        }

# Example usage with MELD and HEU Emotion datasets
def load_datasets():
    meld_dataset = EmotionDataset('/data/emotions/meld.csv')
    heu_dataset = EmotionDataset('/data/emotions/heu_emotion.csv')
    return meld_dataset, heu_dataset

if __name__ == "__main__":
    meld, heu = load_datasets()
    dataloader = DataLoader(meld, batch_size=16, shuffle=True)
    for batch in dataloader:
        print(batch)
