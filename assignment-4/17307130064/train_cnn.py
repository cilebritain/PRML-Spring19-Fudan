from create_dataset import create_dataset

from cnn import cnn
from fastNLP import Trainer
from fastNLP import CrossEntropyLoss
from fastNLP import AccuracyMetric


vocab, train_data, dev_data, test_data = create_dataset()

model = cnn((len(vocab), 200), num_classes=20, padding=2, dropout=0.1)
model.cuda()

loss = CrossEntropyLoss(pred='pred', target='target')
metrics = AccuracyMetric(pred='pred', target='target')

trainer = Trainer(model=model, train_data=train_data, dev_data=dev_data, loss=loss, metrics=metrics, save_path='./', device=0)
trainer.train()