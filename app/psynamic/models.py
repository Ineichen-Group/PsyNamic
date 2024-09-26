from django.db import models
import pandas as pd


class ClassGroup(models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField()

    def __str__(self):
        return self.name


class LabelClass(models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField()
    class_group = models.ForeignKey(
        ClassGroup, on_delete=models.CASCADE, related_name='classes')
    is_multilabel = models.BooleanField(default=False)

    def __str__(self):
        return self.name

    class Meta:
        unique_together = ('name', 'class_group')


class Label(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=200)
    description = models.TextField()
    label_class = models.ForeignKey(
        LabelClass, on_delete=models.CASCADE, related_name='labels')

    def __str__(self):
        return f'{self.name} ({self.label_class.name})'

    class Meta:
        unique_together = ('name', 'label_class')


class Prediction(models.Model):
    label = models.ForeignKey(Label, on_delete=models.CASCADE)
    model_name = models.CharField(max_length=200)
    probability = models.FloatField()


class Study(models.Model):
    id = models.AutoField(primary_key=True)
    # Title + Abstract as presented in prodigy
    text = models.TextField(default='')
    predictions = models.ManyToManyField(Prediction)

    def __str__(self):
        return self.text

    def get_prediction(self, label_class: str, prob_threshold: float = 0.5) -> list[str]:
        # check if label_class exists
        try:
            label_class = LabelClass.objects.get(name=label_class)
        except LabelClass.DoesNotExist:
            raise ValueError(f'LabelClass {label_class} does not exist')
        labels = label_class.labels.all()
        # get predictions
        predictions = self.predictions.filter(label__in=labels)
        if label_class.is_multilabel:
            return [p.label.name for p in predictions if p.probability >= prob_threshold]
        # if single label return the label with the highest probability
        else:
            breakpoint()
            return [max(predictions, key=lambda x: x.probability).label.name]

    def get_pre_prob(self, label_class: str) -> dict:
        # get the prediction probabilities for a given label class
        # check if label_class exists
        try:
            label_class = LabelClass.objects.get(name=label_class)
        except LabelClass.DoesNotExist:
            raise ValueError(f'LabelClass {label_class} does not exist')

        labels = label_class.labels.all()
        # get predictions
        predictions = self.predictions.filter(label__in=labels)
        
        return {p.label.name: p.probability for p in predictions}

    @classmethod
    def get_prediction_df(self, label_classes: list[str], prob_thresholds: list[str] = None) -> pd.DataFrame:
        df = None
        # Loop through each label class and its corresponding probability threshold
        for label_class, prob_threshold in zip(label_classes, prob_thresholds):
            class_df = self.get_predictions(label_class, prob_threshold)
            if df is None:
                df = class_df
            else:
                df = df.merge(class_df, on='id')
        return df

    @classmethod
    def get_predictions(self, label_class: str, prob_threshold: str):
        "Returns prediction for all studies as a list"

        data = []
        #TODO: Speed this up, massively!
        for study in self.objects.all():
            data_instance = {
                'id': study.id,
                label_class: study.get_prediction(label_class, prob_threshold)
            }
            data.append(data_instance)

        return pd.DataFrame(data)
