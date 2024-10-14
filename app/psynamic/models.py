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
    id = models.IntegerField(primary_key=True)
    text = models.TextField(default='')
    predictions = models.ManyToManyField(Prediction)
    title = models.TextField(default='')
    abstract = models.TextField(default='')
    url = models.URLField(default='')
    keywords = models.TextField(default='')
    doi = models.CharField(max_length=200, default='')
    year = models.IntegerField(default=0)
    authors = models.TextField(default='')
    dummy = models.BooleanField(default=False)

    def __str__(self):
        return f'{self.id}: {self.title}'

    def get_prediction(self, label_class: str, prob_threshold: float = 0.1) -> list[str]:
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

    @property
    def authors_short(self):
        # convert "['Kleeblatt, J.', 'Betzler, F.', 'Kilarski, L. L.', 'Bschor, T.', 'Köhler, S.']" --> Kleeblatt et al.

        authors = self.authors.strip('][').split(', ')
        # strip ' & "
        authors = [author.strip("'") for author in authors]
        authors = [author.strip('"') for author in authors]
        first_author = authors[0].split(',')[0]
        if len(authors) > 2:
            return f'{first_author} et al.'
        elif len(authors) == 2:
            second_author = authors[1].split(',')[0]
            return f'{first_author} & {second_author}'
        else:
            return first_author

    @property
    def condition(self) -> list[str]:
        return self.get_prediction('Condition')
    
    @property
    def substance(self) -> list[str]:
        return self.get_prediction('Substances')
    
    
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
        # TODO: Speed this up, massively!
        for study in self.objects.all():
            data_instance = {
                'id': study.id,
                label_class: study.get_prediction(label_class, prob_threshold)
            }
            data.append(data_instance)

        return pd.DataFrame(data)
    
    @classmethod
    def get_most_frequent_substance_for_condition(cls, condition: str) -> str:
        """ Returns the most frequent substance for a given condition, in absolute numbers and percentage"""
        # Get all studies
        studies = cls.objects.all()
        # Filter studies by condition
        condition_studies = [study for study in studies if condition in study.condition]
        # Get all substances
        substances = [substance for study in condition_studies for substance in study.substance]
        if not substances:
            return "No substances found for the given condition."
        # Count the substances
        substance_counts = pd.Series(substances).value_counts()
        breakpoint()
        # Get the most frequent substance
        most_frequent_substance = substance_counts.idxmax()
        # Get the percentage of the most frequent substance
        percentage = substance_counts[most_frequent_substance] / len(studies) * 100
        return f'{most_frequent_substance} ({percentage:.2f}%)'

    @classmethod
    def get_most_frequent_condition_substance(cls) -> str:
        """ Returns the most frequent condition and substance combination, in absolute numbers and percentage, exlude Healthy Participants"""
        # Get all studies
        studies = cls.objects.all()
        # Get all condition-substance combinations
        condition_substance = [
            (condition, substance)
            for study in studies
            for condition in study.condition
            for substance in study.substance
        ]
        if not condition_substance:
            return "No condition-substance combinations found."
        # Count the condition-substance combinations
        condition_substance_counts = pd.Series(condition_substance).value_counts()
        # Get the most frequent condition-substance combination
        most_frequent_condition_substance = condition_substance_counts.idxmax()
        # Get the percentage of the most frequent condition-substance combination
        percentage = condition_substance_counts[most_frequent_condition_substance] / len(studies) * 100
        breakpoint()
        return f'{most_frequent_condition_substance[0]} - {most_frequent_condition_substance[1]} ({percentage:.2f}%)'

    @classmethod
    def get_distribution(cls, label_class: str) -> dict:
        """ Returns the distribution of a given label class"""
        try:
            label_class = LabelClass.objects.get(name=label_class)
        except LabelClass.DoesNotExist:
            raise ValueError(f'LabelClass {label_class} does not exist')
        labels = label_class.labels.all()
        # Get predictions
        predictions = Prediction.objects.filter(label__in=labels)
        distribution = {label.name: len([p for p in predictions if p.label == label]) for label in labels}
        return distribution