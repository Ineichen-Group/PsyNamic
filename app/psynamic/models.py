from django.db import models

# Create your models here.


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
    is_multilable = models.BooleanField(default=False)

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
    text = models.TextField(default='') # Title + Abstract as presented in prodigy
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

        if label_class.is_multilable:
            return [p.label.name for p in predictions if p.probability >= prob_threshold]
        # if single label return the label with the highest probability
        else:
            return [max(predictions, key=lambda x: x.probability).label.name]