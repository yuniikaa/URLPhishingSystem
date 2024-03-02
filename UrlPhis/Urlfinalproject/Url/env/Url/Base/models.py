from django.db import models
from django.contrib.auth.models import User
from django.db import models
from joblib import dump, load
from io import BytesIO


class LogisticEvaluationReport(models.Model):
    accuracy = models.FloatField(null=True, blank=True)
    precision_class_0 = models.FloatField(null=True, blank=True)
    recall_class_0 = models.FloatField(null=True, blank=True)
    f1_class_0 = models.FloatField(null=True, blank=True)
    precision_class_1 = models.FloatField(null=True, blank=True)
    recall_class_1 = models.FloatField(null=True, blank=True)
    f1_class_1 = models.FloatField(null=True, blank=True)


class evaluationMetrics(models.Model):
    accuracy = models.FloatField(null=True, blank=True)
    precision_class_0 = models.FloatField(null=True, blank=True)
    recall_class_0 = models.FloatField(null=True, blank=True)
    f1_class_0 = models.FloatField(null=True, blank=True)
    precision_class_1 = models.FloatField(null=True, blank=True)
    recall_class_1 = models.FloatField(null=True, blank=True)
    f1_class_1 = models.FloatField(null=True, blank=True)


class StoredModel(models.Model):
    serialized_model = models.BinaryField()

    @property
    def model(self):
        return load(BytesIO(self.serialized_model))

    @model.setter
    def model(self, value):
        bio = BytesIO()
        dump(value, bio)
        self.serialized_model = bio.getvalue()


class StoredModel_scratch(models.Model):
    serialized_model = models.BinaryField()

    @property
    def model(self):
        return load(BytesIO(self.serialized_model))

    @model.setter
    def model(self, value):
        bio = BytesIO()
        dump(value, bio)
        self.serialized_model = bio.getvalue()


class PickleFile(models.Model):
    text_field = models.CharField(max_length=100, null=True, blank=True)


class ModelSelected(models.Model):
    text_field = models.CharField(max_length=100)


class Account(models.Model):
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)


class TestFile(models.Model):
    text_field = models.CharField(max_length=100, null=True, blank=True)


class ConfusionMatrix(models.Model):
    true_positive = models.IntegerField()
    true_negative = models.IntegerField()
    false_positive = models.IntegerField()
    false_negative = models.IntegerField()


class LogisticRegressionConfusionMatrix(models.Model):
    true_positive = models.IntegerField()
    true_negative = models.IntegerField()
    false_positive = models.IntegerField()
    false_negative = models.IntegerField()
