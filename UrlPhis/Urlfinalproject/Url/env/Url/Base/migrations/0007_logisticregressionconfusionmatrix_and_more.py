# Generated by Django 4.2.7 on 2024-02-23 17:10

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Base', '0006_rename_file_testfile_testfile'),
    ]

    operations = [
        migrations.CreateModel(
            name='LogisticRegressionConfusionMatrix',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('true_positive', models.IntegerField()),
                ('true_negative', models.IntegerField()),
                ('false_positive', models.IntegerField()),
                ('false_negative', models.IntegerField()),
            ],
        ),
        migrations.RenameModel(
            old_name='analysisreport',
            new_name='LogisticEvaluationReport',
        ),
    ]
