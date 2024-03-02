from django.shortcuts import render
import re
import csv
import pandas as pd
import numpy as np
from io import StringIO
from geopy.geocoders import Nominatim
from django.contrib import messages
from django.conf import settings
from django.contrib.messages import get_messages
from .models import *
from django.shortcuts import redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from django.http import Http404, HttpResponse
from .Utils import *
from django.urls import reverse
import os
from django.core.files.storage import FileSystemStorage
from .models import PickleFile
import os

from .models import PickleFile

import pickle

from sklearn.metrics import precision_recall_fscore_support
from django.http import HttpResponseRedirect

global_list = []


def manual_result(request):
    return render(request, "manualresult.html")


def Home(request):
    if request.method == "POST":
        url = request.POST.get("homepage")
        if url.startswith("http://"):
            url = url[len("http://") :]
        elif url.startswith("https://"):
            url = url[len("https://") :]

        default = "lr"
        if ModelSelected.objects.exists():
            latest_model_selected = ModelSelected.objects.latest("id")
            text_field_value = latest_model_selected.text_field
            if text_field_value == "lr":
                print("lr is selected")
                res = Single_url_check_lr(url)
                if res == 0:
                    messages.success(request, "This URL is Safe")
                else:
                    messages.error(request, "This URL is Unsafe")
            else:
                print("mnb is selected")
                res = Single_url_check_Nb(url)
                if res == 0:
                    messages.success(request, "This URL is Safe")
                else:
                    messages.error(request, "This URL is Unsafe")
        else:
            res = Single_url_check_lr(url)
            if res == 0:
                messages.success(request, "This URL is Safe")
            else:
                messages.error(request, "This URL is Unsafe")
        return redirect("home")
    return render(request, "index.html")


def about_us(request):

    return render(request, "about.html")


def ourteam(request):
    return render(request, "ourteam.html")


def login_page(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        if not User.objects.filter(username=username).exists():
            messages.info(request, "Username doesnt exist")
            return redirect("/login/")
        user = authenticate(username=username, password=password)
        if user is None:
            messages.info(request, "Invalid credentials")
            return redirect("/login/")
        else:
            
            login(request, user)
            return redirect("/Userlog/")

    return render(request, "login.html")

def registration(request):
    if request.method == "POST":
        first_name = request.POST.get("first_name")
        last_name = request.POST.get("last_name")
        username = request.POST.get("username")
        password = request.POST.get("password")
        
        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists")
            return redirect("/registration/")
        
        user = User.objects.create_user(username=username, first_name=first_name, last_name=last_name, password=password)
        messages.success(request, "Registered successfully")
        return redirect("/registration/")
    
    return render(request, "registration.html")

def manual_dataentry(request):
    storage = get_messages(request)
    for message in storage:
        pass
    if request.method == "POST":
        url = request.POST.get("homepage")
        if url.startswith("http://"):
            url = url[len("http://") :]
        elif url.startswith("https://"):
            url = url[len("https://") :]
        print(url)
        
        res1 = Single_url_check_Nb(url)
        res2 = Single_url_check_lr(url)
        
        if res1 == 0:
            messages.success(request, "Result from Multinomial Naive Bayes: Safe")
        else:
            messages.error(request, "Result from Multinomial Naive Bayes: Unsafe")
        
        if res2 == 0:
            messages.success(request, "Result from Logistic Regression: Safe")
        else:
            messages.error(request, "Result from Logistic Regression: Unsafe")
        
        if "checkbox1" in request.POST:
            print("Checkbox 1 is selected.")
            mnb = "mnb"
            model_selected = ModelSelected.objects.create(text_field=str(mnb))
            print("Naive Bayes is saved")
            model_selected.save()
        else:
            print("Checkbox 1 is not selected.")

        if "checkbox2" in request.POST:
            print("Checkbox 2 is selected.")
            lr = "lr"
            model_selected = ModelSelected.objects.create(text_field=str(lr))
            print("logistic_model is saved")
            model_selected.save()
        else:
            print("Checkbox 2 is not selected.")
        
        return redirect("manual_dataentry")
    
    return render(request, "manual.html")

def User_log(request):

    return render(request, "Userlogged.html")


def DataTrain(request):
    if request.method == "POST":
        uploaded_file = request.FILES.get("file")
        if uploaded_file and uploaded_file.name.endswith(".pkl"):
            uploaded_file.name = "train_data.pkl"

            curr_dir = os.getcwd()
            folder_name = "uploads"
            upload_dir = os.path.join(curr_dir, folder_name)

            fs = FileSystemStorage(location=upload_dir)
            if fs.exists(uploaded_file.name):

                fs.delete(uploaded_file.name)

            fs.save(uploaded_file.name, uploaded_file)
            file_path = os.path.join(upload_dir, uploaded_file.name)
            value = "true"
            pickle_file_instance = PickleFile()
            pickle_file_instance.text_field = value
            pickle_file_instance.save()
            print("Pickle file uploaded successfully:", file_path)
            return render(request, "DataTrain.html", {"file_path": file_path})
        else:
            print("No valid pickle file uploaded.")
    return render(request, "DataTrain.html")


def Trainmodel(request):
    if PickleFile.objects.exists():
        Lr = LogisticRegression_Model()
        Nb = NaiveBayes_model()
        train_check = False
        if Lr and Nb:
            train_check = "Training Complete"
        while train_check:
            return render(request, "TrainedModel.html", {"train_check": train_check})
    else:
        raise Http404("No PickleFile object exists.")

    return render(request, "TrainedModel.html", {"train_check": train_check})


def Testdata(request):
    if request.method == "POST":
        uploaded_file = request.FILES.get("testFile")
        if uploaded_file and uploaded_file.name.endswith(".pkl"):
            uploaded_file.name = "test_data.pkl"

            curr_dir = os.getcwd()
            folder_name = "testfiles"
            upload_dir = os.path.join(curr_dir, folder_name)
            print(upload_dir)

            fs = FileSystemStorage(location=upload_dir)
            if fs.exists(uploaded_file.name):

                fs.delete(uploaded_file.name)

            fs.save(uploaded_file.name, uploaded_file)
            file_path = os.path.join(upload_dir, uploaded_file.name)

            print("Pickle file uploaded successfully:", file_path)
            value = "true"
            test_file_instance = TestFile()
            test_file_instance.text_field = value
            test_file_instance.save()

            return render(request, "testdata.html", {"file_path": file_path})
        else:
            print("No valid pickle file uploaded.")

    return render(request, "testdata.html")


def analytic(request):
    try:
        metrics = evaluationMetrics.objects.latest("id")
        Nb_accuracy = metrics.accuracy
        Nb_precision_class_0 = metrics.precision_class_0
        Nb_recall_class_0 = metrics.recall_class_0
        Nb_precision_class_1 = metrics.precision_class_1
        Nb_recall_class_1 = metrics.recall_class_1
        Nb_f1_class_0 = metrics.f1_class_0
        Nb_f1_class_1 = metrics.f1_class_1
        print(
            Nb_accuracy,
            Nb_precision_class_0,
            Nb_precision_class_1,
            Nb_recall_class_0,
            Nb_recall_class_1,
            Nb_f1_class_0,
            Nb_f1_class_1,
        )
        report = LogisticEvaluationReport.objects.latest("id")

        lr_accuracy = report.accuracy
        lr_precision_class_0 = report.precision_class_0
        lr_recall_class_0 = report.recall_class_0
        lr_precision_class_1 = report.precision_class_1
        lr_recall_class_1 = report.recall_class_1
        lr_f1_class_0 = report.f1_class_0
        lr_f1_class_1 = report.f1_class_1

        print(
            lr_accuracy,
            lr_precision_class_0,
            lr_precision_class_1,
            lr_recall_class_0,
            lr_recall_class_1,
            lr_f1_class_0,
            lr_f1_class_1,
        )
        confusion_matrix = ConfusionMatrix.objects.latest("id")
        true_positive = confusion_matrix.true_positive
        true_negative = confusion_matrix.true_negative
        false_positive = confusion_matrix.false_positive
        false_negative = confusion_matrix.false_negative

        lr_confusion_matrix = LogisticRegressionConfusionMatrix.objects.latest("id")
        lr_true_positive = lr_confusion_matrix.true_positive
        lr_true_negative = lr_confusion_matrix.true_negative
        lr_false_positive = lr_confusion_matrix.false_positive
        lr_false_negative = lr_confusion_matrix.false_negative

        evaluation_metrics = {
            "true_positive": 66,
            "true_negative": 33,
            "false_positive": 0,
            "false_negative": 1,
        }

        context = {
            "evaluation_metrics": evaluation_metrics,
            "saved_conf_matrix": confusion_matrix,
            "lr_saved_conf_matrix": lr_confusion_matrix,
            "Nb_accuracy": Nb_accuracy,
            "Nb_f1_class_0": Nb_f1_class_0,
            "Nb_precision_class_0": Nb_precision_class_0,
            "Nb_recall_class_0": Nb_recall_class_0,
            "Nb_precision_class_1": Nb_precision_class_1,
            "Nb_recall_class_1": Nb_recall_class_1,
            "Nb_f1_class_1": Nb_f1_class_1,

            "lr_accuracy": lr_accuracy,
            "lr_f1_class_0": lr_f1_class_0,
            "lr_precision_class_0": lr_precision_class_0,
            "lr_recall_class_0": lr_recall_class_0,
            "lr_precision_class_1": lr_precision_class_1,
            "lr_recall_class_1": lr_recall_class_1,
            "lr_f1_class_1": lr_f1_class_1,
        }

    except evaluationMetrics.DoesNotExist:
        return HttpResponse("no metrics to load")

    return render(request, "analysis.html", context)


def Testing(request):
    if TestFile.objects.exists():
        lr_test = Logistic_testing()
        Nb_test = NaiveBayes_testing()
        if Nb_test and lr_test:
            test_check = "Test Complete"
            while test_check:
                return render(request, "testing.html", {"test_check": test_check})
    else:
        raise Http404("No PickleFile object exists.")
    return render(request, "testing.html")


def simultaionResult(request):
    return render(request, "simulationresult.html")


def logout(request):
    deletetrainPickle = PickleFile.objects.all().delete()
    deletetestPickle = TestFile.objects.all().delete()
    print(deletetrainPickle, deletetestPickle)
    return redirect("home")
